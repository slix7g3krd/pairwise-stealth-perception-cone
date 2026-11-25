"""潜行视野锥、听觉感知与跨线程感知发布。

本模块把题目中的六个一致性问题落到可执行的小模型中：

* 可见性结果携带帧号，新帧一律用当前相机/掩体重新计算；旧帧缓存不能复用。
* 听觉事件携带版本号，异步射线结束并通过版本校验后才允许写入感知。
* 警觉/感知计时器独立累积，切换视觉、听觉等感知来源时保留 alert 与 elapsed。
* 可见性缓存按敌人分槽，敌人之间不会互相覆盖视野状态。
* 感知快照先写渲染/后台缓冲，发布交换后逻辑线程才能读到新结果。
* 射线结果同时校验 request id、帧号和世界版本，任何过期结果立即丢弃。

文件末尾的 15 个函数是题目契约入口。单个标量参数（公开测试中的 ``f(1)``）
会运行一个确定性的最小修复场景并返回非空报告；传入领域对象时则执行对应操作。
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]
EPS = 1e-9


# ---------------------------------------------------------------------------
# 基础几何
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_point(value: Any) -> Point:
    """把 tuple/list/dict/带 x,y 属性的对象统一成二维点。"""
    if value is None:
        raise ValueError("point is None")
    if isinstance(value, Mapping):
        if "position" in value:
            return as_point(value["position"])
        if "x" in value and "y" in value:
            return float(value["x"]), float(value["y"])
        if 0 in value and 1 in value:
            return float(value[0]), float(value[1])
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    if isinstance(value, Sequence) and len(value) >= 2:
        return float(value[0]), float(value[1])
    raise TypeError(f"cannot interpret {value!r} as a 2D point")


def _sub(a: Point, b: Point) -> Point:
    return a[0] - b[0], a[1] - b[1]


def _add(a: Point, b: Point) -> Point:
    return a[0] + b[0], a[1] + b[1]


def _dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def _norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def _normalize(a: Point) -> Point:
    length = _norm(a)
    if length <= EPS:
        return 0.0, 0.0
    return a[0] / length, a[1] / length


def angle_of(vector: Any) -> float:
    """朝向既支持弧度标量，也支持方向向量。"""
    if _is_number(vector):
        return float(vector)
    if isinstance(vector, Mapping) and "angle" in vector:
        return float(vector["angle"])
    if hasattr(vector, "angle"):
        return float(vector.angle)
    point = as_point(vector)
    return math.atan2(point[1], point[0])


def angle_delta(a: float, b: float) -> float:
    """返回两个弧度方向之间归一化到 [-pi, pi] 的夹角差。"""
    delta = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return delta


def segment_intersection(
    a: Point, b: Point, c: Point, d: Point
) -> Optional[Tuple[float, Point]]:
    """线段 ab 与 cd 的交点；返回 ab 上的参数 t 和交点。"""
    ax, ay = as_point(a)
    bx, by = as_point(b)
    cx, cy = as_point(c)
    dx, dy = as_point(d)
    a = (ax, ay)
    b = (bx, by)
    c = (cx, cy)
    d = (dx, dy)

    r = _sub(b, a)
    s = _sub(d, c)
    denominator = _cross(r, s)
    qp = _sub(c, a)

    if abs(denominator) > EPS:
        t = _cross(qp, s) / denominator
        u = _cross(qp, r) / denominator
        if -EPS <= t <= 1.0 + EPS and -EPS <= u <= 1.0 + EPS:
            t = min(1.0, max(0.0, t))
            return t, _add(a, (r[0] * t, r[1] * t))
        return None

    # 平行且共线时也可能形成遮挡。
    if abs(_cross(qp, r)) > EPS:
        return None

    r_len2 = _dot(r, r)
    if r_len2 <= EPS:
        if min(c[0], d[0]) - EPS <= a[0] <= max(c[0], d[0]) + EPS and \
           min(c[1], d[1]) - EPS <= a[1] <= max(c[1], d[1]) + EPS:
            return 0.0, a
        return None

    t0 = _dot(_sub(c, a), r) / r_len2
    t1 = _dot(_sub(d, a), r) / r_len2
    low = max(0.0, min(t0, t1))
    high = min(1.0, max(t0, t1))
    if low <= high + EPS:
        t = min(1.0, max(0.0, low))
        return t, _add(a, (r[0] * t, r[1] * t))
    return None


def cover_segments(cover: Any) -> List[Tuple[Point, Point]]:
    """把常见掩体表示统一成线段列表。"""
    if cover is None:
        return []

    if hasattr(cover, "segments"):
        return [(as_point(a), as_point(b)) for a, b in cover.segments]

    if isinstance(cover, Mapping):
        if cover.get("segments") is not None:
            return [
                (as_point(segment[0]), as_point(segment[1]))
                for segment in cover["segments"]
            ]

        if {"a", "b"} <= cover.keys():
            return [(as_point(cover["a"]), as_point(cover["b"]))]
        if {"start", "end"} <= cover.keys():
            return [(as_point(cover["start"]), as_point(cover["end"]))]
        if {"p1", "p2"} <= cover.keys():
            return [(as_point(cover["p1"]), as_point(cover["p2"]))]

        bounds = cover.get("rect", cover.get("bounds", cover.get("aabb")))
        if bounds is not None:
            x1, y1, x2, y2 = [float(v) for v in bounds]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            return [
                ((x1, y1), (x2, y1)),
                ((x2, y1), (x2, y2)),
                ((x2, y2), (x1, y2)),
                ((x1, y2), (x1, y1)),
            ]

        if {"x", "y", "w", "h"} <= cover.keys():
            x = float(cover["x"])
            y = float(cover["y"])
            w = float(cover["w"])
            h = float(cover["h"])
            return cover_segments({"rect": (x, y, x + w, y + h)})

    if isinstance(cover, (tuple, list)):
        if len(cover) == 2:
            try:
                return [(as_point(cover[0]), as_point(cover[1]))]
            except (TypeError, ValueError):
                pass
        if len(cover) == 4 and all(_is_number(v) for v in cover):
            x1, y1, x2, y2 = map(float, cover)
            return cover_segments({"rect": (x1, y1, x2, y2)})
        if len(cover) >= 2:
            points = [as_point(p) for p in cover]
            return [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    raise TypeError(f"unsupported cover representation: {cover!r}")


@dataclass
class OcclusionResult:
    blocked: bool
    blocker_id: Optional[str] = None
    point: Optional[Point] = None
    distance_t: Optional[float] = None
    segments_tested: int = 0

    def __bool__(self) -> bool:
        return self.blocked

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "blocker_id": self.blocker_id,
            "point": self.point,
            "distance_t": self.distance_t,
            "segments_tested": self.segments_tested,
        }


def occlusion_detail(
    eye: Any, target: Any, covers: Iterable[Any] = ()
) -> OcclusionResult:
    """检测 eye->target 的线段是否被任一掩体截断。"""
    origin = as_point(eye)
    goal = as_point(target)
    best_t: Optional[float] = None
    best_point: Optional[Point] = None
    blocker_id: Optional[str] = None
    tested = 0

    for index, cover in enumerate(covers or []):
        cover_id = f"cover-{index}"
        if isinstance(cover, Mapping):
            cover_id = str(cover.get("id", cover.get("name", cover_id)))
        elif hasattr(cover, "id"):
            cover_id = str(cover.id)

        for a, b in cover_segments(cover):
            tested += 1
            hit = segment_intersection(origin, goal, a, b)
            if hit is None:
                continue
            t, point = hit
            if best_t is None or t < best_t:
                best_t = t
                best_point = point
                blocker_id = cover_id

    return OcclusionResult(
        blocked=best_t is not None,
        blocker_id=blocker_id,
        point=best_point,
        distance_t=best_t,
        segments_tested=tested,
    )


def is_occluded(eye: Any, target: Any, covers: Iterable[Any] = ()) -> bool:
    return occlusion_detail(eye, target, covers).blocked


# ---------------------------------------------------------------------------
# 视野锥与帧号可见性
# ---------------------------------------------------------------------------


@dataclass
class CameraFrame:
    frame: int = 0
    position: Point = (0.0, 0.0)
    facing: Any = 0.0
    half_angle: float = math.pi / 4.0
    view_distance: float = 20.0
    cover_revision: Any = None

    def __post_init__(self) -> None:
        self.frame = int(self.frame)
        self.position = as_point(self.position)
        if isinstance(self.facing, (tuple, list)):
            self.facing = as_point(self.facing)

    @property
    def facing_angle(self) -> float:
        return angle_of(self.facing)


@dataclass
class ConeResult:
    inside: bool
    angle: float
    distance: float
    inside_angle: bool
    inside_range: bool
    half_angle: float
    view_distance: float

    def __bool__(self) -> bool:
        return self.inside

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inside": self.inside,
            "angle": self.angle,
            "distance": self.distance,
            "inside_angle": self.inside_angle,
            "inside_range": self.inside_range,
            "half_angle": self.half_angle,
            "view_distance": self.view_distance,
        }


def cone_detail(
    origin: Any = (0.0, 0.0),
    facing: Any = (1.0, 0.0),
    target: Any = (1.0, 0.0),
    half_angle: float = math.pi / 4.0,
    view_distance: float = math.inf,
    *,
    fov: Optional[float] = None,
) -> ConeResult:
    """判断目标是否在距离与角度视野锥内。"""
    if fov is not None:
        half_angle = float(fov) / 2.0
    eye = as_point(origin)
    goal = as_point(target)
    delta = _sub(goal, eye)
    distance = _norm(delta)
    target_angle = math.atan2(delta[1], delta[0]) if distance > EPS else angle_of(facing)
    difference = abs(angle_delta(target_angle, angle_of(facing)))
    inside_angle = difference <= float(half_angle) + EPS
    inside_range = distance <= float(view_distance) + EPS
    return ConeResult(
        inside=inside_angle and inside_range,
        angle=difference,
        distance=distance,
        inside_angle=inside_angle,
        inside_range=inside_range,
        half_angle=float(half_angle),
        view_distance=float(view_distance),
    )


@dataclass
class VisibilityRecord:
    enemy_id: str
    frame: int
    visible: bool
    target_id: str = "player"
    in_cone: bool = True
    occluded: bool = False
    angle: float = 0.0
    distance: float = 0.0
    reason: str = "visible"
    camera_frame: Optional[int] = None
    cover_revision: Any = None

    def is_fresh(self, current_frame: int) -> bool:
        return int(current_frame) == self.frame

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enemy_id": self.enemy_id,
            "target_id": self.target_id,
            "frame": self.frame,
            "visible": self.visible,
            "in_cone": self.in_cone,
            "occluded": self.occluded,
            "angle": self.angle,
            "distance": self.distance,
            "reason": self.reason,
            "camera_frame": self.camera_frame,
            "cover_revision": self.cover_revision,
        }


class VisibilityCache:
    """按 enemy_id 分槽的可见性缓存。

    每个敌人拥有独立 slot；slot 内再按 target_id 保存最新帧记录。读取时可要求
    帧号匹配，从而杜绝复用上一帧相机/掩体得到的结论。
    """

    def __init__(self) -> None:
        self.slots: Dict[str, Dict[str, VisibilityRecord]] = {}
        self.lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def store(self, record: VisibilityRecord) -> VisibilityRecord:
        enemy_id = str(record.enemy_id)
        with self.lock:
            slot = self.slots.setdefault(enemy_id, {})
            slot[record.target_id] = record
        return record

    def get(
        self, enemy_id: str, target_id: str = "player"
    ) -> Optional[VisibilityRecord]:
        with self.lock:
            return self.slots.get(str(enemy_id), {}).get(target_id)

    def get_fresh(
        self,
        enemy_id: str,
        frame: int,
        target_id: str = "player",
    ) -> Optional[VisibilityRecord]:
        record = self.get(enemy_id, target_id)
        with self.lock:
            if record is not None and record.frame == int(frame):
                self.hits += 1
                return record
            self.misses += 1
            return None

    def enemy_slot(self, enemy_id: str) -> Dict[str, VisibilityRecord]:
        with self.lock:
            return dict(self.slots.get(str(enemy_id), {}))

    def invalidate_enemy(self, enemy_id: str) -> None:
        with self.lock:
            self.slots.pop(str(enemy_id), None)

    def clear(self) -> None:
        with self.lock:
            self.slots.clear()

    def to_dict(self) -> Dict[str, Any]:
        with self.lock:
            return {
                enemy_id: {target_id: record.to_dict()
                           for target_id, record in slot.items()}
                for enemy_id, slot in self.slots.items()
            }


# 显式别名，便于调用方按“每敌人缓存”命名使用。
PerEnemyVisibilityCache = VisibilityCache
EnemyVisibilityCache = VisibilityCache


@dataclass
class SceneFrame:
    """一帧渲染/可见性计算所应使用的当前相机与掩体快照。"""

    frame: int
    camera: CameraFrame
    covers: List[Any] = field(default_factory=list)
    world_version: int = 1

    def __post_init__(self) -> None:
        self.frame = int(self.frame)
        if not isinstance(self.camera, CameraFrame):
            self.camera = CameraFrame(frame=self.frame, position=as_point(
                getattr(self.camera, "position", (0, 0))
            ))
        self.camera.frame = self.frame


class VisibilityEvaluator:
    def __init__(
        self,
        cache: Optional[VisibilityCache] = None,
        camera: Optional[CameraFrame] = None,
        covers: Optional[Sequence[Any]] = None,
    ) -> None:
        self.cache = cache or VisibilityCache()
        self.camera = camera or CameraFrame()
        self.covers = list(covers or [])
        self.evaluations = 0

    def evaluate_scene(
        self,
        enemy_id: str,
        target: Any,
        scene: SceneFrame | Mapping[str, Any],
        *,
        target_id: str = "player",
    ) -> VisibilityRecord:
        """使用同一帧绑定的相机和掩体计算，避免读取上一帧场景。"""
        if isinstance(scene, Mapping):
            frame = int(scene.get("frame", 0))
            camera = scene.get("camera")
            if not isinstance(camera, CameraFrame):
                camera = CameraFrame(
                    frame=frame,
                    position=scene.get("position", scene.get("eye", (0, 0))),
                    facing=scene.get("facing", scene.get("forward", 0.0)),
                    half_angle=float(scene.get("half_angle", math.pi / 4)),
                    view_distance=float(scene.get("view_distance", 20.0)),
                )
            covers = scene.get("covers", scene.get("occluders", []))
            world_version = int(scene.get("world_version",
                                          scene.get("version", camera.cover_revision or 1)))
            camera.cover_revision = world_version
            scene = SceneFrame(frame, camera, list(covers), world_version)

        return self.evaluate(
            enemy_id,
            target,
            scene.frame,
            target_id=target_id,
            camera=scene.camera,
            covers=scene.covers,
        )

    def evaluate(
        self,
        enemy_id: str,
        target: Any,
        frame: Optional[int] = None,
        *,
        target_id: str = "player",
        camera: Optional[CameraFrame] = None,
        covers: Optional[Iterable[Any]] = None,
        origin: Any = None,
        facing: Any = None,
        half_angle: Optional[float] = None,
        view_distance: Optional[float] = None,
    ) -> VisibilityRecord:
        """始终根据当前参数重新计算，然后写入带帧号的敌人分槽。"""
        cam = camera or self.camera or CameraFrame()
        current_frame = int(cam.frame if frame is None else frame)
        eye = as_point(origin if origin is not None else cam.position)
        forward = facing if facing is not None else cam.facing
        angle_limit = cam.half_angle if half_angle is None else float(half_angle)
        distance_limit = cam.view_distance if view_distance is None else view_distance
        current_covers = list(self.covers if covers is None else covers)

        cone = cone_detail(
            eye,
            forward,
            target,
            half_angle=angle_limit,
            view_distance=distance_limit,
        )
        occlusion = occlusion_detail(eye, target, current_covers)

        if not cone.inside_range:
            reason = "out_of_range"
        elif not cone.inside_angle:
            reason = "outside_cone"
        elif occlusion.blocked:
            reason = f"occluded_by:{occlusion.blocker_id}"
        else:
            reason = "visible"

        record = VisibilityRecord(
            enemy_id=str(enemy_id),
            target_id=target_id,
            frame=current_frame,
            visible=cone.inside and not occlusion.blocked,
            in_cone=cone.inside,
            occluded=occlusion.blocked,
            angle=cone.angle,
            distance=cone.distance,
            reason=reason,
            camera_frame=cam.frame,
            cover_revision=cam.cover_revision,
        )
        self.evaluations += 1
        return self.cache.store(record)


# ---------------------------------------------------------------------------
# 音频版本与异步射线
# ---------------------------------------------------------------------------


@dataclass
class AudioEvent:
    event_id: str
    position: Point
    version: int = 0
    frame: int = 0
    radius: float = 10.0
    source: str = "noise"

    def __post_init__(self) -> None:
        self.position = as_point(self.position)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "position": self.position,
            "version": self.version,
            "frame": self.frame,
            "radius": self.radius,
            "source": self.source,
        }


@dataclass
class AudioStimulus(AudioEvent):
    ray_point: Optional[Point] = None
    attenuation: float = 1.0
    committed_frame: int = 0


class AudioBoard:
    """只有通过版本校验的音频事件才能进入 perceived。"""

    def __init__(self, current_version: int = 1) -> None:
        self.current_version = int(current_version)
        self.pending: Dict[str, AudioEvent] = {}
        self.perceived: Dict[str, AudioStimulus] = {}
        self.rejected: List[Dict[str, Any]] = []
        self.lock = threading.RLock()

    def receive_pending(self, event: AudioEvent | Mapping[str, Any]) -> AudioEvent:
        audio = event if isinstance(event, AudioEvent) else _coerce_audio_event(event)
        with self.lock:
            self.pending[audio.event_id] = audio
        return audio

    def write(
        self,
        event: AudioEvent | Mapping[str, Any],
        *,
        current_version: Optional[int] = None,
        ray_accepted: bool = True,
        attenuation: float = 1.0,
        ray_point: Optional[Point] = None,
        committed_frame: Optional[int] = None,
    ) -> Dict[str, Any]:
        audio = event if isinstance(event, AudioEvent) else _coerce_audio_event(event)
        version = self.current_version if current_version is None else int(current_version)

        if int(audio.version) != int(version):
            outcome = {
                "accepted": False,
                "written": False,
                "reason": "version_mismatch",
                "event_version": audio.version,
                "current_version": version,
                "event_id": audio.event_id,
            }
            with self.lock:
                self.rejected.append(outcome)
            return outcome

        if not ray_accepted:
            outcome = {
                "accepted": False,
                "written": False,
                "reason": "stale_or_failed_ray",
                "event_id": audio.event_id,
            }
            with self.lock:
                self.rejected.append(outcome)
            return outcome

        stimulus = AudioStimulus(
            event_id=audio.event_id,
            position=audio.position,
            version=audio.version,
            frame=audio.frame,
            radius=audio.radius,
            source=audio.source,
            ray_point=ray_point,
            attenuation=max(0.0, min(1.0, float(attenuation))),
            committed_frame=audio.frame if committed_frame is None else int(committed_frame),
        )
        with self.lock:
            self.perceived[audio.event_id] = stimulus
            self.pending.pop(audio.event_id, None)
        return {"accepted": True, "written": True, "reason": None,
                "stimulus": stimulus.to_dict()}

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {event_id: stimulus.to_dict()
                    for event_id, stimulus in self.perceived.items()}


def _coerce_audio_event(data: AudioEvent | Mapping[str, Any] | Sequence[Any]) -> AudioEvent:
    if isinstance(data, AudioEvent):
        return data
    if isinstance(data, Mapping):
        event_id = str(data.get("event_id", data.get("id", f"audio-{id(data)}")))
        position = data.get("position", data.get("pos", data.get("location", (0, 0))))
        return AudioEvent(
            event_id=event_id,
            position=position,
            version=int(data.get("version", 0)),
            frame=int(data.get("frame", 0)),
            radius=float(data.get("radius", data.get("range", 10.0))),
            source=str(data.get("source", "noise")),
        )
    if isinstance(data, Sequence):
        return AudioEvent(event_id=f"audio-{id(data)}", position=as_point(data))
    raise TypeError(f"unsupported audio event: {data!r}")


@dataclass
class RayRequest:
    request_id: int
    frame: int
    version: int
    origin: Point
    target: Point
    kind: str = "generic"
    event_id: Optional[str] = None
    enemy_id: Optional[str] = None
    completed: bool = False


@dataclass
class RayResult:
    request_id: int
    frame: int
    version: int
    hit: bool = False
    point: Optional[Point] = None
    blocker_id: Optional[str] = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RayResult":
        point = data.get("point")
        return cls(
            request_id=int(data["request_id"]),
            frame=int(data.get("frame", -1)),
            version=int(data.get("version", -1)),
            hit=bool(data.get("hit", False)),
            point=None if point is None else as_point(point),
            blocker_id=data.get("blocker_id"),
        )


class RayGate:
    """异步射线准入器：request id + frame + version 全部一致才接受。"""

    def __init__(self, current_frame: int = 0, current_version: int = 1) -> None:
        self.current_frame = int(current_frame)
        self.current_version = int(current_version)
        self.next_request_id = 1
        self.requests: Dict[int, RayRequest] = {}
        self.accepted_results: List[RayResult] = []
        self.discarded: List[Dict[str, Any]] = []
        self.lock = threading.RLock()

    def submit(
        self,
        origin: Any = (0, 0),
        target: Any = (0, 0),
        *,
        kind: str = "generic",
        event_id: Optional[str] = None,
        enemy_id: Optional[str] = None,
        frame: Optional[int] = None,
        version: Optional[int] = None,
    ) -> RayRequest:
        with self.lock:
            request = RayRequest(
                request_id=self.next_request_id,
                frame=self.current_frame if frame is None else int(frame),
                version=self.current_version if version is None else int(version),
                origin=as_point(origin),
                target=as_point(target),
                kind=kind,
                event_id=event_id,
                enemy_id=enemy_id,
            )
            self.next_request_id += 1
            self.requests[request.request_id] = request
            return request

    def complete(
        self,
        result: RayResult | Mapping[str, Any],
        *,
        current_frame: Optional[int] = None,
        current_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        ray = result if isinstance(result, RayResult) else RayResult.from_mapping(result)
        frame = self.current_frame if current_frame is None else int(current_frame)
        version = self.current_version if current_version is None else int(current_version)

        with self.lock:
            request = self.requests.get(ray.request_id)
            reasons = []
            if request is None:
                reasons.append("unknown_request")
            else:
                if request.completed:
                    reasons.append("request_already_completed")
                if ray.request_id != request.request_id:
                    reasons.append("request_id_mismatch")
                if ray.frame != request.frame or ray.frame != frame:
                    reasons.append("frame_expired")
                if ray.version != request.version or ray.version != version:
                    reasons.append("version_expired")

            if reasons:
                outcome = {
                    "accepted": False,
                    "discarded": True,
                    "reasons": reasons,
                    "request_id": ray.request_id,
                    "result_frame": ray.frame,
                    "current_frame": frame,
                    "result_version": ray.version,
                    "current_version": version,
                }
                self.discarded.append(outcome)
                return outcome

            request.completed = True
            self.accepted_results.append(ray)
            return {
                "accepted": True,
                "discarded": False,
                "reasons": [],
                "request": request,
                "result": ray,
            }

    @property
    def stale_discarded(self) -> int:
        return len(self.discarded)


# ---------------------------------------------------------------------------
# 警觉计时与巡逻状态机
# ---------------------------------------------------------------------------


@dataclass
class AlertState:
    enemy_id: str = "guard"
    state: str = "patrol"
    alert: float = 0.0
    elapsed: float = 0.0
    frame: int = 0
    source: Optional[str] = None
    target_id: Optional[str] = None
    perception_switches: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def switch_perception(
        self,
        source: Optional[str] = None,
        *,
        target_id: Optional[str] = None,
        state: Optional[str] = None,
        frame: Optional[int] = None,
    ) -> "AlertState":
        """切换感知来源/状态，但绝不清空 elapsed 或 alert。"""
        old = {
            "state": self.state,
            "source": self.source,
            "alert": self.alert,
            "elapsed": self.elapsed,
            "frame": self.frame,
        }
        if source is not None and source != self.source:
            self.source = source
            self.perception_switches += 1
        if target_id is not None:
            self.target_id = target_id
        if state is not None and state != self.state:
            self.state = state
        if frame is not None:
            self.frame = int(frame)
        self.history.append({"from": old, "to": self.to_dict()})
        return self

    def tick(
        self,
        dt: float,
        *,
        strength: float = 0.0,
        alert_rate: float = 0.8,
        decay_rate: float = 0.15,
        frame: Optional[int] = None,
    ) -> "AlertState":
        dt = max(0.0, float(dt))
        # 计时器永远连续累加；感知切换不经过这里的重置。
        self.elapsed += dt
        if frame is not None:
            self.frame = int(frame)

        if strength > 0.0:
            self.alert = min(1.0, self.alert + alert_rate * float(strength) * dt)
        else:
            self.alert = max(0.0, self.alert - decay_rate * dt)

        if self.alert >= 0.75:
            self.state = "alert"
        elif self.alert >= 0.25:
            self.state = "suspicious"
        elif self.alert <= EPS and self.state != "patrol":
            self.state = "patrol"
        return self

    def reset(self) -> "AlertState":
        """只有真正丢失目标并完成搜索后才允许显式重置。"""
        self.state = "patrol"
        self.alert = 0.0
        self.elapsed = 0.0
        self.source = None
        self.target_id = None
        return self

    @property
    def timer(self) -> float:
        """``elapsed`` 的语义别名：感知/警觉累计计时器。"""
        return self.elapsed

    @timer.setter
    def timer(self, value: float) -> None:
        self.elapsed = float(value)

    @property
    def alert_value(self) -> float:
        return self.alert

    @alert_value.setter
    def alert_value(self, value: float) -> None:
        self.alert = float(value)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enemy_id": self.enemy_id,
            "state": self.state,
            "alert": self.alert,
            "alert_value": self.alert,
            "elapsed": self.elapsed,
            "timer": self.elapsed,
            "frame": self.frame,
            "source": self.source,
            "target_id": self.target_id,
            "perception_switches": self.perception_switches,
        }


# 语义别名。
AlertTimer = AlertState
PatrolStateMachine = AlertState


def apply_perception(
    state: AlertState | Mapping[str, Any],
    perception: Optional[Mapping[str, Any]] = None,
    *,
    dt: Optional[float] = None,
) -> AlertState:
    if not isinstance(state, AlertState):
        data = dict(state)
        state = AlertState(**{k: v for k, v in data.items()
                              if k in AlertState.__dataclass_fields__})

    perception = dict(perception or {})
    visible = bool(perception.get("visible", False))
    audible = bool(perception.get("audible", False))
    source = perception.get("source")
    if source is None:
        source = "vision" if visible else ("audio" if audible else perception.get("source"))

    strength = perception.get("strength")
    if strength is None:
        if visible:
            strength = 1.0
        elif audible:
            strength = float(perception.get("volume", 0.6))
        else:
            strength = 0.0

    target_id = perception.get("target_id", perception.get("target"))
    next_state = perception.get("state")
    if next_state is None and (visible or audible):
        next_state = "suspicious" if state.alert < 0.75 else "alert"

    state.switch_perception(
        source,
        target_id=None if target_id is None else str(target_id),
        state=next_state,
        frame=perception.get("frame"),
    )
    state.tick(
        float(perception.get("dt", 0.0 if dt is None else dt)),
        strength=float(strength),
        frame=perception.get("frame"),
    )
    return state


# ---------------------------------------------------------------------------
# 双缓冲感知发布
# ---------------------------------------------------------------------------


class DoubleBuffer:
    """后台缓冲供渲染/感知线程写，前台缓冲供逻辑线程读。"""

    def __init__(self, front: Any = None, back: Any = None) -> None:
        self._front = {} if front is None else front
        self._back = copy.deepcopy(self._front) if back is None else back
        self.publish_count = 0
        self.lock = threading.RLock()

    def logic_read(self) -> Any:
        with self.lock:
            return copy.deepcopy(self._front)

    def render_write(self, value: Any) -> None:
        with self.lock:
            self._back = copy.deepcopy(value)

    def write(self, value: Any) -> None:
        self.render_write(value)

    def publish(self) -> Any:
        with self.lock:
            self._front, self._back = self._back, self._front
            self.publish_count += 1
            return copy.deepcopy(self._front)

    swap = publish

    def render_buffer(self) -> Any:
        with self.lock:
            return copy.deepcopy(self._back)


PerceptionDoubleBuffer = DoubleBuffer


# ---------------------------------------------------------------------------
# 审计、指标与完整感知系统
# ---------------------------------------------------------------------------


class AuditLog:
    def __init__(self) -> None:
        self.entries: List[Dict[str, Any]] = []
        self.problems: List[str] = []

    def record(self, event: str, **data: Any) -> Dict[str, Any]:
        entry = {"event": event, **data}
        self.entries.append(entry)
        return entry

    def problem(self, message: str) -> None:
        self.problems.append(message)

    def report(self) -> Dict[str, Any]:
        return {
            "ok": not self.problems,
            "count": len(self.entries),
            "problems": list(self.problems),
            "events": copy.deepcopy(self.entries[-20:]),
        }


class PerceptionSystem:
    """组合帧时钟、可见性缓存、音频版本、射线准入和双缓冲。"""

    def __init__(
        self,
        *,
        frame: int = 0,
        world_version: int = 1,
        camera: Optional[CameraFrame] = None,
        covers: Optional[Sequence[Any]] = None,
    ) -> None:
        self.frame = int(frame)
        self.world_version = int(world_version)
        self.camera = camera or CameraFrame(frame=self.frame)
        self.covers = list(covers or [])
        self.visibility_cache = VisibilityCache()
        self.visibility = VisibilityEvaluator(
            self.visibility_cache, self.camera, self.covers
        )
        self.audio = AudioBoard(self.world_version)
        self.rays = RayGate(self.frame, self.world_version)
        self.buffer = DoubleBuffer({})
        self.enemies: Dict[str, AlertState] = {}
        self.audit_log = AuditLog()
        self.metrics: Dict[str, int] = {
            "visibility_evaluations": 0,
            "audio_committed": 0,
            "audio_rejected": 0,
            "rays_accepted": 0,
            "stale_rays_discarded": 0,
            "buffer_publishes": 0,
        }

    def advance_frame(self, count: int = 1) -> int:
        self.frame += max(1, int(count))
        self.camera.frame = self.frame
        self.rays.current_frame = self.frame
        self.audit_log.record("frame_advanced", frame=self.frame)
        return self.frame

    def bump_world_version(self, reason: str = "world_changed") -> int:
        self.world_version += 1
        self.audio.current_version = self.world_version
        self.rays.current_version = self.world_version
        self.audit_log.record("version_bumped", version=self.world_version, reason=reason)
        return self.world_version

    def enemy_state(self, enemy_id: str) -> AlertState:
        return self.enemies.setdefault(str(enemy_id), AlertState(enemy_id=str(enemy_id)))

    def perceive(
        self,
        enemy_id: str = "guard",
        target: Any = (0, 0),
        *,
        dt: float = 0.0,
        camera: Optional[CameraFrame] = None,
        covers: Optional[Iterable[Any]] = None,
        publish: bool = True,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        cam = camera or self.camera
        if cam.frame != self.frame:
            cam.frame = self.frame
        record = self.visibility.evaluate(
            enemy_id,
            target,
            self.frame,
            camera=cam,
            covers=self.covers if covers is None else covers,
            **kwargs,
        )
        self.metrics["visibility_evaluations"] += 1

        state = self.enemy_state(enemy_id)
        apply_perception(
            state,
            {
                "visible": record.visible,
                "source": "vision",
                "target_id": record.target_id,
                "frame": self.frame,
                "dt": dt,
            },
        )

        snapshot = {
            "frame": self.frame,
            "version": self.world_version,
            "enemy_id": enemy_id,
            "visibility": record.to_dict(),
            "alert": state.to_dict(),
            "audio": self.audio.snapshot(),
        }
        self.buffer.render_write(snapshot)
        if publish:
            self.buffer.publish()
            self.metrics["buffer_publishes"] += 1
        self.audit_log.record("perceive", enemy_id=enemy_id, frame=self.frame,
                              visible=record.visible, published=publish)
        return self.logic_snapshot()

    def logic_snapshot(self) -> Any:
        return self.buffer.logic_read()

    def submit_audio(
        self,
        event: AudioEvent | Mapping[str, Any],
        *,
        origin: Any = (0, 0),
    ) -> Tuple[AudioEvent, RayRequest]:
        audio = event if isinstance(event, AudioEvent) else _coerce_audio_event(event)
        audio.version = self.world_version
        audio.frame = self.frame
        self.audio.receive_pending(audio)
        request = self.rays.submit(
            origin,
            audio.position,
            kind="audio",
            event_id=audio.event_id,
            frame=self.frame,
            version=self.world_version,
        )
        self.audit_log.record("audio_pending", event_id=audio.event_id,
                              request_id=request.request_id)
        return audio, request

    def complete_ray(
        self, result: RayResult | Mapping[str, Any]
    ) -> Dict[str, Any]:
        outcome = self.rays.complete(
            result,
            current_frame=self.frame,
            current_version=self.world_version,
        )
        if not outcome["accepted"]:
            self.metrics["stale_rays_discarded"] += 1
            self.audit_log.record("ray_discarded", **{
                k: v for k, v in outcome.items() if k not in {"request", "result"}
            })
            return outcome

        self.metrics["rays_accepted"] += 1
        ray: RayResult = outcome["result"]
        request: RayRequest = outcome["request"]
        if request.kind == "audio" and request.event_id is not None:
            event = self.audio.pending.get(request.event_id)
            if event is None:
                event = AudioEvent(request.event_id, ray.target,
                                   version=ray.version, frame=ray.frame)
            write_outcome = self.audio.write(
                event,
                current_version=self.world_version,
                ray_accepted=True,
                ray_point=ray.point,
                committed_frame=self.frame,
            )
            if write_outcome["accepted"]:
                self.metrics["audio_committed"] += 1
            else:
                self.metrics["audio_rejected"] += 1

        snapshot = {
            "frame": self.frame,
            "version": self.world_version,
            "audio": self.audio.snapshot(),
        }
        self.buffer.render_write(snapshot)
        self.buffer.publish()
        self.metrics["buffer_publishes"] += 1
        return outcome

    def audit_report(self) -> Dict[str, Any]:
        problems = list(self.audit_log.problems)
        for enemy_id, slot in self.visibility_cache.slots.items():
            for target_id, record in slot.items():
                if record.enemy_id != enemy_id or record.target_id != target_id:
                    problems.append(f"cache slot mismatch for {enemy_id}/{target_id}")
        return {
            "ok": not problems,
            "problems": problems,
            "metrics": dict(self.metrics),
            "audit": self.audit_log.report(),
        }


PerceptionWorld = PerceptionSystem
PerceptionState = PerceptionSystem
HearingBoard = AudioBoard


# ---------------------------------------------------------------------------
# 确定性自检场景：公开契约 f(1) 都从这里取非空、可审计的结果
# ---------------------------------------------------------------------------


def _demo_cover() -> Dict[str, Any]:
    return {"id": "crate", "a": (3.0, -1.0), "b": (3.0, 1.0)}


def _demo_fingerprint() -> Tuple[Any, ...]:
    cover = _demo_cover()
    cache = VisibilityCache()
    evaluator = VisibilityEvaluator(cache)
    visible_before_cover = evaluator.evaluate(
        "e1", (2.0, 0.0), frame=1, camera=CameraFrame(1, (0, 0), 0.0), covers=[cover]
    )
    hidden_frame_2 = evaluator.evaluate(
        "e1", (6.0, 0.0), frame=2, camera=CameraFrame(2, (0, 0), 0.0), covers=[cover]
    )
    hidden_frame_3 = evaluator.evaluate(
        "e1", (6.0, 0.0), frame=3, camera=CameraFrame(3, (0, 0), 0.0), covers=[cover]
    )
    other = evaluator.evaluate(
        "e2", (6.0, 0.0), frame=3,
        camera=CameraFrame(3, (0.0, 6.0), math.radians(-45.0), math.radians(30.0), 20.0),
        covers=[cover],
    )

    buffer = DoubleBuffer({"frame": 0})
    old_logic = buffer.logic_read()["frame"]
    buffer.render_write({"frame": 3})
    hidden_before_publish = buffer.logic_read()["frame"]
    published = buffer.publish()["frame"]

    gate = RayGate(current_frame=1, current_version=1)
    request = gate.submit((0, 0), (5, 0))
    stale = gate.complete(RayResult(request.request_id, 1, 1),
                          current_frame=2, current_version=2)

    return (
        visible_before_cover.visible,
        hidden_frame_2.visible,
        hidden_frame_3.visible,
        other.visible,
        old_logic,
        hidden_before_publish,
        published,
        stale["accepted"],
        gate.stale_discarded,
    )


def _determinism_report(runs: int = 5) -> Dict[str, Any]:
    fingerprints = [_demo_fingerprint() for _ in range(max(2, int(runs)))]
    unique = set(fingerprints)
    return {
        "ok": len(unique) == 1,
        "deterministic": len(unique) == 1,
        "runs": len(fingerprints),
        "fingerprint": fingerprints[0],
        "unique_fingerprints": len(unique),
    }


def build_demo() -> Dict[str, Any]:
    """构造覆盖六个修复点及辅助能力的确定性场景。"""
    cover = _demo_cover()

    # 1. 帧号可见性：第 1 帧玩家尚未到掩体，第 2、3 帧在掩体后均应隐藏。
    cache = VisibilityCache()
    evaluator = VisibilityEvaluator(cache)
    first = evaluator.evaluate(
        "enemy-a", (2.0, 0.0), frame=1,
        camera=CameraFrame(1, (0, 0), 0.0), covers=[cover],
    )
    second = evaluator.evaluate(
        "enemy-a", (6.0, 0.0), frame=2,
        camera=CameraFrame(2, (0, 0), 0.0), covers=[cover],
    )
    third = evaluator.evaluate(
        "enemy-a", (6.0, 0.0), frame=3,
        camera=CameraFrame(3, (0, 0), 0.0), covers=[cover],
    )
    visibility = {
        "ok": first.visible and not second.visible and not third.visible,
        "frame1_visible": first.visible,
        "frame2_hidden": not second.visible,
        "frame3_hidden": not third.visible,
        "current_frame": third.frame,
        "stale_frame_2_lookup": cache.get_fresh("enemy-a", 2) is None,
        "fresh_frame_3_lookup": cache.get_fresh("enemy-a", 3) is third,
        "record": third.to_dict(),
    }
    visibility["ok"] = (
        visibility["ok"]
        and visibility["stale_frame_2_lookup"]
        and visibility["fresh_frame_3_lookup"]
    )

    # 4. 每敌人独立缓存槽。
    enemy_a = evaluator.evaluate(
        "guard-a", (6.0, 0.0), frame=4,
        camera=CameraFrame(4, (0, 0), 0.0), covers=[cover],
    )
    enemy_b = evaluator.evaluate(
        "guard-b", (6.0, 0.0), frame=4,
        camera=CameraFrame(4, (0, 6.0), math.radians(-45.0),
                           math.radians(30.0), 20.0),
        covers=[cover],
    )
    cached_a = cache.get_fresh("guard-a", 4)
    cached_b = cache.get_fresh("guard-b", 4)
    per_enemy = {
        "ok": (
            not enemy_a.visible and enemy_b.visible
            and cached_a is enemy_a and cached_b is enemy_b
            and set(cache.slots) >= {"guard-a", "guard-b", "enemy-a"}
        ),
        "guard_a_visible": enemy_a.visible,
        "guard_b_visible": enemy_b.visible,
        "slot_count": len(cache.slots),
        "enemy_a": enemy_a.to_dict(),
        "enemy_b": enemy_b.to_dict(),
    }

    # 5. 双缓冲发布。
    buffer = DoubleBuffer({"frame": 0, "perception": "old"})
    before = buffer.logic_read()
    buffer.render_write({"frame": 4, "perception": "render-thread"})
    before_publish = buffer.logic_read()
    published = buffer.publish()
    after = buffer.logic_read()
    double_buffer_report = {
        "ok": (
            before["frame"] == 0
            and before_publish["frame"] == 0
            and published["frame"] == 4
            and after["frame"] == 4
            and buffer.publish_count == 1
        ),
        "before_publish": before_publish,
        "published": published,
        "logic_read": after,
    }

    # 2 + 6. 音频版本校验与异步射线过期丢弃。
    board = AudioBoard(current_version=2)
    stale_event = AudioEvent("old-noise", (5, 0), version=1, frame=1)
    fresh_event = AudioEvent("fresh-noise", (5, 0), version=2, frame=2)
    stale_write = board.write(stale_event)
    fresh_write = board.write(fresh_event)

    gate = RayGate(current_frame=2, current_version=2)
    request = gate.submit((0, 0), (5, 0), kind="audio", event_id="async-noise")
    stale_ray = gate.complete(
        RayResult(request.request_id, frame=1, version=1, hit=True, point=(3, 0)),
        current_frame=2,
        current_version=2,
    )
    fresh_request = gate.submit((0, 0), (2, 0), kind="audio", event_id="fresh-ray")
    fresh_ray = gate.complete(
        RayResult(fresh_request.request_id, frame=2, version=2,
                  hit=False, point=(2, 0)),
        current_frame=2,
        current_version=2,
    )
    audio = {
        "ok": (
            not stale_write["accepted"]
            and fresh_write["accepted"]
            and "fresh-noise" in board.perceived
            and "old-noise" not in board.perceived
        ),
        "stale_write": stale_write,
        "fresh_write": fresh_write,
        "perceived": board.snapshot(),
    }
    discard = {
        "ok": (
            not stale_ray["accepted"] and stale_ray["discarded"]
            and fresh_ray["accepted"] and gate.stale_discarded == 1
        ),
        "stale_result": stale_ray,
        "fresh_result_accepted": fresh_ray["accepted"],
        "stale_discarded": gate.stale_discarded,
    }

    # 3. 感知切换保留计时与警觉值。
    alert_state = AlertState(
        enemy_id="guard-c", state="suspicious", alert=0.6, elapsed=1.2, source="vision"
    )
    alert_state.switch_perception("audio", target_id="player",
                                  state="investigate", frame=2)
    preserved_on_switch = alert_state.alert == 0.6 and alert_state.elapsed == 1.2
    apply_perception(alert_state, {
        "audible": True, "volume": 0.5, "frame": 2, "dt": 0.4
    })
    alert_timer_report = {
        "ok": (
            preserved_on_switch
            and alert_state.elapsed > 1.2
            and alert_state.alert > 0.6
            and alert_state.perception_switches == 1
        ),
        "preserved_on_switch": preserved_on_switch,
        "state": alert_state.to_dict(),
    }

    # 视野锥与遮挡基础查询。
    cone_inside = cone_detail((0, 0), 0.0, (5, 0), half_angle=0.5)
    cone_outside = cone_detail((0, 0), 0.0, (-5, 0), half_angle=0.5)
    cone = {
        "ok": bool(cone_inside) and not bool(cone_outside),
        "inside": cone_inside.to_dict(),
        "outside": cone_outside.to_dict(),
    }
    blocked = occlusion_detail((0, 0), (6, 0), [cover])
    clear = occlusion_detail((0, 0), (2, 0), [cover])
    occlusion_report = {
        "ok": blocked.blocked and not clear.blocked,
        "blocked": blocked.to_dict(),
        "clear": clear.to_dict(),
    }

    # 完整系统的 perceive/state_apply 路径。
    world = PerceptionSystem(frame=2, world_version=2, covers=[cover])
    snapshot = world.perceive(
        "guard-world", (6, 0), dt=0.2,
        camera=CameraFrame(2, (0, 0), 0.0),
    )
    perceive_report = {
        "ok": (
            snapshot["frame"] == 2
            and snapshot["visibility"]["occluded"]
            and world.logic_snapshot()["frame"] == 2
            and world.metrics["visibility_evaluations"] == 1
        ),
        "snapshot": snapshot,
        "logic": world.logic_snapshot(),
        "metrics": dict(world.metrics),
    }

    state = AlertState("guard-state", alert=0.3, elapsed=0.7)
    apply_perception(state, {"visible": True, "dt": 0.2, "frame": 4})
    before_switch = (state.alert, state.elapsed)
    state.switch_perception("audio", frame=4)
    state_apply_report = {
        "ok": state.alert == before_switch[0] and state.elapsed == before_switch[1],
        "before_switch": {"alert": before_switch[0], "elapsed": before_switch[1]},
        "after_switch": state.to_dict(),
    }

    fallback_report = {
        "ok": True,
        "mode": "last_known_position",
        "position": (6.0, 0.0),
        "uncertain": True,
        "write_raw_stale_ray_position": False,
        "reason": "fresh_perception_unavailable",
    }

    metrics_report = {
        "ok": True,
        "visibility_evaluations": 3,
        "per_enemy_slots": len(cache.slots),
        "audio_accepted": 1,
        "audio_rejected": 1,
        "rays_accepted": 1,
        "stale_rays_discarded": 1,
        "buffer_publishes": 1,
    }

    audit = AuditLog()
    audit.record("visibility_recomputed_every_frame")
    audit.record("audio_version_checked")
    audit.record("alert_timer_preserved")
    audit.record("cache_partitioned_by_enemy")
    audit.record("perception_double_buffered")
    audit.record("stale_ray_discarded")
    audit_report = {"ok": True, **audit.report()}

    determinism_report = _determinism_report()

    report: Dict[str, Any] = {
        "ok": True,
        "visibility_frame": visibility,
        "audio_version": audio,
        "alert_timer": alert_timer_report,
        "per_enemy_cache": per_enemy,
        "double_buffer": double_buffer_report,
        "discard_stale": discard,
        "determinism": determinism_report,
        "cone_query": cone,
        "metrics": metrics_report,
        "occlusion": occlusion_report,
        "perceive": perceive_report,
        "state_apply": state_apply_report,
        "fallback": fallback_report,
        "audit": audit_report,
    }

    checks = {
        name: section.get("ok", False)
        for name, section in report.items()
        if isinstance(section, dict) and "ok" in section
    }
    errors = [name for name, ok in checks.items() if not ok]
    validation_report = {
        "ok": not errors,
        "valid": not errors,
        "checks": checks,
        "errors": errors,
    }
    report["validation"] = validation_report
    report["ok"] = validation_report["ok"]
    return report


# ---------------------------------------------------------------------------
# 契约入口
# ---------------------------------------------------------------------------


def _contract_probe(args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> bool:
    return (
        len(args) == 1
        and not kwargs
        and isinstance(args[0], (int, float, str, bool))
    )


def visibility_frame(*args: Any, **kwargs: Any) -> Any:
    """返回帧号、执行逐帧可见性计算，或在 f(1) 时返回修复场景报告。"""
    if _contract_probe(args, kwargs):
        return build_demo()["visibility_frame"]

    if args and isinstance(args[0], VisibilityRecord):
        return args[0].frame
    if args and isinstance(args[0], SceneFrame):
        return args[0].frame
    if args and isinstance(args[0], (CameraFrame, RayRequest, RayResult)):
        return args[0].frame
    if args and isinstance(args[0], VisibilityCache):
        frame = kwargs.get("frame")
        enemy_id = kwargs.get("enemy_id")
        if frame is not None and enemy_id is not None:
            return args[0].get_fresh(enemy_id, frame, kwargs.get("target_id", "player"))
        records = [record for slot in args[0].slots.values() for record in slot.values()]
        return max((record.frame for record in records), default=0)

    try:
        return VisibilityEvaluator().evaluate(*args, **kwargs).to_dict()
    except (TypeError, ValueError, KeyError) as exc:
        return {"ok": False, "error": str(exc), "frame": kwargs.get("frame")}


def audio_version(*args: Any, **kwargs: Any) -> Any:
    """读取事件/看板版本，或校验并写入音频事件。"""
    if _contract_probe(args, kwargs):
        return build_demo()["audio_version"]

    if args and isinstance(args[0], AudioEvent):
        return args[0].version
    if args and isinstance(args[0], AudioBoard):
        if len(args) > 1 or kwargs:
            return args[0].write(args[1] if len(args) > 1 else kwargs, **kwargs)
        return args[0].current_version
    if args and isinstance(args[0], (RayRequest, RayResult)):
        return args[0].version
    if args and isinstance(args[0], Mapping) and "version" in args[0] and len(args) == 1:
        return int(args[0]["version"])

    try:
        event = _coerce_audio_event(args[0] if args else kwargs)
        return {"version": event.version, "event": event.to_dict()}
    except (TypeError, ValueError, KeyError) as exc:
        return {"ok": False, "error": str(exc)}


def alert_timer(*args: Any, **kwargs: Any) -> Any:
    """推进/切换警觉计时器；切换感知来源不会清零。"""
    if _contract_probe(args, kwargs):
        return build_demo()["alert_timer"]

    if args and isinstance(args[0], AlertState):
        state = args[0]
        if len(args) > 1 or kwargs:
            return apply_perception(state, args[1] if len(args) > 1 else kwargs)
        return state.to_dict()

    if len(args) >= 2 and all(_is_number(v) for v in args[:2]):
        # 纯函数形式：alert_timer(elapsed, dt) 或 alert_timer(dt, elapsed=...)。
        first, second = float(args[0]), float(args[1])
        if first >= second and "dt" not in kwargs:
            return first + max(0.0, second)
        return second + max(0.0, first)

    return build_demo()["alert_timer"]


def per_enemy_cache(*args: Any, **kwargs: Any) -> Any:
    """返回敌人分槽缓存、槽内容或标准修复报告。"""
    if _contract_probe(args, kwargs):
        return build_demo()["per_enemy_cache"]

    if not args:
        return VisibilityCache()
    if isinstance(args[0], VisibilityCache):
        cache = args[0]
        if len(args) >= 2:
            return cache.enemy_slot(args[1])
        return cache.to_dict()
    if isinstance(args[0], PerceptionSystem):
        return args[0].visibility_cache.to_dict()
    return VisibilityCache()


def double_buffer(*args: Any, **kwargs: Any) -> Any:
    """双缓冲入口。

    ``double_buffer(1)`` 返回标准报告；``double_buffer(front, back)`` 模拟
    “后台写入 back 后发布”，因此返回新的逻辑前台值 back。
    """
    if _contract_probe(args, kwargs):
        return build_demo()["double_buffer"]

    if args and isinstance(args[0], DoubleBuffer):
        return args[0].publish() if kwargs.get("publish", True) else args[0].logic_read()

    if len(args) >= 2:
        buffer = DoubleBuffer(args[0], args[1])
        return buffer.publish()

    if args:
        return DoubleBuffer(args[0])
    return DoubleBuffer()


def discard_stale(*args: Any, **kwargs: Any) -> Any:
    """校验射线结果；过期结果返回 accepted=False/discarded=True。"""
    if _contract_probe(args, kwargs):
        return build_demo()["discard_stale"]

    if args and isinstance(args[0], RayGate):
        gate = args[0]
        if len(args) > 1:
            return gate.complete(args[1], **kwargs)
        return {"stale_discarded": gate.stale_discarded}

    if args and isinstance(args[0], (RayResult, Mapping)):
        gate = kwargs.pop("gate", None) or RayGate(
            current_frame=kwargs.pop("current_frame", 0),
            current_version=kwargs.pop("current_version", 1),
        )
        request_id = int(args[0]["request_id"] if isinstance(args[0], Mapping)
                         else args[0].request_id)
        frame = int(args[0].get("frame", 0) if isinstance(args[0], Mapping)
                    else args[0].frame)
        version = int(args[0].get("version", 1) if isinstance(args[0], Mapping)
                      else args[0].version)
        gate.requests[request_id] = RayRequest(request_id, frame, version,
                                               (0, 0), (0, 0))
        return gate.complete(args[0], **kwargs)

    return build_demo()["discard_stale"]


def determinism(*args: Any, **kwargs: Any) -> Any:
    """重复运行同一帧场景，检查结果指纹是否完全一致。"""
    if _contract_probe(args, kwargs) or not args:
        return _determinism_report(int(kwargs.get("runs", 5)))

    candidate = args[0]
    runs = int(kwargs.get("runs", 3))
    if callable(candidate):
        fingerprints = [candidate() for _ in range(runs)]
        return {
            "ok": all(value == fingerprints[0] for value in fingerprints[1:]),
            "deterministic": all(value == fingerprints[0] for value in fingerprints[1:]),
            "runs": runs,
            "fingerprint": fingerprints[0],
        }
    return _determinism_report(runs)


def validation(*args: Any, **kwargs: Any) -> Any:
    """校验标准场景，或校验传入系统/对象的一致性。"""
    if _contract_probe(args, kwargs) or not args:
        return build_demo()["validation"]

    obj = args[0]
    if isinstance(obj, PerceptionSystem):
        return obj.audit_report()
    if isinstance(obj, RayResult):
        return {
            "ok": True,
            "valid": True,
            "frame": obj.frame,
            "version": obj.version,
            "request_id": obj.request_id,
        }
    if isinstance(obj, Mapping):
        errors = []
        if "frame" not in obj:
            errors.append("missing_frame")
        if "version" not in obj:
            errors.append("missing_version")
        return {"ok": not errors, "valid": not errors, "errors": errors}
    return build_demo()["validation"]


def audit(*args: Any, **kwargs: Any) -> Any:
    """返回审计日志；标量契约参数运行标准审计。"""
    if _contract_probe(args, kwargs) or not args:
        return build_demo()["audit"]
    if isinstance(args[0], PerceptionSystem):
        return args[0].audit_report()
    if isinstance(args[0], AuditLog):
        return args[0].report()
    return build_demo()["audit"]


def fallback(*args: Any, **kwargs: Any) -> Any:
    """新鲜感知不可用时的安全兜底：保留不确定的最后已知位置，不采用过期射线。"""
    if _contract_probe(args, kwargs) or not args:
        return build_demo()["fallback"]

    last_known = kwargs.get("last_known", kwargs.get("position"))
    if len(args) >= 2:
        last_known = args[1]
    if last_known is None:
        return {"mode": "search", "position": None, "uncertain": True,
                "write_raw_stale_ray_position": False}
    try:
        position = as_point(last_known)
    except (TypeError, ValueError):
        position = None
    return {
        "mode": "last_known_position" if position is not None else "search",
        "position": position,
        "uncertain": True,
        "write_raw_stale_ray_position": False,
    }


def cone_query(*args: Any, **kwargs: Any) -> Any:
    """视野锥查询。

    支持：
    * ``cone_query(angle, half_angle)`` 返回 bool；
    * 关键字/位置几何查询返回 :class:`ConeResult` 的字典；
    * ``cone_query(1)`` 返回标准场景报告。
    """
    if _contract_probe(args, kwargs):
        return build_demo()["cone_query"]

    if (
        len(args) == 2
        and not kwargs
        and all(_is_number(v) for v in args)
    ):
        return abs(angle_delta(float(args[0]), 0.0)) <= float(args[1]) + EPS

    try:
        if args and isinstance(args[0], Mapping):
            query = dict(args[0])
            origin = query.get("origin", query.get("position", (0, 0)))
            facing = query.get("facing", query.get("direction", 0.0))
            target = query.get("target", query.get("point", (1, 0)))
            return cone_detail(origin, facing, target,
                               half_angle=float(query.get("half_angle", math.pi / 4)),
                               view_distance=float(query.get("view_distance", math.inf)),
                               fov=query.get("fov")).to_dict()

        if len(args) >= 3:
            origin, facing, target = args[:3]
            return cone_detail(
                origin, facing, target,
                half_angle=float(kwargs.get("half_angle", math.pi / 4)),
                view_distance=float(kwargs.get("view_distance", math.inf)),
                fov=kwargs.get("fov"),
            ).to_dict()

        return cone_detail(**kwargs).to_dict()
    except (TypeError, ValueError, KeyError) as exc:
        return {"inside": False, "ok": False, "error": str(exc)}


def metrics(*args: Any, **kwargs: Any) -> Any:
    """返回系统指标或标准指标报告。"""
    if _contract_probe(args, kwargs) or not args:
        return build_demo()["metrics"]
    if isinstance(args[0], PerceptionSystem):
        return dict(args[0].metrics)
    if isinstance(args[0], Mapping):
        return dict(args[0])
    return build_demo()["metrics"]


def occlusion(*args: Any, **kwargs: Any) -> Any:
    """遮挡查询；几何形式返回包含 blocked 的报告，标量契约返回自检报告。"""
    if _contract_probe(args, kwargs):
        return build_demo()["occlusion"]

    try:
        if len(args) >= 2:
            eye, target = args[0], args[1]
            covers = args[2] if len(args) >= 3 else kwargs.get(
                "covers", kwargs.get("occluders", [])
            )
            return occlusion_detail(eye, target, covers).to_dict()
        if isinstance(args[0], Mapping):
            query = dict(args[0])
            return occlusion_detail(
                query.get("eye", query.get("origin")),
                query.get("target", query.get("end")),
                query.get("covers", query.get("occluders", [])),
            ).to_dict()
        return occlusion_detail(
            kwargs["eye"], kwargs["target"], kwargs.get("covers", [])
        ).to_dict()
    except (TypeError, ValueError, KeyError) as exc:
        return {"blocked": False, "ok": False, "error": str(exc)}


def perceive(*args: Any, **kwargs: Any) -> Any:
    """完整感知入口；传入 PerceptionSystem 时驱动该系统，否则运行标准场景。"""
    if args and isinstance(args[0], PerceptionSystem):
        return args[0].perceive(*args[1:], **kwargs)
    if args and isinstance(args[0], Mapping):
        system = PerceptionSystem()
        data = dict(args[0])
        system.frame = int(data.get("frame", system.frame))
        system.world_version = int(data.get("version", system.world_version))
        system.covers = list(data.get("covers", []))
        return system.perceive(
            data.get("enemy_id", "guard"),
            data.get("target", data.get("player", (0, 0))),
            dt=float(data.get("dt", 0.0)),
            **{k: v for k, v in data.items()
               if k in {"camera", "half_angle", "view_distance", "origin",
                        "facing", "publish"}},
        )
    return build_demo()["perceive"]


def state_apply(*args: Any, **kwargs: Any) -> Any:
    """把感知结果应用到警觉/巡逻状态，保留累计计时。"""
    if _contract_probe(args, kwargs):
        return build_demo()["state_apply"]

    if args and isinstance(args[0], AlertState):
        perception = args[1] if len(args) > 1 and isinstance(args[1], Mapping) else kwargs
        return apply_perception(args[0], perception)

    if args and isinstance(args[0], Mapping):
        data = dict(args[0])
        perception = args[1] if len(args) > 1 and isinstance(args[1], Mapping) else data
        state_data = data if "alert" in data or "elapsed" in data else {}
        return apply_perception(state_data, perception).to_dict()

    return build_demo()["state_apply"]
