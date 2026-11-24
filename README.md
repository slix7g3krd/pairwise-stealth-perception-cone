# pairwise-stealth-perception-cone

## 领域

游戏、图形与多媒体

## 主题

潜行视野锥与听觉感知

## 说明

可见性帧号、听觉版本校验与感知双缓冲。

初始状态下 `gameplay/stealth_perception_cone.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
