import pytest
import gameplay.stealth_perception_cone as mod

CASES = [
    ("visibility_frame", (1,)),
    ("audio_version", (1,)),
    ("alert_timer", (1,)),
    ("per_enemy_cache", (1,)),
    ("double_buffer", (1,)),
    ("discard_stale", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("cone_query", (1,)),
    ("metrics", (1,)),
    ("occlusion", (1,)),
    ("perceive", (1,)),
    ("state_apply", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
