from videc_uw.channel import Channel
from videc_uw.scheduler import choose_level


def test_expected_delay_monotonic():
    ch = Channel("test", bandwidth_bps=1000, fixed_latency_s=1.0, loss_prob=0.0)
    assert ch.expected_delay(100) < ch.expected_delay(200)


def test_scheduler_returns_valid_level():
    assert choose_level(0.2, 1000) in {"l0", "l1", "l2", "l3"}
    assert choose_level(0.8, 10_000) == "l3"
