import numpy as np
from videc_uw.verification import compute_verification_metrics


def test_verification_metrics_bounds():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:25, 10:50] = 1
    metrics = compute_verification_metrics(mask)
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, (k, v)
    assert metrics["iou_l3"] >= metrics["iou_l1"]
