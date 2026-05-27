"""Rule-based channel-aware scheduler for progressive evidence levels."""
from __future__ import annotations


def choose_level(uncertainty: float, bandwidth_bps: float) -> str:
    """Choose L0/L1/L2/L3 based on uncertainty and channel bandwidth.

    This is intentionally transparent for the first ATC-style experiment. It can
    later be replaced by RL, contextual bandits, or constrained optimization.
    """
    u = float(uncertainty)
    b = float(bandwidth_bps)

    if b <= 2_000:
        return "l1" if u < 0.40 else "l0"
    if b <= 20_000:
        if u < 0.30:
            return "l1"
        if u < 0.70:
            return "l2"
        return "l3"
    if b <= 1_000_000:
        return "l2" if u < 0.60 else "l3"
    return "l2" if u < 0.45 else "l3"


METHOD_TO_SIZE = {
    "fixed_l0": "l0_bytes",
    "fixed_l1": "l1_bytes",
    "fixed_l2": "l2_bytes",
    "fixed_l3": "l3_bytes",
    "roi_jpeg": "roi_jpeg_bytes",
    "full_jpeg": "full_jpeg_bytes",
}

METHOD_TO_UTILITY = {
    "fixed_l0": "utility_l0",
    "fixed_l1": "utility_l1",
    "fixed_l2": "utility_l2",
    "fixed_l3": "utility_l3",
    "roi_jpeg": "utility_roi",
    "full_jpeg": "utility_full",
}
