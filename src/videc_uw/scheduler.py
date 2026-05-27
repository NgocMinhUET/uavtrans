"""Rule-based channel-aware scheduler for progressive evidence levels."""
from __future__ import annotations


def choose_level(uncertainty: float, bandwidth_bps: float) -> str:
    """Choose L0/L1/L2/L3 based on uncertainty and channel bandwidth."""
    u = float(uncertainty)
    b = float(bandwidth_bps)

    # Extremely poor acoustic link: only alert/coarse evidence.
    if b <= 2_000:
        return "l1" if u < 0.85 else "l0"

    # Normal acoustic link: avoid L3, use L2 for uncertain cases.
    if b <= 10_000:
        if u < 0.45:
            return "l1"
        return "l2"

    # Better acoustic link: allow L3 for high-uncertainty cases.
    if b <= 20_000:
        if u < 0.25:
            return "l1"
        if u < 0.70:
            return "l2"
        return "l3"

    # Medium-rate channel: L2/L3.
    if b <= 1_000_000:
        return "l2" if u < 0.55 else "l3"

    # Optical/high-rate link: more verification-oriented.
    return "l2" if u < 0.65 else "l3"

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
