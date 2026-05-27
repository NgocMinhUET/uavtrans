"""Underwater channel abstraction for evidence transmission experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import math


@dataclass(frozen=True)
class Channel:
    name: str
    bandwidth_bps: float
    fixed_latency_s: float = 0.0
    loss_prob: float = 0.0
    queue_delay_s: float = 0.0

    def expected_delay(self, size_bytes: float, retransmission: bool = True) -> float:
        tx = 8.0 * float(size_bytes) / max(self.bandwidth_bps, 1e-9)
        one_shot = tx + self.fixed_latency_s + self.queue_delay_s
        if retransmission:
            success_prob = max(1e-6, 1.0 - self.loss_prob)
            return one_shot / success_prob
        return one_shot


def default_channels() -> Dict[str, Channel]:
    return {
        "acoustic_poor_1kbps": Channel("acoustic_poor_1kbps", 1_000, 1.5, 0.15),
        "acoustic_normal_10kbps": Channel("acoustic_normal_10kbps", 10_000, 0.8, 0.08),
        "acoustic_good_20kbps": Channel("acoustic_good_20kbps", 20_000, 0.5, 0.03),
        "optical_5mbps": Channel("optical_5mbps", 5_000_000, 0.05, 0.02),
        "data_mule": Channel("data_mule", 20_000_000, 300.0, 0.01),
    }
