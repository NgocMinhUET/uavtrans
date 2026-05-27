"""Run channel simulation over a table of evidence events."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .channel import Channel, default_channels
from .scheduler import METHOD_TO_SIZE, METHOD_TO_UTILITY, choose_level


def simulate_methods(events: pd.DataFrame, channels: Dict[str, Channel] | None = None) -> pd.DataFrame:
    if channels is None:
        channels = default_channels()
    records: List[dict] = []

    baseline_methods = list(METHOD_TO_SIZE.keys())
    for _, row in events.iterrows():
        uncertainty = float(row.get("uncertainty", 1.0 - row.get("confidence", 0.5)))
        for cname, ch in channels.items():
            for method in baseline_methods:
                size_col = METHOD_TO_SIZE[method]
                utility_col = METHOD_TO_UTILITY[method]
                size = float(row[size_col])
                records.append({
                    "event_id": row["event_id"],
                    "channel": cname,
                    "method": method,
                    "selected_level": method.replace("fixed_", ""),
                    "size_bytes": size,
                    "delay_s": ch.expected_delay(size),
                    "utility": float(row[utility_col]),
                    "uncertainty": uncertainty,
                })

            level = choose_level(uncertainty, ch.bandwidth_bps)
            size_col = f"{level}_bytes"
            utility_col = f"utility_{level}"
            size = float(row[size_col])
            records.append({
                "event_id": row["event_id"],
                "channel": cname,
                "method": "proposed_channel_aware_deo",
                "selected_level": level,
                "size_bytes": size,
                "delay_s": ch.expected_delay(size),
                "utility": float(row[utility_col]),
                "uncertainty": uncertainty,
            })
    return pd.DataFrame.from_records(records)


def summarise_results(sim_df: pd.DataFrame) -> pd.DataFrame:
    return (
        sim_df.groupby(["channel", "method"], as_index=False)
        .agg(
            avg_size_bytes=("size_bytes", "mean"),
            avg_delay_s=("delay_s", "mean"),
            avg_utility=("utility", "mean"),
            p95_delay_s=("delay_s", lambda s: s.quantile(0.95)),
            events=("event_id", "count"),
        )
        .sort_values(["channel", "avg_delay_s"])
    )
