<<<<<<< HEAD
# ViDEC-UW: Verification-Driven Evidence Transmission for Underwater AUV Inspection

This repository contains a reproducible proof-of-concept implementation for an ATC-style paper proposal:

> Instead of transmitting raw video or standard ROI frames, an underwater inspection robot transmits a structured **Defect Evidence Object (DEO)** and schedules progressive evidence levels under channel constraints.

The first research prototype focuses on the communication layer, not a full physical AUV deployment. It is designed to answer three experimental questions:

1. Does a DEO reduce transmitted data compared with full-image or ROI-image transmission?
2. Does progressive evidence preserve enough information for verification-oriented inspection?
3. Does a channel-aware scheduler reduce delay/cost compared with fixed transmission policies?

---

## 1. Evidence representation

Each event is represented as:

```text
E = {H, G, M, V, U, R}
```

| Symbol | Component | Content |
|---|---|---|
| H | Header | event ID, timestamp, sensor, pose/depth placeholders |
| G | Geometry | bbox, contour, skeleton, endpoints, branch points |
| M | Metrology | length, width, area, severity cues |
| V | Visual support | compact ROI thumbnail |
| U | Uncertainty | confidence and ambiguity score |
| R | Selective residual | unresolved/uncertain residual support |

Progressive levels:

| Level | Payload | Intended use |
|---|---|---|
| L0 | alert only | very low bandwidth warning |
| L1 | H + G + M + U | acoustic-friendly verification graph |
| L2 | L1 + V | add compact visual support |
| L3 | L2 + R | add refinement/residual for uncertain cases |

---

## 2. Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

---

## 3. Quick reproducible experiment

Generate a synthetic underwater-like crack dataset:

```bash
python scripts/generate_synthetic_dataset.py --out data/synthetic --num 120
```

Run the full evidence extraction and channel simulation pipeline:

```bash
python scripts/run_experiment.py \
  --image-dir data/synthetic/images \
  --mask-dir data/synthetic/masks \
  --out results/exp_synthetic
```

Outputs:

```text
results/exp_synthetic/evidence_events.csv
results/exp_synthetic/channel_simulation_events.csv
results/exp_synthetic/channel_simulation_summary.csv
results/exp_synthetic/fig_delay_by_method.png
results/exp_synthetic/fig_utility_cost.png
results/exp_synthetic/fig_selected_levels.png
```

---

## 4. Channel simulation model

The simulator abstracts underwater links as bandwidth-latency-loss channels:

| Channel | Bandwidth | Latency | Loss probability |
|---|---:|---:|---:|
| acoustic_poor_1kbps | 1 kbps | 1.5 s | 0.15 |
| acoustic_normal_10kbps | 10 kbps | 0.8 s | 0.08 |
| acoustic_good_20kbps | 20 kbps | 0.5 s | 0.03 |
| optical_5mbps | 5 Mbps | 0.05 s | 0.02 |
| data_mule | 20 Mbps | 300 s | 0.01 |

Transmission delay is estimated as:

```text
T_total = (8 * size_bytes / bandwidth_bps + fixed_latency + queue_delay) / (1 - loss_prob)
```

This is intentionally a network abstraction suitable for a first ATC proof-of-concept. For a Q1 extension, this can be replaced by Aqua-Sim/NS-3, a packet-level acoustic model, or a simulator-integrated channel.

---

## 5. Baselines

The simulator compares:

| Method | Description |
|---|---|
| fixed_l0 | always send L0 |
| fixed_l1 | always send L1 |
| fixed_l2 | always send L2 |
| fixed_l3 | always send L3 |
| roi_jpeg | send JPEG-compressed ROI image |
| full_jpeg | send JPEG-compressed full image |
| proposed_channel_aware_deo | choose L0/L1/L2/L3 based on uncertainty and bandwidth |

---

## 6. Scheduler

The first version uses a transparent rule-based policy:

```text
Poor acoustic channel: send L0/L1
Normal acoustic channel: send L1/L2/L3 depending on uncertainty
Optical channel: send L2/L3
```

This is deliberately simple so the paper can isolate the benefit of evidence representation and progressive transmission. Future versions may replace it with reinforcement learning, contextual bandits, or constrained optimization.

---

## 7. Edge profiling on Jetson or laptop

Channel simulation should be run on a laptop/PC. Jetson is useful for edge-feasibility profiling:

```bash
python scripts/profile_edge.py \
  --image-dir data/synthetic/images \
  --mask-dir data/synthetic/masks \
  --out results/edge_profile.csv \
  --repeats 3
```

This reports DEO construction time per event and can be used to support claims about onboard feasibility.

---

## 8. Suggested ATC experiment table

| Experiment | Question | Output |
|---|---|---|
| E1: Communication efficiency | Is DEO smaller than ROI/full-image transmission? | bytes/event, delay |
| E2: Utility-cost trade-off | Does DEO preserve verification utility? | utility vs bytes |
| E3: Channel-aware scheduling | Does adaptive level selection reduce delay? | delay/utility summary |
| E4: Level distribution | Does the scheduler behave sensibly under different channels? | selected-level histogram |
| E5: Edge feasibility | Can DEO be constructed onboard? | runtime on laptop/Jetson |

---

## 9. Repository structure

```text
src/videc_uw/evidence.py        # DEO construction: H, G, M, V, U, R
src/videc_uw/channel.py         # underwater channel abstraction
src/videc_uw/scheduler.py       # channel-aware progressive level selection
src/videc_uw/simulate.py        # experiment simulation
src/videc_uw/plots.py           # ATC-ready result figures
src/videc_uw/jetson_profile.py  # edge runtime profiling helper
scripts/generate_synthetic_dataset.py
scripts/run_experiment.py
scripts/profile_edge.py
```

---

## 10. Research note

The synthetic dataset is only for reproducibility and sanity checking. For the paper, replace or complement it with real defect datasets, underwater-degraded structural images, or HoloOcean/Unreal-generated inspection scenes.
=======
# uavtrans
>>>>>>> fd6de186c27f82e168f52c619b28bb0a764be4e5
