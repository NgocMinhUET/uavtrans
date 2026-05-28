# Paper Outline: Uncertainty-Driven Selective Evidence Transmission

## Proposed Title Options

1. **"Uncertainty-Driven Selective Transmission for Bandwidth-Constrained Visual Inspection"**
2. **"Send More Where Uncertain: Adaptive Evidence Transmission for VCM Crack Verification"**
3. **"UDST: Uncertainty-Driven Selective Transmission for Edge-Cloud Crack Inspection"**

---

## Abstract (Draft)

Visual inspection systems must often operate over bandwidth-constrained links where transmitting full images is impractical. Instead of uniform transmission rates, we observe that regions with high detection confidence need minimal data (the edge prediction can be trusted), while uncertain regions require detailed information for cloud verification. This paper proposes Uncertainty-Driven Selective Transmission (UDST), a framework that adapts bit allocation based on pixel-wise uncertainty: high-confidence regions transmit only compact geometry, medium-confidence regions transmit compressed masks, and low-confidence regions transmit image patches for cloud-side re-detection. Experiments on 500 Crack500 image-mask pairs show that UDST achieves Dice = 0.XX at YY KB, reducing payload by ZZ% compared to fixed-rate DEO transmission while maintaining comparable accuracy. The key insight is that adaptive, uncertainty-aware transmission can significantly outperform uniform transmission when edge detection quality varies across regions.

---

## 1. Introduction

### 1.1 Problem Statement
- Bandwidth-constrained visual inspection (underwater, remote, mobile)
- Trade-off: transmission cost vs. task accuracy
- Current approaches transmit uniformly, ignoring detection confidence

### 1.2 Key Insight
**"Not all pixels need equal transmission effort"**
- High-confidence detections → Trust edge, minimal transmission
- Low-confidence detections → Need verification, more transmission

### 1.3 Research Question
> Given limited bandwidth, how should bits be allocated across regions with different uncertainty levels to maximize end-to-end task accuracy?

### 1.4 Contributions
1. **UDST Framework**: First uncertainty-driven selective transmission for VCM inspection
2. **Adaptive Policy**: Principled bit allocation based on confidence classes
3. **Comprehensive Evaluation**: Rate-accuracy analysis on real crack data
4. **Practical Insights**: When and why uncertainty-aware transmission helps

---

## 2. Related Work

### 2.1 Video Coding for Machines (VCM)
- Traditional coding optimizes for human viewing
- VCM optimizes for machine analysis
- **Gap**: Most VCM work focuses on features, not task-oriented evidence

### 2.2 Semantic Communication
- Task-oriented information transmission
- **Gap**: Limited work on adapting to detection confidence

### 2.3 Uncertainty Quantification in Deep Learning
- Epistemic vs. aleatoric uncertainty
- MC Dropout, ensembles, entropy-based methods
- **Gap**: Uncertainty rarely used for transmission decisions

### 2.4 Edge-Cloud Collaborative Inference
- Split computing, early exit
- **Gap**: Not applied to inspection transmission

---

## 3. Problem Formulation

### 3.1 System Model
```
Edge Device → Detector → Uncertainty Estimation → Adaptive Encoder → Channel → Cloud Verifier
```

### 3.2 Transmission Decision
- Given: Image I, Edge prediction M̂, Uncertainty map U
- Output: Bitstream with adaptive payload per region

### 3.3 Objective
```
min  E[TaskLoss(M̂_cloud, M_gt)]
s.t. Rate ≤ Budget
```

---

## 4. UDST Framework

### 4.1 Uncertainty Estimation
- **Entropy-based**: H(p) from softmax outputs
- **Edge-based**: Distance to segmentation boundaries
- **Combined**: Weighted sum

### 4.2 Confidence Classification
```
HIGH:   U < τ_low  → Trust edge, geometry only
MEDIUM: τ_low ≤ U < τ_high → Compressed mask
LOW:    U ≥ τ_high → Image patch for cloud detection
```

### 4.3 Payload Construction
| Class | Payload Type | Typical Size |
|-------|-------------|--------------|
| HIGH | Skeleton + bbox | ~50-100 B |
| MEDIUM | Downsampled mask (RLE) | ~200-500 B |
| LOW | JPEG image patch | ~1-5 KB |

### 4.4 Cloud Verification
- HIGH: Reconstruct from geometry (coarse but trusted)
- MEDIUM: Upsample compressed mask
- LOW: Run cloud detector on received patch

### 4.5 Fusion Strategy
- Overlay reconstructions with priority: LOW > MEDIUM > HIGH

---

## 5. Experimental Setup

### 5.1 Dataset
- Crack500: 500 valid image-mask pairs
- Simulated edge detector with varying confidence

### 5.2 Baselines
| Method | Description |
|--------|-------------|
| Fixed DEO L0-L3 | Previous approach, uniform levels |
| ROI-JPEG | Crop and compress ROI |
| Full-JPEG | Compress entire image |
| Direct Mask | PNG/RLE mask transmission |
| Oracle | Perfect uncertainty (upper bound) |

### 5.3 Metrics
- **Primary**: Rate (KB) vs. Task accuracy (Dice, IoU)
- **Secondary**: Confidence class distribution, threshold sensitivity

### 5.4 UDST Configurations
- τ_low ∈ {0.1, 0.2, 0.3, 0.4}
- τ_high ∈ {0.4, 0.5, 0.6, 0.7, 0.8}

---

## 6. Results

### 6.1 Rate-Accuracy Trade-off (Main Result)
**Table: Comparison of Methods**

| Method | Payload (KB) | Dice | IoU | vs UDST |
|--------|-------------|------|-----|---------|
| UDST (τ=0.2,0.6) | X.X | 0.XX | 0.XX | baseline |
| Fixed DEO-L2 | Y.Y | 0.XX | 0.XX | +Z% size |
| Fixed DEO-L3 | Z.Z | 0.XX | 0.XX | +W% size |
| ROI-JPEG Q75 | A.A | ~1.0 | ~1.0 | +B% size |

**Key Finding**: UDST achieves comparable accuracy to DEO-L3 at lower payload.

### 6.2 Confidence Distribution Analysis
- What fraction of regions fall into each class?
- How does this vary with threshold settings?

### 6.3 Threshold Sensitivity
- Plot: Accuracy vs. payload for different (τ_low, τ_high)
- Identify Pareto-optimal configurations

### 6.4 When Does UDST Win?
- Images with mixed confidence (some clear, some ambiguous)
- UDST wins less when all regions are similar confidence

---

## 7. Discussion

### 7.1 Key Insights
1. Uncertainty-aware transmission is most beneficial when edge detector confidence varies
2. Thresholds should be tuned to match edge detector calibration
3. Even simple uncertainty estimates (entropy) provide useful signal

### 7.2 Limitations
- Requires edge detector with calibrated uncertainty
- Overhead of uncertainty computation
- Simulated (not real) edge detector

### 7.3 Future Work
- Learned adaptive policies (RL)
- Real underwater imagery
- Integration with actual acoustic modems

---

## 8. Conclusion

This paper presented UDST, a framework that adapts transmission effort to detection uncertainty. Unlike fixed-rate approaches that treat all regions equally, UDST allocates more bits to uncertain regions that need cloud verification. Experiments show that this adaptive strategy achieves XX% payload reduction at comparable accuracy, demonstrating the value of uncertainty-aware transmission for bandwidth-constrained inspection.

---

## Key Figures

1. **Fig 1**: System architecture (Edge → UDST → Channel → Cloud)
2. **Fig 2**: Rate-accuracy trade-off curve
3. **Fig 3**: Confidence class distribution
4. **Fig 4**: Threshold sensitivity analysis
5. **Fig 5**: Qualitative examples (high/medium/low confidence regions)

---

## Key Tables

1. **Table I**: UDST payload structure
2. **Table II**: Main rate-accuracy comparison
3. **Table III**: Confidence class breakdown
4. **Table IV**: Threshold sensitivity

---

## Novelty Checklist

✅ **New Problem Framing**: Uncertainty-driven (not uniform) transmission
✅ **Adaptive Policy**: Bit allocation based on confidence
✅ **Principled Design**: Clear rationale for each component
✅ **Fair Comparison**: Multiple baselines with same task metrics
✅ **Practical Insights**: When/why UDST helps
