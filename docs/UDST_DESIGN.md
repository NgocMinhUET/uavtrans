# Uncertainty-Driven Selective Transmission (UDST)
## A Novel Framework for VCM-Oriented Structural Inspection

---

## 1. Research Problem

### 1.1 Motivation
Existing approaches transmit visual information **uniformly** regardless of detection confidence:
- High-confidence crack regions → Same transmission cost as uncertain regions
- This is wasteful when bandwidth is limited

### 1.2 Key Insight
**"Transmit MORE where UNCERTAIN, transmit LESS where CONFIDENT"**

- Regions where edge detector is confident → Minimal bits (trust edge)
- Regions where edge detector is uncertain → More bits (need cloud verification)

### 1.3 Research Question
> Given a fixed bit budget B, how should we allocate bits across regions with different uncertainty levels to maximize end-to-end task accuracy?

---

## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    UDST FRAMEWORK OVERVIEW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐     │
│  │   Image I   │───▶│  Edge Detector   │───▶│  Prediction M̂     │     │
│  │             │    │  + Uncertainty   │    │  Uncertainty U     │     │
│  └─────────────┘    └──────────────────┘    └────────┬───────────┘     │
│                                                       │                 │
│                                              ┌────────▼───────────┐     │
│                                              │  Region Classifier │     │
│                                              │  by Uncertainty    │     │
│                                              └────────┬───────────┘     │
│                                                       │                 │
│                     ┌─────────────────────────────────┼───────────┐     │
│                     │                                 │           │     │
│              ┌──────▼──────┐  ┌────────▼────────┐  ┌──▼──────────┐     │
│              │ High Conf   │  │  Medium Conf    │  │ Low Conf    │     │
│              │ U < τ_low   │  │ τ_low≤U<τ_high  │  │ U ≥ τ_high  │     │
│              └──────┬──────┘  └────────┬────────┘  └──────┬──────┘     │
│                     │                  │                  │            │
│              ┌──────▼──────┐  ┌────────▼────────┐  ┌──────▼──────┐     │
│              │  Geometry   │  │ Compressed Mask │  │ Image Patch │     │
│              │  Only (~50B)│  │    (~500B)      │  │  (~2-5KB)   │     │
│              └──────┬──────┘  └────────┬────────┘  └──────┬──────┘     │
│                     │                  │                  │            │
│                     └──────────────────┼──────────────────┘            │
│                                        │                               │
│                              ┌─────────▼─────────┐                     │
│                              │  UDST Bitstream   │                     │
│                              │  (Adaptive Rate)  │                     │
│                              └─────────┬─────────┘                     │
│                                        │                               │
│                              ══════════╪══════════  Channel            │
│                                        │                               │
│                              ┌─────────▼─────────┐                     │
│                              │   Cloud Verifier  │                     │
│                              └─────────┬─────────┘                     │
│                                        │                               │
│              ┌─────────────────────────┼─────────────────────────┐     │
│              │                         │                         │     │
│       ┌──────▼──────┐         ┌────────▼────────┐       ┌────────▼───┐ │
│       │ Trust Edge  │         │ Decode & Use    │       │ Run Cloud  │ │
│       │ Prediction  │         │ Edge Mask       │       │ Detector   │ │
│       └──────┬──────┘         └────────┬────────┘       └────────┬───┘ │
│              │                         │                         │     │
│              └─────────────────────────┼─────────────────────────┘     │
│                                        │                               │
│                              ┌─────────▼─────────┐                     │
│                              │   Final Mask M*   │                     │
│                              │   (Fused Result)  │                     │
│                              └───────────────────┘                     │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Key Components

### 3.1 Uncertainty Estimation Methods

| Method | Description | Pros | Cons |
|--------|-------------|------|------|
| **Entropy-based** | H = -Σ p·log(p) from softmax | Simple, no extra computation | May not capture epistemic uncertainty |
| **MC Dropout** | Multiple forward passes with dropout | Captures model uncertainty | Slow (multiple passes) |
| **Ensemble** | Multiple models, variance of predictions | Most accurate | Very expensive |
| **Direct Head** | Train auxiliary uncertainty head | Fast, learnable | Needs uncertainty labels |
| **Edge Density** | Uncertainty at segmentation boundaries | Domain-specific, fast | Heuristic |

**Recommended for Edge**: Entropy-based + Edge Density (fast, no extra inference)

### 3.2 Uncertainty Thresholds

```
Uncertainty Scale: [0, 1]  (0 = certain, 1 = completely uncertain)

τ_low = 0.2   → Below this: HIGH confidence
τ_high = 0.6  → Above this: LOW confidence

Region Classification:
- HIGH_CONF:   U < 0.2     → Trust edge, send geometry only
- MEDIUM_CONF: 0.2 ≤ U < 0.6 → Send compressed mask  
- LOW_CONF:    U ≥ 0.6     → Need cloud verification, send image patch
```

### 3.3 Adaptive Bit Allocation

Given:
- Total bit budget: B bits
- N regions with uncertainties: {u_1, u_2, ..., u_N}

Objective:
```
min  Σ TaskLoss_i(bits_i)
s.t. Σ bits_i ≤ B
     bits_i ≥ min_bits(confidence_class_i)
```

Simplified Policy:
```python
def allocate_bits(regions, budget_B):
    # Sort by uncertainty (highest first)
    sorted_regions = sort_by_uncertainty(regions, descending=True)
    
    remaining = budget_B
    allocation = {}
    
    for region in sorted_regions:
        if region.uncertainty >= τ_high:
            bits = IMAGE_PATCH_BITS  # ~2-5KB
        elif region.uncertainty >= τ_low:
            bits = COMPRESSED_MASK_BITS  # ~500B
        else:
            bits = GEOMETRY_ONLY_BITS  # ~50B
        
        if remaining >= bits:
            allocation[region] = bits
            remaining -= bits
        else:
            # Budget exhausted, use minimum for rest
            allocation[region] = GEOMETRY_ONLY_BITS
            remaining -= GEOMETRY_ONLY_BITS
    
    return allocation
```

### 3.4 Transmission Payload Structure

```
┌─────────────────────────────────────────────────────────────────┐
│                    UDST PACKET STRUCTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HEADER (Fixed ~32 bytes)                                       │
│  ├── event_id: 8B                                               │
│  ├── timestamp: 8B                                              │
│  ├── image_shape: 8B                                            │
│  ├── num_regions: 2B                                            │
│  └── flags: 6B                                                  │
│                                                                 │
│  REGION_TABLE (Variable, ~16B per region)                       │
│  ├── region_id: 2B                                              │
│  ├── bbox_xywh: 8B                                              │
│  ├── uncertainty: 2B (quantized)                                │
│  ├── confidence_class: 1B (0=high, 1=medium, 2=low)             │
│  ├── payload_type: 1B                                           │
│  └── payload_offset: 2B                                         │
│                                                                 │
│  PAYLOADS (Variable)                                            │
│  ├── HIGH_CONF regions: geometry only (skeleton points)         │
│  ├── MEDIUM_CONF regions: compressed mask (RLE or downsampled)  │
│  └── LOW_CONF regions: JPEG image patches                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Cloud Verification Strategy

### 4.1 Fusion Algorithm

```python
def fuse_predictions(edge_pred, edge_uncertainty, cloud_patches):
    """
    Fuse edge predictions with cloud verification results.
    
    For HIGH_CONF regions: Use edge prediction directly
    For MEDIUM_CONF regions: Use edge mask (transmitted)
    For LOW_CONF regions: Use cloud detector result on received patch
    """
    final_mask = np.zeros_like(edge_pred)
    
    for region in regions:
        if region.confidence_class == HIGH_CONF:
            # Trust edge prediction
            final_mask[region.bbox] = edge_pred[region.bbox]
            
        elif region.confidence_class == MEDIUM_CONF:
            # Use transmitted compressed mask
            final_mask[region.bbox] = region.decoded_mask
            
        else:  # LOW_CONF
            # Run cloud detector on received patch
            cloud_result = cloud_detector(region.image_patch)
            final_mask[region.bbox] = cloud_result
    
    return final_mask
```

### 4.2 When Does UDST Win?

UDST outperforms fixed-rate transmission when:
1. **Edge detector is reasonably accurate** (>70% regions are high-confidence)
2. **Uncertainty estimation is calibrated** (high uncertainty → actually wrong)
3. **Bandwidth is limited** (need to prioritize what to send)

---

## 5. Evaluation Plan

### 5.1 Baselines

| Method | Description |
|--------|-------------|
| **Full-Image JPEG** | Send entire image at quality Q |
| **ROI-JPEG** | Send cropped ROI at quality Q |
| **Fixed DEO-L2** | Current approach, fixed mask quality |
| **Fixed DEO-L3** | Current approach, high quality |
| **UDST (Proposed)** | Uncertainty-adaptive transmission |
| **Oracle** | Transmit only pixels where edge is wrong |

### 5.2 Metrics

**Primary Metrics:**
- **Rate-Accuracy Curve**: Dice/IoU vs Payload (bytes)
- **BD-Rate**: Bit savings at same accuracy

**Secondary Metrics:**
- Latency (transmission time)
- Edge computation overhead
- Uncertainty calibration (ECE)

### 5.3 Experiments

1. **Exp 1: Rate-Accuracy Trade-off**
   - Vary bit budget B
   - Measure task accuracy (Dice, IoU)
   - Compare all methods

2. **Exp 2: Uncertainty Threshold Sensitivity**
   - Vary τ_low, τ_high
   - Find optimal operating points

3. **Exp 3: Channel Conditions**
   - Acoustic poor/normal/good
   - Optical
   - How does UDST adapt?

4. **Exp 4: Edge Detector Quality**
   - Strong detector (U-Net) vs Weak detector (simple CNN)
   - When is UDST most beneficial?

---

## 6. Expected Contributions

1. **Novel Framework**: First uncertainty-driven selective transmission for VCM crack inspection

2. **Adaptive Policy**: Principled bit allocation based on uncertainty

3. **Empirical Insights**: When and why uncertainty-aware transmission helps

4. **Practical System**: Deployable on edge devices (Jetson AGX)

---

## 7. Paper Title (Draft)

**"Uncertainty-Driven Selective Evidence Transmission for Bandwidth-Constrained Visual Inspection"**

Or:

**"Send More Where Uncertain: Adaptive Evidence Transmission for VCM-Oriented Crack Verification"**
