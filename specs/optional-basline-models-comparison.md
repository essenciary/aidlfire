# Optional: Baseline Models Comparison

**Note:** This document contains optional baseline model implementations for comparison with the deep learning approach. Baseline models are not required for the capstone project but can provide valuable context and demonstrate the value of deep learning.

---

## Baseline Models

Baseline models are **simple, non-deep-learning approaches** that provide a performance benchmark. They help answer the critical question: "Does deep learning actually add value, or could we solve this problem with simpler methods?"

**Why Implement Baselines?**

1. **Demonstrate value:** Show that deep learning outperforms traditional methods (or identify when it doesn't)
2. **Set expectations:** Understand what performance is achievable with simple methods
3. **Academic rigor:** Standard practice in research to compare against baselines
4. **Debugging:** If deep learning performs worse than baselines, indicates a problem (bug, wrong data, etc.)
5. **Capstone requirement:** Demonstrates understanding of the problem domain and appropriate method selection

**Baseline Selection Rationale:**

For wildfire detection, we need baselines that:
- Are **simple to implement** (don't require weeks of work)
- Are **standard in remote sensing** (well-established methods)
- Use **similar inputs** (same satellite imagery, same bands)
- Provide **meaningful comparison** (not trivial to beat)

---

## Simple Baseline Models

### Baseline 1: Threshold-Based Method

**What it is:**
- Simple rule-based approach using spectral thresholds
- Based on physical properties: fires emit strongly in SWIR (Short-Wave Infrared) bands
- Standard approach in remote sensing for fire detection

**How it works:**
- Apply threshold to SWIR bands (B11, B12) - fires have high SWIR values
- Optionally combine with NBR (Normalized Burn Ratio) for burned area detection
- Binary classification: pixel above threshold = fire, below = no fire

**Why This Baseline?**

**Advantages:**
- ✅ **Simple and fast:** No training required, instant predictions
- ✅ **Interpretable:** Clear physical rationale (fires emit in SWIR)
- ✅ **Standard method:** Widely used in remote sensing literature
- ✅ **Baseline for comparison:** If deep learning can't beat this, something is wrong
- ✅ **Quick to implement:** Can be done in hours, not days

**Limitations (Why Deep Learning Should Beat It):**
- ⚠️ **Fixed thresholds:** Cannot adapt to different conditions (atmosphere, sensor calibration, time of day)
- ⚠️ **No spatial context:** Each pixel classified independently (ignores neighboring pixels)
- ⚠️ **False positives:** Many non-fire objects have high SWIR (hot surfaces, urban areas, clouds)
- ⚠️ **No learning:** Cannot learn complex patterns or adapt to data distribution
- ⚠️ **Single-band focus:** Only uses SWIR bands, ignores other spectral information

**Expected Performance:**
- **Precision:** Moderate (50-70%) - many false positives from hot surfaces
- **Recall:** Moderate (60-75%) - misses small fires and fires with low intensity
- **IoU:** Low (0.40-0.55) - poor boundary accuracy
- **Why lower than deep learning:** Cannot learn complex patterns, no spatial context, fixed thresholds

**When Threshold-Based Might Be Sufficient:**
- Very large, intense fires (clear SWIR signal)
- Simple use cases where false positives are acceptable
- Real-time systems with extreme computational constraints
- When training data is unavailable

**Implementation Approach:**
1. Calculate SWIR band values (B11, B12) for all pixels
2. Apply threshold (e.g., B11 > 0.3 or B12 > 0.25) - values depend on normalization
3. Optionally combine with NBR threshold for burned areas
4. Generate binary mask
5. Evaluate on same test set as deep learning model

**What to Report:**
- Threshold values used (and how they were chosen)
- Performance metrics (IoU, Dice, Precision, Recall)
- Comparison with deep learning model
- Analysis of when threshold-based fails (false positives, missed fires)
- Visual examples showing threshold-based vs. deep learning predictions

---

### Baseline 2: Random Forest

**What it is:**
- Traditional machine learning classifier (ensemble of decision trees)
- Trained on hand-crafted features (spectral indices, band values)
- Non-deep-learning approach that can learn from data

**How it works:**
- Extract features for each pixel: raw band values (B2, B3, B4, B8, B11, B12), spectral indices (NBR, NDVI, BAI)
- Train Random Forest classifier to predict fire vs. non-fire
- Make pixel-wise predictions (similar to deep learning output)

**Why This Baseline?**

**Advantages:**
- ✅ **Learns from data:** Unlike threshold-based, can adapt to training data
- ✅ **Handles features well:** Good at learning relationships between spectral indices and fire
- ✅ **Interpretable:** Can analyze feature importance (which bands/indices matter most)
- ✅ **Standard ML method:** Well-established, widely used in remote sensing
- ✅ **Good baseline:** Represents "traditional ML" approach (pre-deep-learning era)
- ✅ **Reasonable performance:** Should achieve better results than threshold-based

**Limitations (Why Deep Learning Should Beat It):**
- ⚠️ **No spatial context:** Each pixel classified independently (no understanding of fire regions)
- ⚠️ **Hand-crafted features:** Relies on human-designed features (spectral indices) rather than learned features
- ⚠️ **Limited complexity:** Cannot learn complex, non-linear patterns that deep learning can
- ⚠️ **Pixel-wise only:** Cannot understand that fires form connected regions
- ⚠️ **Feature engineering required:** Need to design good features (spectral indices)

**Expected Performance:**
- **Precision:** Good (70-80%) - better than threshold-based, learns from data
- **Recall:** Good (70-80%) - can learn fire patterns from training data
- **IoU:** Moderate (0.55-0.65) - better than threshold-based, but worse than deep learning
- **Why lower than deep learning:** No spatial context, limited to hand-crafted features, pixel-wise only

**When Random Forest Might Be Sufficient:**
- Limited training data (Random Forest works with less data than deep learning)
- Need for interpretability (feature importance analysis)
- Computational constraints (faster inference than deep learning)
- Simple use cases where pixel-wise classification is sufficient

**Feature Engineering for Random Forest:**

**Raw Band Values:**
- B2 (Blue), B3 (Green), B4 (Red), B8 (NIR), B11 (SWIR-1), B12 (SWIR-2)
- Direct pixel values from satellite imagery

**Spectral Indices (Hand-Crafted Features):**
- **NBR (Normalized Burn Ratio):** `(B8 - B12) / (B8 + B12)` - detects burned areas
- **NDVI (Normalized Difference Vegetation Index):** `(B8 - B4) / (B8 + B4)` - vegetation health
- **BAI (Burned Area Index):** `1 / ((0.1 - B4)² + (0.06 - B8)²)` - fire detection
- **NDWI (Normalized Difference Water Index):** `(B3 - B8) / (B3 + B8)` - water detection (helps exclude water)
- **EVI (Enhanced Vegetation Index):** More advanced vegetation index

**Why These Features?**
- **Spectral indices:** Capture relationships between bands that are meaningful for fire detection
- **Domain knowledge:** Based on remote sensing research and physical properties
- **Complementary:** Different indices capture different aspects (vegetation, burn, water)

**Random Forest Configuration:**
- **Number of trees:** 100-500 (more trees = better performance, but slower)
- **Max depth:** 10-20 (deeper trees = more complex patterns, but risk overfitting)
- **Min samples per leaf:** 5-10 (prevents overfitting)
- **Feature sampling:** Use all features (or sample subset for each tree)

**What to Report:**
- Features used (raw bands + spectral indices)
- Random Forest hyperparameters (number of trees, max depth, etc.)
- Feature importance analysis (which features matter most for fire detection)
- Performance metrics (IoU, Dice, Precision, Recall)
- Comparison with deep learning model
- Analysis of when Random Forest fails (spatial context, complex patterns)
- Visual examples showing Random Forest vs. deep learning predictions

---

## Baseline Comparison Strategy

**Evaluation Protocol:**
1. **Same test set:** Evaluate all baselines and deep learning model on identical test set
2. **Same metrics:** Use same evaluation metrics (IoU, Dice, Precision, Recall) for fair comparison
3. **Same preprocessing:** Apply same data preprocessing to all methods
4. **Multiple runs:** Run each baseline multiple times (if stochastic) and report average

**Expected Results:**
- **Threshold-based:** Lowest performance (IoU: 0.40-0.55)
- **Random Forest:** Moderate performance (IoU: 0.55-0.65)
- **Deep Learning (U-Net):** Highest performance (IoU: 0.70-0.75) - target from Section 1.4

**Why Deep Learning Should Outperform:**
1. **Spatial context:** U-Net understands that fires form connected regions (not just individual pixels)
2. **Learned features:** Deep learning learns optimal features automatically (better than hand-crafted)
3. **Complex patterns:** Can learn non-linear, complex relationships between bands
4. **End-to-end learning:** Optimizes entire pipeline for the task (not just classification step)
5. **Transfer learning:** Benefits from pretrained features (ImageNet → satellite imagery)

**When to Investigate (If Deep Learning Doesn't Outperform):**
- **Bug in implementation:** Check data loading, model architecture, training loop
- **Insufficient training:** Model may need more epochs or better hyperparameters
- **Data issues:** Check if training data is correct, sufficient, or properly preprocessed
- **Overfitting:** Deep learning may be overfitting to training set (check train/val gap)
- **Baseline too strong:** If Random Forest achieves IoU > 0.70, problem may be too simple for deep learning

---

## Baseline Implementation Requirements

**For Capstone Project:**
- [ ] Implement threshold-based baseline (SWIR thresholds)
- [ ] Implement Random Forest baseline (with spectral indices)
- [ ] Evaluate both baselines on test set
- [ ] Compare baselines with deep learning model
- [ ] Document baseline performance in final report
- [ ] Explain why deep learning outperforms (or investigate if it doesn't)
- [ ] Include baseline results in evaluation section
- [ ] Visual comparison: show example predictions from all three methods

**Baseline Documentation:**
- **Method description:** How each baseline works
- **Hyperparameters:** Threshold values, Random Forest parameters
- **Performance metrics:** IoU, Dice, Precision, Recall for each baseline
- **Comparison table:** Side-by-side comparison of all methods
- **Failure analysis:** When and why baselines fail
- **Conclusion:** Why deep learning is the right choice for this problem

---

*This is an optional component. Focus on the deep learning model first, and implement baselines only if time permits.*
