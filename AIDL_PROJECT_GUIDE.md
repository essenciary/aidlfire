# AIDL Project Guide - Fire Detection from Satellite Imagery

A comprehensive guide to the CEMS Wildfire Dataset and how to use it for deep learning fire segmentation.

---

## Table of Contents

- [The Big Picture](#the-big-picture)
- [What's in the Data?](#whats-in-the-data)
  - [Satellite Images (Multiple Channels)](#satellite-images-multiple-channels)
  - [Fire Masks (Labels)](#fire-masks-labels)
- [Why Do We Cut Images into Patches?](#why-do-we-cut-images-into-patches)
- [Classification vs Segmentation](#classification-vs-segmentation)
- [The Training Process](#the-training-process)
- [What About Inference?](#what-about-inference)
- [Where Are the Labels?](#where-are-the-labels)
- [Training Data vs Real-Time Inference](#training-data-vs-real-time-inference)
- [Understanding the GRA (Severity) Mask](#understanding-the-gra-severity-mask)
- [Why Cloud Masks Matter](#why-cloud-masks-matter)
- [Images with Zero Burned Pixels](#images-with-zero-burned-pixels)
- [The Complete Pipeline](#the-complete-pipeline)
- [Summary](#summary)

---

## The Big Picture

Imagine you have satellite photos of forests that caught fire. Your goal is to teach a computer to look at a satellite photo and say "fire happened here, here, and here" - like coloring in the burned areas.

```
SATELLITE PHOTO              YOUR MODEL'S JOB
┌─────────────────┐         ┌─────────────────┐
│  🌲🌲🔥🔥🌲🌲   │         │  ⬜⬜🟥🟥⬜⬜   │
│  🌲🔥🔥🔥🔥🌲   │   ──►   │  ⬜🟥🟥🟥🟥⬜   │
│  🌲🌲🔥🔥🌲🌲   │         │  ⬜⬜🟥🟥⬜⬜   │
│  🌲🌲🌲🌲🌲🌲   │         │  ⬜⬜⬜⬜⬜⬜   │
└─────────────────┘         └─────────────────┘
   Input Image               Output Mask
                            (red = burned)
```

---

## What's in the Data?

### Satellite Images (Multiple Channels)

Regular photos have 3 channels: Red, Green, Blue (RGB). But satellite images are special - they capture light our eyes can't see:

```
Normal Photo: 3 channels (RGB)
┌─────┬─────┬─────┐
│ Red │Green│Blue │
└─────┴─────┴─────┘

Satellite Image: 12 channels!
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│Blue │Green│ Red │ ... │ NIR │ ... │SWIR1│SWIR2│ ... │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  ▲                        ▲           ▲     ▲
  │                        │           └─────┴── These "see" burned areas!
  │                        │                     (infrared light)
  └── Visible light        └── Near-infrared
      (what we see)            (plants reflect this)
```

**Why so many channels?** Different wavelengths reveal different things:
- **SWIR (Short-Wave Infrared)**: Burned areas show up really clearly here - like a superpower for detecting fire damage
- **NIR (Near-Infrared)**: Healthy plants reflect this strongly, dead/burned plants don't

### Fire Masks (Labels)

For every satellite image, experts have drawn where the fire burned. This is your "answer key":

**DEL (Delineation)** - Binary: burned or not burned
```
0 = not burned (background)
1 = burned (fire area)
```

**GRA (Grading)** - Severity levels
```
0 = no damage
1 = minimal damage
2 = moderate damage
3 = high damage
4 = destroyed
```

---

## Why Do We Cut Images into Patches?

### Problem: Images Are HUGE

A single satellite image might be 1500 × 1500 pixels or bigger. Neural networks:
1. Need consistent input sizes (like 256 × 256)
2. Would run out of memory with huge images
3. Learn better from many small examples than few big ones

### Solution: Slice It Up!

```
Original Image (1500 × 1500)
┌─────────────────────────────────────┐
│                                     │
│    ┌─────┬─────┬─────┬─────┐       │
│    │  1  │  2  │  3  │  4  │       │
│    ├─────┼─────┼─────┼─────┤       │
│    │  5  │  6  │  7  │  8  │  ...  │
│    ├─────┼─────┼─────┼─────┤       │
│    │  9  │ 10  │ 11  │ 12  │       │
│    └─────┴─────┴─────┴─────┘       │
│                                     │
└─────────────────────────────────────┘

Each patch = 256 × 256 pixels
```

### Why Overlap During Training?

We use 50% overlap (stride of 128 pixels) during training:

```
Without overlap:        With 50% overlap:
┌─────┬─────┐          ┌─────────────┐
│  1  │  2  │          │  1  ┼──2    │
├─────┼─────┤          │  ┼──┼──┼    │
│  3  │  4  │          │  3  ┼──4    │
└─────┴─────┘          └─────────────┘
4 patches              More patches, edges
                       seen multiple times
```

**Why overlap?**
- A fire at the edge of patch 1 appears in the middle of patch 2
- The model sees each area from different "contexts"
- It's like data augmentation - more training examples!

---

## Classification vs Segmentation

### Binary Classification (Simpler)

**Question**: "Does this patch contain ANY fire?"
**Answer**: Yes (1) or No (0)

```
Input: 256×256×7 image
Output: Single number (0 or 1)

┌─────────────┐
│ 🌲🔥🌲🌲🌲  │
│ 🌲🔥🔥🌲🌲  │  ──► Model ──► "Yes, there's fire" (1)
│ 🌲🌲🌲🌲🌲  │
└─────────────┘
```

You could do this by checking: "if any pixel in mask > 0, label = 1"

### Segmentation (What We Actually Want)

**Question**: "Which PIXELS are burned?"
**Answer**: A mask the same size as the input

```
Input: 256×256×7 image
Output: 256×256 mask (one prediction per pixel!)

┌─────────────┐         ┌─────────────┐
│ 🌲🔥🌲🌲🌲  │         │ 0 1 0 0 0   │
│ 🌲🔥🔥🌲🌲  │  ──►    │ 0 1 1 0 0   │
│ 🌲🌲🌲🌲🌲  │         │ 0 0 0 0 0   │
└─────────────┘         └─────────────┘
   Input Image           Output Mask
```

**Segmentation is harder** because you're making thousands of predictions (one per pixel) instead of just one.

---

## The Training Process

### What the Neural Network Sees

```
Input Tensor Shape: (batch_size, 7, 256, 256)
                         │    │    │
                         │    │    └── Width & Height
                         │    └── 7 spectral bands (we pick the useful ones)
                         └── Multiple images at once (batch)

Target Mask Shape: (batch_size, 256, 256)
                         │       │
                         │       └── Same size as input
                         └── One label per pixel (0, 1, 2, 3, or 4)
```

### Training Loop (Simplified)

```python
for each batch of patches:

    # 1. Forward pass: model makes predictions
    predictions = model(images)      # Shape: (batch, num_classes, 256, 256)

    # 2. Compare to ground truth
    loss = compare(predictions, masks)  # How wrong are we?

    # 3. Backpropagation: figure out how to improve
    loss.backward()

    # 4. Update model weights
    optimizer.step()
```

### What the Model Learns

The model learns patterns like:
- "When SWIR bands show high values + NIR shows low values → probably burned"
- "This texture pattern usually means fire damage"
- "Edges between these colors often indicate burn boundaries"

---

## What About Inference?

**Inference** = using your trained model on new images

### Does the Input Look the Same?

**YES!** The input format is identical:
- Same 7 bands (or 12 if you use all)
- Same 256×256 patch size
- Same value range (0-1)

```
Training:                          Inference:
┌──────────────┐                  ┌──────────────┐
│ Image Patch  │                  │ NEW Image    │
│ (7, 256,256) │                  │ (7, 256,256) │
│      +       │                  │              │
│ Ground Truth │    ──────►       │ No labels!   │
│ Mask         │    (trained      │ Model        │
└──────────────┘     model)       │ predicts     │
                                  └──────────────┘
```

### Inference on a Full Image

Since real images are big, you:
1. Cut the new image into patches (no overlap needed, or small overlap)
2. Run each patch through the model
3. Stitch the predictions back together

```
New Satellite Image          Run Model on Each          Stitch Together
┌─────┬─────┬─────┐         ┌─────┐ ┌─────┐         ┌─────────────────┐
│  1  │  2  │  3  │         │pred1│ │pred2│  ...    │                 │
├─────┼─────┼─────┤   ──►   └─────┘ └─────┘   ──►   │  Full Predicted │
│  4  │  5  │  6  │         ┌─────┐ ┌─────┐         │      Mask       │
└─────┴─────┴─────┘         │pred4│ │pred5│         │                 │
                            └─────┘ └─────┘         └─────────────────┘
```

---

## Where Are the Labels?

The **labels are the mask files** (DEL and GRA). For every satellite image, there's a corresponding mask that shows where fire burned:

```
For each satellite image:
├── EMSR230_AOI01_01_S2L2A.tif    ← INPUT (what satellite sees)
├── EMSR230_AOI01_01_DEL.tif      ← LABEL (binary: fire/no fire)
├── EMSR230_AOI01_01_GRA.tif      ← LABEL (severity: 0-4)
└── EMSR230_AOI01_01_CM.tif       ← Cloud mask (helper data)
```

After patching:
```
├── patch_r0_c0_image.npy         ← INPUT  (256×256×7 channels)
└── patch_r0_c0_mask.npy          ← LABEL  (256×256 with 0/1 or 0-4)
```

### How Was the Data Labeled?

**NOT by an algorithm** — by human experts at Copernicus Emergency Management Service (CEMS):

```
┌─────────────────────────────────────────────────────────────────┐
│  LABELING PROCESS (done by CEMS analysts)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Fire event reported (e.g., Portugal 2017)                   │
│                         ↓                                       │
│  2. Analysts get PRE-fire satellite image                       │
│                         ↓                                       │
│  3. Analysts get POST-fire satellite image                      │
│                         ↓                                       │
│  4. Analysts MANUALLY draw polygons around burned areas         │
│     (comparing pre vs post, using spectral signatures)          │
│                         ↓                                       │
│  5. Polygons converted to raster masks (DEL, GRA)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This is why the labels are high quality — they're **human-verified ground truth**, not automated predictions.

### What the Model Learns

The model learns the **relationship between spectral patterns and fire presence**:

```
TRAINING (what you have):
┌──────────────────┐     ┌──────────────────┐
│  Satellite Image │     │   Ground Truth   │
│  (7 channels)    │ ──► │   Mask (DEL)     │
│                  │     │   0 = no fire    │
│  INPUT           │     │   1 = fire       │
└──────────────────┘     └──────────────────┘
        ↓                        ↓
    Model learns:  "when pixels look like THIS → label is THAT"


INFERENCE (real-time):
┌──────────────────┐     ┌──────────────────┐
│  Satellite Image │     │        ?         │
│  (7 channels)    │ ──► │   YOUR MODEL     │
│                  │     │   PREDICTS THIS  │
│  INPUT           │     │                  │
└──────────────────┘     └──────────────────┘
```

---

## Training Data vs Real-Time Inference

### Key Differences

| Aspect | Training Data | Real-Time Inference |
|--------|---------------|---------------------|
| **Satellite image** | ✅ Yes | ✅ Yes |
| **Mask (labels)** | ✅ Yes (DEL/GRA) | ❌ **NO** — this is what you predict! |
| **Cloud mask** | ✅ Provided | ⚠️ Need to generate or estimate |
| **When acquired** | Post-fire (burned area visible) | During or right after fire |
| **Quality** | Curated, cloud-free selected | Whatever is available |
| **Time context** | Historical events | Live/current events |

### What Real-Time Data is Missing

When you use your trained model on new satellite images:

1. **No mask** — that's your model's job to predict!
2. **No curated cloud mask** — you may need to:
   - Use Sentinel-2's built-in cloud detection
   - Train a separate cloud detection model
   - Use external cloud masking services
3. **No guaranteed quality** — training images were selected for good visibility; real-time may have:
   - Partial cloud cover
   - Smoke from active fires
   - Haze or atmospheric interference
4. **No geographic metadata** — you'll need to track coordinates yourself

### The Inference Pipeline You'll Build

```
Real-time Sentinel-2 image
         ↓
┌────────────────────────┐
│ 1. Download from ESA   │  (Copernicus Open Access Hub)
│    Sentinel-2 API      │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 2. Select same 7 bands │  indices (1,2,3,7,8,10,11)
│    Normalize to 0-1    │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 3. Generate cloud mask │  (optional but recommended)
│    or estimate clouds  │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 4. Cut into 256×256    │  stride=256 (no overlap needed)
│    patches             │
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 5. Run YOUR MODEL      │  → predicted mask per patch
└────────────────────────┘
         ↓
┌────────────────────────┐
│ 6. Stitch patches back │  → full image prediction
│    into full image     │
└────────────────────────┘
         ↓
    Fire/burn map!
```

### Key Takeaway

Your trained model learns patterns like "low NIR + high SWIR = burned area" from labeled training data. At inference time, it applies these learned patterns to new, unlabeled images to predict where fires/burns are located — **no labels needed**.

---

## Understanding the GRA (Severity) Mask

### Why Does the TIF Look Black?

The TIF file contains integer values **0, 1, 2, 3, 4**. When your image viewer opens it:

```
TIF values:     0    1    2    3    4
                │    │    │    │    │
                ▼    ▼    ▼    ▼    ▼
Image viewer    ■    ■    ■    ■    ■     (all look black!)
expects 0-255
```

Since 0-4 are all near zero on a 0-255 scale, they appear black. The **data is correct**, it's just not visible without scaling.

### PNG vs TIF

The PNG is just a **visualization** with a colormap applied:

```
┌───────┬─────────────┬─────────────────┬──────────────────┐
│ Value │ Meaning     │ PNG Color       │ RGB              │
├───────┼─────────────┼─────────────────┼──────────────────┤
│   0   │ No damage   │ ⬛ Black        │ (0, 0, 0)        │
│   1   │ Negligible  │ 🟩 Light Green  │ (181, 254, 142)  │
│   2   │ Moderate    │ 🟨 Yellow       │ (254, 217, 142)  │
│   3   │ High        │ 🟧 Orange       │ (254, 153, 41)   │
│   4   │ Destroyed   │ 🟥 Dark Red     │ (204, 76, 2)     │
└───────┴─────────────┴─────────────────┴──────────────────┘
```

**For training, use the TIF** (with values 0-4). The PNG is just for humans to look at.

### Not All Images Have GRA

Some fires only have DEL (binary burned/not burned), while some have both DEL and GRA (severity levels). This depends on what CEMS analysts provided. The `satelliteData.csv` has a `GRA` column (0 or 1) indicating availability.

### Visualizing the TIF Properly

```python
import rasterio
import matplotlib.pyplot as plt

with rasterio.open("path/to/GRA.tif") as src:
    gra = src.read(1)

# Now you can see it!
plt.imshow(gra, cmap='YlOrRd', vmin=0, vmax=4)
plt.colorbar(label='Severity (0-4)')
plt.show()
```

---

## Why Cloud Masks Matter

### The Problem

When the satellite takes a photo, it captures whatever is there - including clouds:

```
What satellite sees:          What's actually on the ground:
┌─────────────────────┐       ┌─────────────────────┐
│ 🌲🌲🌲☁️☁️☁️🌲🌲│       │ 🌲🌲🌲🔥🔥🔥🌲🌲│
│ 🌲🌲☁️☁️☁️☁️🌲🌲│       │ 🌲🌲🔥🔥🔥🔥🌲🌲│
│ 🌲🌲🌲🌲🔥🔥🌲🌲│       │ 🌲🌲🌲🌲🔥🔥🌲🌲│
└─────────────────────┘       └─────────────────────┘
     S2L2A image               Reality (hidden by cloud)

The satellite image has cloud pixels where fire actually exists!
```

**The cloud pixels in S2L2A contain cloud reflectance, NOT ground information.** That data is useless for fire detection.

### How Cloud Mask Helps

```
S2L2A (satellite image)       CM (cloud mask)           What to trust
┌─────────────────────┐      ┌─────────────────────┐   ┌─────────────────────┐
│ 🌲🌲🌲☁️☁️☁️🌲🌲│      │ 0 0 0 1 1 1 0 0 │   │ ✓ ✓ ✓ ✗ ✗ ✗ ✓ ✓ │
│ 🌲🌲☁️☁️☁️☁️🌲🌲│  +   │ 0 0 1 1 1 1 0 0 │ = │ ✓ ✓ ✗ ✗ ✗ ✗ ✓ ✓ │
│ 🌲🌲🌲🌲🔥🔥🌲🌲│      │ 0 0 0 0 0 0 0 0 │   │ ✓ ✓ ✓ ✓ ✓ ✓ ✓ ✓ │
└─────────────────────┘      └─────────────────────┘   └─────────────────────┘
                              0 = clear, 1 = cloud      ✗ = ignore these pixels
```

### Think of it Like This

Imagine you're grading a student's answers, but someone spilled coffee on part of the paper:

- **S2L2A** = the paper (some parts have coffee stains)
- **Cloud mask** = shows you WHERE the coffee stains are
- **DEL/GRA** = the answer key

You wouldn't grade the coffee-stained parts - you'd skip them. Same with clouds!

### In Practice

**During training:**
- Skip patches with too many clouds (>50% cloudy = unreliable)
- Or mask out cloudy pixels from the loss calculation

**During inference:**
- Flag predictions in cloudy areas as "uncertain"
- Or don't make predictions for those pixels

```python
# Example: mask out cloudy pixels during training
cloud_mask = load("CM.tif")  # 0=clear, 1+=cloud
is_clear = (cloud_mask == 0)

# Only compute loss on clear pixels
loss = criterion(prediction[is_clear], target[is_clear])
```

---

## Images with Zero Burned Pixels

### Why Do Some Images Have No Fire?

About 4% of images have `pixelBurned = 0`. When a large fire is mapped, CEMS divides it into **tiles** (like a grid). Some tiles at the edges might not contain any actual fire:

```
Fire Event EMSR207 - Area of Interest 01
Divided into 9 tiles:

┌─────────┬─────────┬─────────┐
│  tile_01│  tile_02│  tile_03│
│  🔥🔥   │  🔥🔥🔥 │         │
├─────────┼─────────┼─────────┤
│  tile_04│  tile_05│  tile_06│
│  🔥🔥🔥 │  🔥🔥   │         │
├─────────┼─────────┼─────────┤
│  tile_07│  tile_08│  tile_09│   ◄── tile_07 has 0 burned pixels!
│         │  🔥     │         │       (it's in the dataset but no fire)
└─────────┴─────────┴─────────┘
```

### Should You Keep Them?

| Approach | Pros | Cons |
|----------|------|------|
| Keep all | Model learns "no fire" cases | Class imbalance if too many |
| Remove all | Only train on fire images | Model might predict fire everywhere |
| Keep some | Balanced training | Need to decide how many |

**Recommendation:** For segmentation, having some negative patches (no fire) is actually useful so the model learns to predict "0" when there's no fire.

---

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA PREPARATION                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Raw Satellite Images        Cut into Patches         Ready for PyTorch │
│  (huge GeoTIFFs)            (256×256 chunks)          (numpy arrays)    │
│                                                                         │
│  ┌─────────────┐            ┌─────┐ ┌─────┐          image.npy          │
│  │             │            │     │ │     │          mask.npy           │
│  │  1500×1500  │    ──►     ├─────┼─────┤    ──►    metadata.csv        │
│  │  12 bands   │            │     │ │     │                             │
│  │             │            └─────┘ └─────┘                             │
│  └─────────────┘                                                        │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                              TRAINING                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Load Patches          Feed to Model           Update Weights           │
│                                                                         │
│  ┌─────────┐          ┌─────────────┐         ┌─────────────┐          │
│  │ image   │          │             │         │ Loss: 0.5   │          │
│  │ (7,256, │   ──►    │   U-Net     │   ──►   │ Loss: 0.3   │          │
│  │  256)   │          │   or other  │         │ Loss: 0.1   │          │
│  │         │          │   segmenter │         │     ↓       │          │
│  │ mask    │          │             │         │ Model gets  │          │
│  │(256,256)│          └─────────────┘         │ better!     │          │
│  └─────────┘                                  └─────────────┘          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             INFERENCE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  New Image             Trained Model            Prediction              │
│  (never seen!)         (frozen weights)         (fire map!)             │
│                                                                         │
│  ┌─────────┐          ┌─────────────┐         ┌─────────────┐          │
│  │ new     │          │             │         │ ⬜⬜🟥🟥⬜   │          │
│  │ image   │   ──►    │   U-Net     │   ──►   │ ⬜🟥🟥🟥⬜   │          │
│  │ (same   │          │  (trained)  │         │ ⬜⬜🟥⬜⬜   │          │
│  │ format) │          │             │         │             │          │
│  └─────────┘          └─────────────┘         └─────────────┘          │
│                                                                         │
│  NO MASK NEEDED!                               "Fire is HERE"           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

| Concept | What It Means |
|---------|---------------|
| **Satellite bands** | Different "colors" including invisible light - SWIR sees fire damage |
| **Patching** | Cutting big images into small pieces the neural network can handle |
| **Overlap** | Patches share edges during training = more data, better learning |
| **DEL mask** | Binary answer: burned (1) or not (0) |
| **GRA mask** | Severity: how badly burned (0-4) |
| **Cloud mask** | Shows which pixels are blocked by clouds (unreliable data) |
| **Classification** | "Is there fire in this patch?" → Yes/No |
| **Segmentation** | "Where exactly is the fire?" → pixel-by-pixel map |
| **Inference** | Using trained model on new images (same format, no labels needed) |

---

## Key Files Reference

```
Each image directory contains:
├── *_S2L2A.tif      ◄── INPUT: Satellite image (12 bands)
├── *_DEL.tif        ◄── TARGET: Binary fire mask (0 or 1)
├── *_GRA.tif        ◄── TARGET: Severity mask (0-4) [not always present]
├── *_CM.tif         ◄── FILTER: Cloud mask (0=clear, 1+=cloud)
├── *_ESA_LC.tif     ◄── EXTRA: Land cover type
└── *.png            ◄── Visualizations (for humans, not training)
```

**The magic**: Once trained, your model can look at a satellite image it's never seen and draw the fire boundaries automatically!
