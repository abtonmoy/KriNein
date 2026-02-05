This Markdown file is structured as a **Complete Technical Specification**. It includes the logic, math, and architectural requirements for all 8 tests. You can upload this to any LLM (Gemini, GPT-4, Claude) and tell it: _"Implement the Python framework described in this specification."_

---

# Specification: HMMD vs. Multi-Baseline Benchmarking Framework

## 1. Project Context

**HMMD (Hierarchical Multi-Modal Deduplication)** is a "cheap-to-expensive" pipeline designed to prune redundant video frames before they reach a Vision Language Model (VLM).
**The Cascade:** Hash Voting (pHash/dHash) LPIPS (Perceptual) CLIP (Semantic) Adaptive Density Selection.

## 2. Objective

Create a Python-based benchmarking suite that processes a directory of videos and compares HMMD against 7 other keyframe extraction/deduplication techniques.

---

## 3. Test Implementations

### **Category A: Temporal & Random (Baselines)**

- **Test 1: Uniform Sampling (1 FPS & 30 FPS)**
- _Logic:_ Extract frames at fixed intervals. 1 FPS represents the common VLM benchmark; 30 FPS represents the "full data" upper bound.

- **Test 2: Random Sampling**
- _Logic:_ Pick frames randomly, where matches the number of frames HMMD selected (for a fair "luck" comparison).

### **Category B: Traditional Computer Vision (CPU-Bound)**

- **Test 3: Color Histogram Correlation**
- _Logic:_ Convert frames to HSV. Compare 3D Histograms. If correlation , the frame is redundant.

- **Test 4: ORB Feature Matching**
- _Logic:_ Use `cv2.ORB_create()`. Detect keypoints. If the Hamming distance between frame and suggests match, mark as a keyframe.

- **Test 5: Optical Flow (Motion Peak)**
- _Logic:_ Use `cv2.calcOpticalFlowFarneback`. Calculate mean motion magnitude. Extract frames where motion spikes or drops (acceleration/deceleration).

### **Category C: Deep Learning (GPU-Bound)**

- **Test 6: CLIP-Only Deduplication**
- _Logic:_ Skip the hierarchy. Encode every frame with `clip-ViT-B-32`. Drop frames with Cosine Similarity .

- **Test 7: K-Means Clustering**
- _Logic:_ Generate CLIP embeddings for all frames. Cluster into groups (where is adaptive or fixed to 10). Pick the centroid of each cluster.

### **Category D: The Proposed Method**

- **Test 8: Full HMMD Cascade**
- _Logic:_ Implement the three-stage filter:

1. **Hash Stage:** Fast bitwise XOR.
2. **LPIPS Stage:** Use `lpips` library for perceptual distance.
3. **CLIP Stage:** Final semantic check.
4. **Density:** Use NMS (Non-Maximum Suppression) to ensure temporal spacing.

---

## 4. Required Evaluation Metrics

The framework must generate a `results.csv` containing:

1. **Selection Count:** Total frames kept.
2. **Compression Ratio:** (Input Frames / Output Frames).
3. **Processing Latency:** Time taken per video (Wall clock).
4. **Information Density:** Average CLIP distance between selected frames (higher is better—indicates unique information).
5. **Estimated VLM Cost:** Calculation based on **$0.015 per 1000 tokens** (approximate token count per frame).

---

## 5. Technical Stack Requirements

- **OpenCV (`cv2`):** For video I/O and traditional CV tests.
- **PyTorch:** For LPIPS and CLIP.
- **Sentence-Transformers:** For easy CLIP implementation.
- **Scikit-Learn:** For K-Means clustering.
- **Pandas:** For logging the benchmark results.

---

## 6. Prompt for the LLM Coder

> "Please implement the Python framework described above. Create a main class `HMMDBenchmark` with methods for each test. Ensure it handles video decoding efficiently using a generator to avoid OOM (Out of Memory) errors. The output should be a structured directory containing the extracted images for each method and a final CSV summary of performance."

---
