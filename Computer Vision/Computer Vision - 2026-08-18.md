# Computer Vision (AIML ZG525) — Master Study Notes

> **Course:** AIML ZG525 · BITS Pilani WILP · **Lead Instructor:** Dr Amar Kumar Verma
> **Notes updated:** 2026-09-16 · **Books:** Szeliski, *Computer Vision: Algorithms & Applications* (2022); Forsyth & Ponce; Prince, *Understanding Deep Learning* (2024)
> **Status:** Living document — append new lectures under *"Update Log"* and extend the topic map.

## How to use this note
1. **Understand** each topic's 💡 *intuition* first, then the 🧮 *math*.
2. **Redo** every ✍️ *worked example* by hand — CV exams are numerical.
3. Before the exam, drill the 🎯 *exam pointers*, **Cheat Sheet**, and **Self-Test**.
4. Icons: 💡 intuition · 🧮 formula · ✍️ worked example · 🎯 exam · ⚠️ trap · 🔁 revision.

---

## 0. The Big Picture

**Computer Vision = turning pixels into "what is where".** A computer receives only a grid of numbers; CV recovers *meaning* (objects, geometry, actions) from it.

- **It is the *inverse* problem:** Graphics goes *model → image* (forward); Vision goes *image → model/description* (inverse); Image processing goes *image → image* (no meaning added). The inverse problem is **under-constrained** — one 2D photo can come from infinitely many 3D worlds (depth is lost in projection).
- **Why vision is hard:** the *same* object gives wildly different pixels due to **viewpoint, scale, lighting, occlusion, deformation, clutter, and intra-class variation**.
- **Three levels:** *low-level* (pixels, edges, colour, gradients) → *mid-level* (regions, textures, keypoints) → *high-level* (objects, scenes, actions).

**Core tasks:** Classification (WHAT, one label) · Detection (WHAT + WHERE, boxes) · Segmentation (label every pixel) · + tracking, depth, captioning, generation, VQA.

**Three eras** 🎯: **Classical** (1960s–2000s, hand-crafted features; wins = OCR, face detection) → **Deep learning** (2012 **AlexNet** cut ImageNet top-5 error ~26%→~15%; CNNs + GPUs) → **Foundation models** (2020s: ViT, CLIP, SAM, diffusion, GPT-4V/Gemini; zero-shot & multimodal). *Three enablers: DATA + COMPUTE (GPUs) + ALGORITHMS.*

### Syllabus / Topic map
| # | Module | Taught | Exam scope |
|---|--------|:---:|:---:|
| L1 | Image basics: pixels, resolution, bit depth, colour | ✅ | EC2 |
| L2 | Vision fundamentals: camera model, sampling, transforms, histograms | ✅ | EC2 |
| L3 | Features: edges (Canny), corners (Harris), SIFT | ✅ | EC2 |
| — | Filtering & convolution, RANSAC, demosaicing | ✅ | EC2 |
| L7 | Segmentation: k-means/mean-shift, **U-Net**, **DeepLab**, dilated conv, Dice/IoU | ✅ | EC3 |
| L8 | Instance segmentation: **Mask R-CNN**, RoI Align, mask AP, **SAM** | ✅ | EC3 |
| L6 | Detection: IoU/AP/mAP, **R-CNN** family, **YOLO**, NMS | ✅ | EC3 |
| L4 | Deep learning: **CNN**, pretrained backbones & transfer learning | ✅ | EC3 |
| L5 | Custom CNN design, training, regularisation & honest evaluation | ✅ | EC3 |
| — | Vision Transformers (**ViT**) | ▶ | EC3 |
| — | Self-supervised: **SimCLR, DINO, MoCo** | ▶ | EC3 |

### 📋 Evaluation
- **EC2 – Mid-sem (30%, closed book):** L1–L3 + filtering. *Numerical* (convolution, Sobel, Gaussian, Harris, gamma, demosaicing, RANSAC).
- **EC3 – Comprehensive (40%, open book):** adds segmentation, detection, CNN/ViT, self-supervised learning.

---

## L1 · Image Formation & Representation

### 1.1 An image is a grid of pixels / a function
- **Pixel** = smallest picture element, stores an intensity. Origin **(0,0) top-left**; `x` = column, `y` = row (exactly how NumPy/OpenCV index, but note `img[row, col]`).
- **Grayscale image = function** $I(x,y)$ returning one intensity per location. **Colour image = 3 functions** (one per channel).
- **Resolution** = width × height (e.g. 1920×1080). More pixels → more spatial detail.
- **Bit depth** = bits/pixel. 8-bit → $2^8 = 256$ levels (0 = black, 255 = white). Fewer levels → **banding**.

✍️ *A 1920×1080, 8-bit grayscale image:* pixels = $1920\times1080 = 2{,}073{,}600$; gray levels = $2^8 = 256$.

### 1.2 Colour spaces 🎯
| Space | Channels | Best for |
|-------|----------|----------|
| **RGB** | Red, Green, Blue (additive) | display, capture, storage. 8-bit/channel → ~16.7 M colours |
| **Grayscale** | intensity only | edges, texture, speed/memory. **Luma** = $0.299R + 0.587G + 0.114B$ |
| **HSV** | Hue, Saturation, Value | **colour selection, lighting-robust** — separates colour (H) from brightness (V) |

> 💡 **Why HSV for colour tasks?** You can threshold a **Hue** range to grab all "red-ish" pixels in one step, regardless of shading — hard in RGB because brightness is smeared across all three channels.
> ⚠️ **OpenCV loads images as BGR, not RGB.**

---

## L2 · Vision Fundamentals

### 2.1 The pinhole camera model 🎯
A 3D world point $\mathbf{X}=(X,Y,Z)$ is projected through one **optical centre** onto the image plane.
- **Projection relative to the optical axis:** if image-plane coordinates are measured from the principal point and both axes use focal length $f$,

```math
x=f\frac{X}{Z},\qquad y=f\frac{Y}{Z}.
```

The division by depth creates perspective: doubling $Z$ halves $x$ and $y$, so farther objects appear smaller.
- **Pixel coordinates:** an actual pixel grid normally starts at an image corner. With focal lengths $f_x,f_y$ measured in pixels and principal point $(c_x,c_y)$,

```math
u=f_x\frac{X}{Z}+c_x,\qquad v=f_y\frac{Y}{Z}+c_y.
```

Thus $(x,y)$ are centred image-plane coordinates, while $(u,v)$ are pixel coordinates. The first formula is not wrong; it uses a simpler coordinate origin.
- **Focal length** $f$: larger $f$ → narrower field of view + bigger objects (telephoto); smaller $f$ → wide FOV + stronger perspective. *Changing $f$ ≠ moving the camera.*

**Intrinsics vs Extrinsics:**
```math
K=\begin{bmatrix} f_x & 0 & c_x\\ 0 & f_y & c_y\\ 0 & 0 & 1\end{bmatrix}
\quad\text{(intrinsic, inside the camera)},\qquad [\,R \mid t\,]\ \text{(extrinsic, camera pose)}.
```
- **Intrinsic $K$** — focal lengths $f_x,f_y$, principal point $(c_x,c_y)$, skew≈0. Maps *camera → pixels*. Fixed for a camera.
- **Extrinsic $[R\mid t]$** — rotation + translation. Maps *world → camera*. Changes every shot. **Calibration** estimates $K$ + 5 distortion coeffs (need ≥10 images).
- **Four frames:** world → (extrinsics) → camera → (intrinsics) → image plane → pixel grid.

### 2.2 Sampling, quantisation & aliasing 🎯
- **Spatial sampling** = *where* we measure (pixel grid). **Quantisation** = *how many* intensity levels.
- **Aliasing:** fine patterns above the sampling limit appear as false coarse patterns (**moiré**, jaggies). 🧮 For a band-limited signal with highest frequency $f_{max}$, **Nyquist–Shannon** requires

```math
f_s\ge2f_{max}.
```

Here $f_s$ is the sampling frequency. Equality is the theoretical minimum; real systems sample faster to leave room for a non-ideal anti-alias filter. **Fix:** low-pass **before** downsampling.

### 2.3 Geometric transforms & homogeneous coordinates 🎯
Write a 2D point as $\mathbf p = [x, y, 1]^\top$; a **3×3 matrix** then represents linear motion **+** translation: $\mathbf p' = H\mathbf p$.

| Transform | Effect | Preserves |
|-----------|--------|-----------|
| Translation | position | everything else |
| Scaling ($x'=s_x x$) | size | angles (if uniform) |
| Rotation | orientation | lengths & angles |
| Shear | slants | parallelism, straight lines |
| **Affine** (6 params) | combines above; last row $[0,0,1]$ | straight & parallel lines (not lengths/angles) |
| **Homography** (8 DoF) | perspective (convergence) | straight lines only |

- **Rotation:** $x' = x\cos\theta - y\sin\theta,\quad y' = x\sin\theta + y\cos\theta$.
- ⚠️ **Order matters:** $R\cdot T \ne T\cdot R$ (translate-then-rotate ≠ rotate-then-translate).
- An affine map needs **3 non-collinear point pairs**; perspective needs a **homography**.

✍️ *Translate $p=(4,3)$ by $t=(-2,5)$:* $p'=(2,8)$.

### 2.4 Interpolation (resampling) 🎯
| Method | Speed | Quality | Use when |
|--------|-------|---------|----------|
| **Nearest neighbour** | fastest | blocky | **categorical data / label masks** (never invents new labels) |
| **Bilinear** | fast | smooth | natural images (strong default) |
| **Bicubic** | slow | sharper/smoother; may **ring/overshoot** near edges | high-quality resize |

> ⚠️ **Exam rule:** resize a **segmentation mask** with **nearest-neighbour** — bilinear/bicubic would invent impossible in-between class IDs.

### 2.5 Photometric transforms & histograms 🎯
- **Brightness/contrast (linear):** $g(x,y) = \alpha\,f(x,y) + \beta$. $\beta$ shifts (brightness), $\alpha$ scales differences (contrast). Clip to valid range.
- **Gamma (nonlinear):** normalise to $[0,1]$, then $g = f^{\gamma}$. $\gamma<1$ **brightens** mid-tones; $\gamma>1$ **darkens**. Endpoints (0,1) fixed.
- **Histogram** = intensity distribution. Dark image → narrow left-shifted; low-contrast → narrow centred; clipped → spikes at 0/255 (info lost).
- **Histogram equalisation:** build the **CDF**, use the normalised CDF as a monotonic intensity remap → spreads occupied values across the full range → boosts global contrast. ⚠️ Can amplify noise; **CLAHE** (contrast-limited, tile-based) fixes this.

✍️ *Gamma at a pixel:* value 200/255 = 0.784; with $\gamma=2.2$: $0.784^{2.2}\approx0.585 \Rightarrow \approx149$.

---

## Filtering & Convolution (core exam machinery) 🎯

### 3.1 Correlation vs Convolution
Both slide a **kernel** over the image and take weighted sums. **Convolution first flips the kernel** (180°); correlation does not.
```math
(f * g)[n] = \sum_k f[k]\,g[n-k] \quad(\text{convolution, kernel flipped})
```
> ⚠️ **EC2 question:** "difference between correlation and convolution for signal `[…,2,3,4,5,6,…]` with filter `[1,2,4,8,4]`" → for correlation, multiply-add as-is; for convolution, **reverse the filter to `[4,8,4,2,1]`** first, then slide. For **symmetric** kernels the two are identical.

### 3.2 Gaussian smoothing & two key theorems
- **Gaussian filter** blurs / removes noise; **σ** sets the blur scale.
- 🧮 **Separability:** a 2D Gaussian = 1D Gaussian along x, then along y → $O(k^2)\to O(2k)$ per pixel.
- 🧮 **Composition of Gaussians** (⭐ EC2): convolving with $\sigma_1$ then $\sigma_2$ equals a *single* Gaussian with
```math
\sigma = \sqrt{\sigma_1^2 + \sigma_2^2}.
```
✍️ *σ=3 then σ=5:* $\sigma=\sqrt{9+25}=\sqrt{34}\approx 5.83$.
- **Derivative theorem of convolution:** $\dfrac{d}{dx}(f * g) = f * \dfrac{d}{dx}g$ — precompute the *derivative of Gaussian* to smooth **and** differentiate in one pass.

---

## L3 · Features: Edges, Corners, Descriptors

### 4.1 Why features? Characteristics of good ones
Motivating task: **panorama stitching** = (1) extract features → (2) match → (3) align. Good features are **Repeatable** (found despite transforms), **Salient/distinctive**, **Compact** (far fewer than pixels), **Local** (robust to clutter/occlusion). Uses: alignment, 3D reconstruction, tracking, navigation, retrieval, recognition.

### 4.2 Edge detection 🎯
- **Edge** = place of **rapid intensity change** → an **extremum of the first derivative**. Caused by depth / surface-colour / illumination / surface-normal discontinuities.
- 🧮 **Image gradient** $\nabla I = \big[\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}\big]$.
  - **Magnitude** (edge strength): $\lVert\nabla I\rVert = \sqrt{I_x^2 + I_y^2}$.
  - **Direction:** $\theta = \operatorname{atan2}(I_y, I_x)$ — points toward steepest increase, **perpendicular to the edge**.
- **Finite-difference filters:** $[-1\ 1]$; **Sobel** adds smoothing:
```math
G_x=\begin{bmatrix}-1&0&1\\-2&0&2\\-1&0&1\end{bmatrix},\quad G_y=\begin{bmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{bmatrix}.
```
- ⚠️ **Noise** creates spurious derivative peaks → **smooth first** (Gaussian), then differentiate (or use derivative-of-Gaussian).

**Canny edge detector — 4 steps (memorize):**
1. Smooth with **derivative of Gaussian**.
2. Compute gradient **magnitude & orientation**.
3. **Non-maximum suppression (NMS):** thin ridges to 1-pixel width (keep local max along gradient direction).
4. **Hysteresis thresholding:** two thresholds — **high** starts an edge, **low** continues it (links weak edges attached to strong ones).

✍️ *EC2 Sobel example:* apply $G_x, G_y$ at pixel [2,2] (after a 3×3 box smooth) → $\theta=\operatorname{atan2}(I_y,I_x)$ for direction, $\sqrt{I_x^2+I_y^2}$ for magnitude.

### 4.3 Corner detection — Harris 🎯
💡 A **corner** = shifting a small window in **any** direction gives a large intensity change (edge = change in one direction only; flat = no change).
- 🧮 Let $(u,v)$ be a small horizontal/vertical window shift, $I_x,I_y$ the image derivatives, and $w(x,y)$ a window weight that emphasizes nearby pixels. The change caused by the shift is approximated by
```math
E(u,v)\approx
\begin{bmatrix}u&v\end{bmatrix}M
\begin{bmatrix}u\\v\end{bmatrix},
```
with the **second-moment (structure) matrix**
```math
M = \sum_{W} w(x,y)\begin{bmatrix} I_x^2 & I_x I_y\\ I_x I_y & I_y^2\end{bmatrix}.
```
- Let $\lambda_1,\lambda_2$ be the two eigenvalues of $M$. Since $\det(M)=\lambda_1\lambda_2$ and $\operatorname{trace}(M)=\lambda_1+\lambda_2$, the **Harris response** can be read in either equivalent form:

```math
R=\det(M)-k\,[\operatorname{trace}(M)]^2
=\lambda_1\lambda_2-k(\lambda_1+\lambda_2)^2.
```

Here $k\approx0.04$–0.06.
  - $R > 0$ (both $\lambda$ large) → **corner**; $R < 0$ (one $\lambda$ large) → **edge**; $|R|$ small (both small) → **flat**.
- **Rotation-invariant** (eigenvalues don't change with rotation) but **not scale-invariant**.

### 4.4 SIFT & descriptors (scale invariance)
- **SIFT** = Scale-Invariant Feature Transform: detect keypoints across scales (Difference-of-Gaussian pyramid), assign orientation, build a **128-D gradient-orientation-histogram descriptor** → invariant to scale, rotation, illumination. **HOG** = related dense gradient-histogram descriptor. Descriptors let you **match** the same point across images.

### 4.5 Model fitting: RANSAC 🎯
**RANSAC** (RANdom SAmple Consensus) robustly fits a model despite outliers: (a) randomly pick the **minimal** set (2 points for a line), (b) fit, (c) count **inliers** within threshold; repeat, keep the model with most inliers.
> ⚠️ **EC2 efficiency question:** with millions of points, count inliers **vectorized** — compute all point-to-line distances $|ax+by+c|/\sqrt{a^2+b^2}$ in one array op and threshold → $O(N)$ per hypothesis, no Python loop.

### 4.6 Demosaicing (Bayer) 🎯
Sensors capture one colour per pixel via a **Bayer (RGGB) mosaic**. **Demosaicing** reconstructs the full 3-channel image: place each measured value into its channel of a `H×W×3` array, then fill missing values by interpolation (nearest-neighbour in the EC2 question).

---

## Segmentation (EC3) 🎯

Partition an image into meaningful regions / **label every pixel**. Place it on the task ladder taught in class:
**Classification** = *what* (one label per image) → **Detection** = *what + where* (a **box**: pixel $x,y$ + width, height) → **Segmentation** = *what + where at the pixel level* (a **mask**, not a box).

- **Semantic segmentation:** one mask per **class** — every "person" pixel gets the same label (people are not separated).
- **Instance segmentation:** give each mask a **unique id** — person 1, person 2, person 3 are distinct. *(Semantic masks + unique identifiers = instance.)*
- **Panoptic segmentation:** assign every pixel a class, and give a unique instance id only to countable **thing** classes (person, car). Amorphous **stuff** classes (sky, road) receive a class label but no meaningful instance id.

> 🧠 **How to choose:** ask what the answer must look like. "Which pixels are road?" → semantic. "How many people, and which pixels belong to each?" → instance. "Label every pixel and count every object" → panoptic. **Choose the output first; the architecture follows.**

| Method | Idea | Notes |
|--------|------|-------|
| **Thresholding** | pixel > T → foreground | global (Otsu) or adaptive |
| **k-means** | cluster pixels in colour/intensity space into $k$ groups | must pick $k$; no spatial awareness; fast, unsupervised |
| **Mean-shift** | iteratively move each point to the **mean of neighbours within bandwidth** until convergence (mode-seeking) | no $k$ needed; bandwidth controls granularity |
| **U-Net** | encoder–decoder CNN with **skip connections** | supervised; especially useful when fine boundaries matter |
| **DeepLab** | dilated (atrous) CNN with **multiple dilation rates** (small/medium/large objects) | multi-scale field-of-view in one architecture |

✍️ **Mean-shift one iteration:** interpret **spatial radius 2** as “inspect pixels at most two positions away” and **intensity bandwidth 5** as “retain values near the current value 50,” for example the range $[45,55]$ with a hard-window kernel. If the retained intensities are $z_1,\ldots,z_m$ with kernel weights $w_1,\ldots,w_m$, update to
```math
z_{new}=\frac{\sum_{i=1}^{m}w_i z_i}{\sum_{i=1}^{m}w_i}.
```
Then centre the next window at $z_{new}$ and repeat until the shift is tiny. A numeric result **cannot** be computed unless the neighbour values and kernel weights are supplied.

> 🎯 **U-Net:** symmetric encoder (downsample, capture context) + decoder (upsample, localize). **Skip connections** copy high-resolution encoder features to the decoder → recover fine spatial detail lost in downsampling → sharp boundaries. Vs k-means: U-Net is supervised, learns semantics, far better but needs labelled data & compute; k-means is unsupervised, fast, but only groups colours. Lecture 7's skin-lesion ablation found a modest average skip gain on smooth, large lesions: mechanism value depends on the data, so compare matched seeds rather than one lucky run.

### Segmentation-CNN anatomy — two changes vs a classification CNN 🎯
A classification CNN ends **conv → pool → fully-connected → softmax → one class**. A segmentation network keeps the conv backbone but makes **two changes** (6-Sep class):
1. **Dilated (atrous) convolution** layers in the middle instead of plain conv.
2. **Upsampling** layer at the end to recover full pixel resolution as a mask — **no FC / softmax-to-one-class head**.

💡 **Dilated / atrous convolution** = insert **zeros (holes)** between kernel taps, so a $3\times3$ kernel "sees" a larger **field of view** *without adding parameters* (the zeros cost no multiply). The **dilation rate** sets the spacing:
- Small object → **small** rate (e.g. 2) so you don't skip its few pixels.
- Large object → **large** rate (e.g. 12–24) to cover more area cheaply.
- ⚖️ **Why not just a bigger kernel?** A $5\times5$ or $7\times7$ full kernel raises parameters and compute (e.g. 3M → 4M weights); dilation grows the receptive field at **constant** parameter count. Trade-off: dilation **approximates** (skips pixels), so very small objects can be missed — tune the rate.
- **DeepLab/ASPP** runs **several dilation rates in parallel** (small/medium/large) so one model handles many object sizes. U-Net instead restores resolution through a decoder and skip features: picture U-Net as a **V** (down, then up) and DeepLab as an **L** (stop downsampling and operate on a larger map). One pays for a decoder; the other pays compute on the higher-resolution map.

### Evaluating a segmentation 🎯
Same idea as detection but at the **pixel** level — compare the predicted mask to the **ground-truth** mask:
- **IoU (Jaccard)** $=\dfrac{|\text{pred}\cap\text{truth}|}{|\text{pred}\cup\text{truth}|}$ — 1 = exact overlap, 0 = none.
- **Dice similarity coefficient (DSC)** $=\dfrac{2|\text{pred}\cap\text{truth}|}{|\text{pred}|+|\text{truth}|}$ — ranges 0→1, like IoU but weights overlap more; common in medical imaging.
- **Mean IoU across $C$ classes:**

```math
\operatorname{mIoU}=\frac{1}{C}\sum_{c=1}^{C}\operatorname{IoU}_c.
```

This is a macro-average: every stated class counts once, even a rare class or a class with IoU 0. State whether background is included.
- **Pixel accuracy** — fraction of correctly-labelled pixels (misleading for tiny objects / class imbalance).
- **Voxel** accuracy — the 3D/video analogue: $x,y$ **+ depth over moving frames**.
- Report **foreground (object)** and **boundary** accuracy separately — a mask can be right in the middle but wrong at the edges.
- 🧮 **Dice and IoU are the same overlap in two forms:** $D=\dfrac{2J}{1+J}$ and $J=\dfrac{D}{2-D}$, where $D$ is Dice and $J$ is IoU. If IoU $=0.60$, Dice $=0.75$; Dice cannot be below IoU except that both agree at 0 and 1.
- ⚠️ **Why pixel accuracy lies:** if foreground is only 17.3% of pixels, predicting background everywhere scores 82.7% accuracy and finds no object. Prefer foreground/per-class IoU; use boundary F1 when the edge itself is the product.
- **Training loss:** for a binary mask, let $y_i\in\{0,1\}$ be the true label and $p_i\in[0,1]$ the predicted foreground probability at pixel $i$. A differentiable soft Dice loss is

```math
L_{Dice}=1-\frac{2\sum_i p_i y_i+\varepsilon}{\sum_i p_i+\sum_i y_i+\varepsilon}.
```

Here small $\varepsilon>0$ prevents division by zero. Combine it with per-pixel cross-entropy:

```math
L=\lambda L_{CE}+(1-\lambda)L_{Dice},\qquad 0\le\lambda\le1.
```

Dice directly rewards overlap under class imbalance; cross-entropy supplies local pixel-by-pixel supervision. The weight $\lambda$ is a chosen hyperparameter, not a universal constant.

⚠️ **Instance overlap:** when two masks overlap, the intersection gives only the shared area; recover each object's full area with the **union of its own mask**, then score it by IoU/Dice against ground truth.

### Instance segmentation: Mask R-CNN and SAM (L8) 🎯

**Mask R-CNN = Faster R-CNN + a parallel mask head.** The shared backbone/FPN builds features; the RPN proposes regions; each RoI feeds class, box, and mask heads. The loss keeps the questions separate:
```math
L=L_{cls}+L_{box}+L_{mask}.
```
The mask head predicts independent per-pixel sigmoid maps and only the true class's map is graded. The box head has already chosen the class, so mask classes do not need to compete with a per-pixel softmax.

**RoI Align fixes coordinate quantisation.** RoI Pool rounds the proposal and bin boundaries, then max-pools; that small feature-map shift becomes several input pixels at a coarse stride. RoI Align keeps fractional coordinates and uses bilinear interpolation. A box may tolerate the shift, but a mask carries it around its whole boundary. When you see "why does Mask R-CNN replace RoI Pool?", answer **pixel alignment**, not merely "better accuracy".

**Mask evaluation:** match scored predictions greedily to unclaimed truths as in detection, but calculate overlap **mask-to-mask**. COCO mask AP averages thresholds 0.50:0.95. A high AP50 can hide rough boundaries; **box AP is not mask AP**.

**SAM changes categories into prompts.** Its image encoder runs once; a point/box/mask prompt is encoded cheaply; a small mask decoder produces candidate masks and predicted IoU scores. It is **class-agnostic**: it outlines what was prompted but returns no class name and does not decide which objects matter.

| Trigger | Think | Answer skeleton | Trap |
|---|---|---|---|
| "Two touching people must remain separate" | instance, not semantic | one mask + id + label + confidence per object | one class mask merges them |
| "Why RoI Align?" | boundary precision | no rounding → bilinear samples → aligned mask | saying only "it is faster" |
| "Mask R-CNN vs SAM" | trained vocabulary vs prompted shape | Mask R-CNN names known classes; SAM masks prompted unknown objects | claiming SAM classifies the mask |
| "Dense shelf of identical products" | NMS/domain vocabulary failure | overlapping neighbours are real; COCO may lack `product`; use class-agnostic masks + own SKU classifier | lowering NMS threshold can delete more real objects |

> 🧠 **Memory hook:** detection gives **boxes**, semantic segmentation gives a **colouring book**, instance segmentation gives **cut-outs**, and SAM is a **prompted cutter without a label maker**.

---

## L6 · Object Detection (EC3) 🎯

Detection = **classify *and* localize** every object, and say **how many** — the answer is a **list of boxes of unknown length**, unlike classification (one label) or localization (one label + one box for an assumed single object).

### 6.1 The problem shape 🎯
- **Detection = classification + localization** sharing one backbone. Two heads, two losses with **different units**: class via **cross-entropy**, box via **L1 / IoU loss**; a weight **λ balances** them (set λ badly → neat boxes on the wrong objects).
- **Set prediction:** two objects have no natural order; tie each output slot to a **location** so geometry fixes the order.
- ⚠️ **Why not slide a classifier everywhere?** a 640×480 image at stride 8 ≈ **57,600 windows**; at 1 ms each ≈ **58 s/frame** — ~1,728× too slow for 30 fps. Both detector families are ways of *avoiding* this.
- **Every detector = Backbone + Neck + Head:** backbone = classifier with its head removed; **neck** (e.g. **FPN**) fuses resolutions so one head sees big *and* small objects; **head** is the only detection-specific part. *"ResNet-50 FPN"* names only a backbone + neck.

### 6.2 Measuring a detector — IoU, PR curve, AP, mAP 🎯
- **IoU** $=\dfrac{\text{area of overlap}}{\text{area of union}}$ (1 = perfect, 0 = disjoint). Pick a threshold → each prediction is a **hit or miss**. 0.5 is a *convention*; IoU falls smoothly but the **verdict flips instantly** at the threshold, so **always quote the threshold** beside a number.
- **Counting:** sort predictions by confidence; match each to the best unclaimed truth box above the IoU bar; **one prediction per object** (a second box on the same person is a False Positive). **Precision** = fraction of your boxes that were right; **Recall** = fraction of objects you found.
- **AP (Average Precision)** = area under the **precision–recall curve** for one class (sweep the confidence threshold). For $C$ classes, the macro-average is

```math
\operatorname{mAP}=\frac{1}{C}\sum_{c=1}^{C}AP_c.
```

With one class, mAP = AP. ⚠️ The mean can sit far above the worst class, so inspect the per-class table.
- **COCO AP** also averages over the 10 IoU thresholds $T=\{0.50,0.55,\ldots,0.95\}$:

```math
\operatorname{AP}_{COCO}=\frac{1}{10C}\sum_{t\in T}\sum_{c=1}^{C}AP_{c,t}.
```

A fixed box may score 96.9 at IoU 0.5 but 68.1 at 0.9 because only the definition of a correct localization changed.

### 6.3 Two-stage detectors — the R-CNN family 🎯
*Propose regions, then classify them.* Three papers removed the bottleneck step by step:

| Model | Idea | Speed |
|-------|------|:-----:|
| **R-CNN** (2014) | ~2,000 region proposals, run the CNN **on every one**, 3 training stages | ~47 s/img |
| **Fast R-CNN** (2015) | run the CNN **once**, crop features per proposal via **RoI pooling** (fixed 7×7 grid, max per cell) | ~2 s |
| **Faster R-CNN** (2015) | **learn** the proposals from the same features with a **Region Proposal Network (RPN)** | ~0.2 s |

- **RPN:** slide one 3×3 conv over the shared feature map; at each location judge a few **anchor boxes** (object-or-not) via two 1×1 siblings (one **scores**, one **nudges** the box); keep top ~1,000, run **NMS**, hand survivors to stage two.
- **Anchors** = preset reference boxes, usually several scale/aspect-ratio combinations at each feature location. For example, 3 scales × 3 aspect ratios gives 9 anchors per location; the exact choices are model hyperparameters, not fixed laws. **Anchor-free** models instead predict locations or distances to four edges. An anchor-based detector predicts **offsets from the reference box** (position normalized by anchor size; size in log space so shrink/grow are symmetric and width stays positive).
- **Label assignment:** IoU with truth **> 0.7 → positive**, **< 0.3 → negative** (background), in-between **ignored**; the box loss applies **only to positives**.

### 6.4 One-stage detectors — YOLO 🎯
*No proposals, no second network — one forward pass + a clean-up step.*
- **YOLO** ("You Only Look Once"): divide the image into cells; the cell containing an object's centre is responsible. In YOLOv1, each cell predicts $B$ boxes, each with five values $(x,y,w,h,\text{confidence})$, plus $C$ class probabilities:

```math
\mathrm{values\ per\ cell}=5B+C.
```

The original YOLOv1 used $B=2$ and $C=20$, hence $5(2)+20=30$ values per cell.
- **YOLOv8** predicts at **three scales** (80×80, 40×40, 20×20 @640 px) = **~8,400 boxes** in one pass; **anchor-free**, **decoupled head** (separate class/box branches), each edge = a distribution over 16 bins; nano ≈ 3.2 M params. **NMS** then reduces thousands of boxes to the few you see.
- **NMS (Non-Max Suppression):** choose an IoU threshold $T_{NMS}$, sort by confidence, keep the top box, and remove remaining boxes whose IoU with it exceeds $T_{NMS}$; repeat. A value such as 0.5 is common, but the correct threshold is task/model dependent. ⚠️ NMS is greedy, not learned, and can suppress two real objects that genuinely overlap.
- ⚠️ **Class imbalance:** almost all candidate locations are easy background, so their gradient can swamp rare objects. If $p_t$ is the model probability assigned to the true class, focal loss is

```math
FL(p_t)=-\alpha_t(1-p_t)^\gamma\log(p_t),\qquad \gamma\ge0.
```

The factor $(1-p_t)^\gamma$ becomes tiny for an easy example with $p_t\approx1$. When $\gamma=0$, focal loss reduces to class-weighted cross-entropy; $\alpha_t$ optionally balances classes.
- ⚠️ **Small objects vanish:** downsampling shrinks a 24-px pedestrian below one cell at stride 32; a **feature pyramid** predicts at several strides to recover them.

### 6.5 Choosing a detector — accuracy, latency, size 🎯
- **Accuracy metric matters:** AP@0.5 barely separates models; **AP@[.5:.95] ranks them all** (a model can lead at 0.5 and trail at 0.8).
- **"Real-time" is a threshold, not a property:** 30 fps ⇒ **33 ms/frame**; the *same* model clears it on GPU but not CPU. **Always quote hardware, image size and batch.**
- **Object size & crowding** hurt strict AP (large ≈98 AP, medium ≈56, small ≈34, averaged over 7 detectors).
- **Modern trade-off has shifted:** a tiny **YOLOv8n** can be ~8× faster on CPU, ~13× smaller, *and* score higher AP than **Faster R-CNN** — the old "two-stage = accurate, one-stage = fast" rule no longer holds. **Pick two of {accuracy, latency, size}.**
- **The detector is not the product:** pipeline = **capture → detect → track → aggregate → act**; one dropped frame can break a track and double a count. Keep mAP as a *diagnostic*; measure the number the business asked for.
- ⚠️ **Ground truth isn't perfect:** incomplete labels make correct detections count as false positives; some "false positives" are real background people. Precision figures are often a **lower bound**.

> 🎯 **EC3 exam shape (from the paper):** (1) compute **IoU** of a predicted vs ground-truth box and decide TP/FP at threshold 0.5; (2) **differentiate R-CNN vs YOLO** and explain **why R-CNN isn't real-time** (thousands of region-CNN passes); (3) list the parameters **YOLOv1** predicts per cell.

✍️ *IoU:* overlap 20, union 60 → IoU = 0.33 < 0.5 → **False Positive**. ✍️ *AP:* in the worked example two duds pull AP to **0.76**, not 1.0.

---

## Deep Learning for Vision: CNN & ViT (EC3) 🎯

### 7.1 CNN essentials
- **Convolutional layer:** learnable kernels slide over the input → feature maps; **weight sharing** + **local connectivity** → translation-equivariant, far fewer params than dense layers.
- A CNN learns its own hierarchy: early layers detect **edges/colour**, middle layers combine them into **textures/parts**, and late layers encode **objects/classes**. Classical ML usually needs a human-designed feature extractor first.
- Typical block: `convolution → ReLU → pooling`, repeated; then **global average pooling (GAP)** or flattening produces a feature vector for the classifier head. GAP turns an $H\times W\times C$ map into $C$ channel averages and avoids a huge dense layer.
- 🧮 **Output size** of a conv layer (per spatial dimension), including dilation $D$:
```math
K_{\mathrm{eff}}=D(K-1)+1,\qquad O = \left\lfloor \frac{W + 2P - K_{\mathrm{eff}}}{S} \right\rfloor + 1.
```
✍️ *Input 21×21, K=3, S=2, P=1:* $O = \lfloor(21-3+2)/2\rfloor + 1 = \lfloor 20/2\rfloor + 1 = 11 \Rightarrow 11\times11$.
- **Stride $S$** skips input positions: larger stride reduces resolution and compute. **Padding $P$** supplies boundary values (usually zeros), preserving edge information and optionally spatial size; for odd $K$, stride 1, dilation 1, `same` size uses $P=(K-1)/2$.
- **Pooling** downsamples → some local invariance + larger effective receptive field. **Max pooling** keeps the strongest local response; **average pooling** keeps the neighbourhood's mean. Pool/window choice is a modelling trade-off, not a universal “max for one task, average for another” rule.
- **Receptive field** = region of input influencing one neuron. Recurrence: $\text{RF}_{l} = \text{RF}_{l-1} + (K_l - 1)\prod_{i<l} S_i$; effective stride (jump) $= \prod_l S_l$. 💡 Bigger receptive field → sees more context → helps detect large objects (but can hurt small-object precision).
- **Cross-entropy loss:** $L = -\sum_c y_c \log \hat{y}_c$ ($y$ = one-hot ground truth, $\hat y$ = softmax prediction). Penalizes confident wrong predictions.
- **Batching** benefits: stable gradient estimates, GPU parallelism (throughput), and BatchNorm statistics.

#### Parameters, compute, and latency are different budgets
- **Parameters** drive model/storage memory; **MACs/FLOPs** approximate arithmetic; **latency** also depends on hardware, memory access, kernels, and batch size.
- Parameters often accumulate in late, wide layers while compute can be dominated by early, large feature maps. A smaller file is therefore not automatically faster on a particular device: benchmark the deployment target.

### 7.2 ResNet and efficient pretrained backbones
Simply stacking more plain layers can make **training error worse**, not just validation error: an optimization/degradation problem rather than ordinary overfitting. A residual block learns a correction
```math
y=F(x)+x,
```
so the identity shortcut gives signals and gradients a direct path; extra layers can learn $F(x)\approx0$ when no refinement is needed.

| Backbone family | Core idea | Relative profile | Good starting point |
|-----------------|-----------|------------------|---------------------|
| **ResNet** | residual/skip connections | strong, dependable baseline; comparatively heavy | accuracy-oriented server/GPU baseline |
| **EfficientNet** | jointly scale depth, width, resolution | balanced accuracy vs compute | constrained training/inference with a balanced budget |
| **MobileNetV3** | depthwise separable convolutions + mobile-aware design | smallest/fastest of these families | phone/edge CPU deployment |

> ⚠️ Exact speed and accuracy depend on model variant, input resolution, framework, and hardware. Choose from a measured **accuracy–latency–memory** Pareto frontier, not architecture name alone.

### 7.3 Transfer learning: freeze or fine-tune? 🎯
A pretrained model contains a reusable **backbone** (feature extractor) plus a task-specific **head**. Replace the head for the new classes, train it first, then optionally unfreeze later backbone stages using a smaller learning rate.

| Labelled data | Domain close to pretraining? | Starting strategy |
|---------------|:---:|-------------------|
| little | yes | freeze backbone; train head |
| plenty | yes | train head, then fine-tune later blocks |
| plenty | no | fine-tune most/all layers carefully |
| little | no | highest-risk case: collect/augment data or use a closer pretrained domain |

Early filters are generic (edges/textures), while late features are task-specific, so progressive unfreezing normally goes **from the head backward**. Always use the preprocessing attached to the pretrained weights; wrong resize/normalization silently damages accuracy.

```python
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
for parameter in model.parameters():
  parameter.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
preprocess = weights.transforms()
```

### 7.4 Classification evaluation beyond accuracy 🎯
For one class, $TP$ is correctly predicted positives, $FP$ is incorrect positive predictions, and $FN$ is missed positives:
```math
\text{Precision}=\frac{TP}{TP+FP},\quad \text{Recall}=\frac{TP}{TP+FN},\quad F_1=\frac{2PR}{P+R}.
```

✍️ If $TP=40$, $FP=10$, $FN=20$: precision $=40/50=0.80$, recall $=40/60\approx0.667$, and $F_1\approx0.727$.

| Aggregate | How | Interpretation |
|-----------|-----|----------------|
| **Macro** | average the per-class metric | every class equal; exposes rare-class failure |
| **Micro** | pool all class decisions, then compute | every example/decision equal; frequent classes dominate |
| **Weighted** | per-class metric weighted by class support | between macro and frequency-dominated reporting |
| **Top-$k$** | correct label appears among $k$ highest scores | useful when a shortlist is genuinely part of the workflow |

> ⚠️ On imbalanced data, high accuracy or micro-F1 can coexist with poor minority-class recall. Report a confusion matrix and macro/per-class metrics when every class matters.

### 7.5 Custom CNN design: cost it before training 🎯
A custom CNN is useful when there is enough **in-domain labelled data** and deployment needs a small, explainable model. With little data, a pretrained backbone usually wins because it already contains transferable visual features.

**A practical layout:** `stem → [conv → BatchNorm → ReLU] blocks → downsample → global average pooling → linear classifier`.

- **Depth** buys receptive field and feature composition; its parameter cost grows roughly linearly when width is fixed.
- **Width** buys channels/capacity, but convolution weights grow approximately quadratically when both input and output width scale.
- **Kernel:** two $3\times3$ convolutions see the same $5\times5$ region as one $5\times5$, but cost $18C^2$ rather than $25C^2$ weights and insert an extra non-linearity.
- **Channels vs resolution:** doubling channels when halving each spatial dimension is a common balance between representation capacity and compute.
- **Head:** global average pooling (GAP) reduces each $H\times W$ channel map to one value. It removes most classifier parameters and some location sensitivity; flattening preserves spatial layout but can create a very large dense head.

For a standard convolution with bias,
```math
\mathrm{parameters}=K_hK_wC_{in}C_{out}+C_{out}.
```
The Lecture 5 baseline reaches a 65-pixel receptive field on a $96\times96$ input: evidence outside a unit's receptive field cannot affect that unit. Choose **depth for sufficient reach first**, then choose width that fits the parameter, memory, and latency budgets.

### 7.6 Layer semantics: normalise, activate, downsample, classify
**Batch Normalisation** normalises each channel over a mini-batch, then learns scale $\gamma$ and shift $\beta$:
```math
\hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat{x}+\beta.
```
It smooths optimisation and permits larger stable learning rates. Training uses batch statistics; inference uses stored running statistics, so the model must be switched to evaluation mode.

- **Activation:** without a non-linearity, stacked linear/convolution layers collapse into one linear map. ReLU is cheap and avoids positive-side saturation, but can create dead units on its zero-gradient side.
- **Downsampling:** max pooling keeps the strongest response, average pooling keeps the mean, and stride-$2$ convolution learns the downsampling operation at an added parameter cost. All discard location detail, which is more dangerous for detection/segmentation than leaf-level classification.
- **Output:** for 38 mutually exclusive diseases, emit 38 **logits** and use softmax cross-entropy. In PyTorch, `nn.CrossEntropyLoss` applies log-softmax internally; adding softmax in the model applies it twice and is incorrect. Use independent sigmoid outputs only for multi-label tasks.

### 7.7 Training and regularisation: diagnose before treating 🎯
| Choice | What to remember |
|--------|------------------|
| **Learning rate** | step size; too low crawls, too high diverges. Sweep logarithmically ($10^{-4},10^{-3},10^{-2}$) before fine tuning |
| **Optimizer** | SGD follows the gradient; momentum accumulates velocity; Adam adapts each parameter's step; AdamW decouples weight decay |
| **Batch size $B$** | larger $B$ reduces gradient noise roughly as $1/\sqrt{B}$ but gives only $N/B$ updates per epoch and costs memory |
| **Label smoothing** | replaces hard one-hot targets with slightly softened targets, reducing pathological overconfidence |
| **Stopping** | select/checkpoint the best epoch on **validation** data; open the test set once for the final report |

**Read the learning curves before choosing a remedy:**
- High **training error** = **underfitting**. Increase useful capacity, train/optimise better, or improve features; more regularisation will not fix inability to fit.
- Low training error but much higher validation error = **overfitting**. The generalisation gap can respond to augmentation, weight decay, dropout, or early stopping.
- **Dropout** randomly removes activations while training and acts like an ensemble of thinner networks. It is often most useful in wide dense heads; a GAP-based CNN has little dense capacity to regularise.
- **Augmentation is an invariance claim:** use a transform only when it preserves the label. A flipped diseased leaf remains diseased, but strong colour jitter may erase colour symptoms and hurt performance.

### 7.8 Honest evaluation and custom-vs-pretrained evidence 🎯
PlantVillage has **54,305 images but only 20,480 distinct leaves**; the same leaf is photographed multiple times. A random image split can put near-duplicates in train and test, letting the model recognise a leaf rather than generalise to a new one. Split by **leaf/group identity**, train on train, choose settings on validation, and report the untouched test result together with the split method.

Lecture 5's controlled comparison gives the reusable decision rule:
- With only **770 training images**, pretraining was worth about **26 accuracy points**.
- With **38,501 in-domain training images**, the custom model led while using about one-fifth the size of the compared pretrained option.
- Accuracy separated the five tested models by less than one point, while size differed by **78×** and CPU latency by **8×**. Choose the cheapest model that clears the required accuracy, macro-F1, worst-class-F1, memory, and latency bars.

> ⚠️ **The split is part of the result.** Report macro/per-class metrics for imbalanced classes and benchmark latency at the actual deployment batch size and hardware.

### 7.9 Vision Transformers (ViT) 🎯
Split the image into **patches** → linear-embed + positional encoding → **self-attention** across all patches.
| | **CNN** | **ViT** |
|--|--------|---------|
| Receptive field | **local** (grows with depth) | **global** from layer 1 (attention) |
| Inductive bias | strong (locality, translation invariance) | weak → needs **more data** |
| Efficiency | more efficient on small/medium data | attention is $O(N^2)$ in #patches → heavier |
| Long-range deps | via many layers | **self-attention** directly models them |
| Translation invariance | **more** (weight sharing + pooling) | less built-in |

### 7.10 Self-supervised learning (EC3)
Learn representations from **unlabelled** images via pretext tasks:
- **SimCLR** — contrastive: pull together two augmentations of the same image, push apart others; needs **large batches** of negatives.
- **MoCo** — contrastive with a **momentum encoder** + a **queue/memory bank** of negatives → large negative set without huge batches.
- **DINO** — **self-distillation, no labels & no negatives**: a student network matches a momentum **teacher**'s output on different views (ViT-based); emergent object segmentation in attention maps.

---

## Exam-thinking playbook from the supplied papers 🎯

The 12-Sep lecturer tutorial says to expect roughly **3–4 numerical questions** and explicitly drills these in order: **grayscale, bit depth, brightness/contrast, Gaussian composition, correlation/convolution, gradient magnitude/direction, convolution output size, convolution parameters, pooling, box IoU, mean IoU, precision/recall/F1, and Dice from IoU**. Learn them as one pipeline: **pixels → filters → features → network dimensions → predictions → metrics**.

| If the question gives... | Recognise | First line to write | Sanity check / trap |
|---|---|---|---|
| RGB values | grayscale/luma | $Y=0.299R+0.587G+0.114B$ | result must remain in the channel range |
| two Gaussian sigmas | composition | $\sigma=\sqrt{\sigma_1^2+\sigma_2^2}$ | variances add, not sigmas |
| signal + asymmetric mask | correlation vs convolution | reverse the mask only for convolution | symmetric masks hide the difference |
| $G_x,G_y$ | edge strength/direction | $|G|=\sqrt{G_x^2+G_y^2}$, $\theta=\operatorname{atan2}(G_y,G_x)$ | gradient is perpendicular to the edge |
| input, kernel, stride, padding | output dimension | $O=\lfloor(W+2P-K_{eff})/S\rfloor+1$ | include both padding sides and floor |
| kernel + input/output channels | learnable parameters | $K_hK_wC_{in}C_{out}+C_{out}$ with bias: kernel weights plus one bias per output channel | input height/width do not enter |
| two boxes or masks | IoU + verdict | intersection first, then $U=A+B-I$ | state TP/FP using the given threshold |
| class IoUs | mIoU | arithmetic mean over every stated class | do not hide a zero class |
| TP, FP, FN | P/R/F1 | label denominators before substitution | F1 is harmonic, not arithmetic mean |

**Exact previous-paper signals:**
- **EC2 (30 marks):** histogram equalisation (3); correlation vs convolution plus Gaussian composition (9); million-point RANSAC (5); smoothed Sobel plus gamma correction (6); Harris response (2); Bayer demosaicing pseudocode (5).
- **EC3 (40 marks):** U-Net + k-means comparison + one mean-shift iteration (9); box IoU + R-CNN/YOLO + YOLOv1 outputs (9); CNN vs ViT (5); cross-entropy + conv output + batching (6); SimCLR/DINO/MoCo (3); receptive field/jump/context trade-off (8).

> 🧠 **Five-pass numerical habit:** (1) name the object and units, (2) write the formula before numbers, (3) track spatial size and channels separately, (4) calculate, (5) interpret the result in one sentence. The interpretation is often a separate mark: "IoU 0.53 > 0.5, therefore TP" or "larger receptive field adds context but can lose small-object detail."

---

## 🧠 One-Page Cheat Sheet
- **CV = inverse, under-constrained** pixels→meaning. Hard: viewpoint/lighting/occlusion/deformation.
- **Image:** grid of pixels / $I(x,y)$; resolution=W×H; bit-depth $2^b$ levels. **RGB** display · **Gray** ($0.299R{+}0.587G{+}0.114B$) speed · **HSV** colour selection.
- **Camera:** perspective $x=fX/Z$; intrinsics $K$ (camera→pixel), extrinsics $[R|t]$ (world→camera).
- **Nyquist:** sample > 2× max frequency; anti-alias **before** downsample.
- **Transforms:** homogeneous $p'=Hp$; affine (6, keeps parallelism) vs homography (8, perspective); $R T\ne T R$.
- **Interpolation:** NN for masks, bilinear default, bicubic sharp (rings).
- **Photometric:** brightness/contrast $\alpha f+\beta$; gamma $f^\gamma$ ($\gamma<1$ brightens); hist-equalisation via CDF; CLAHE.
- **Convolution flips kernel** (correlation doesn't); Gaussian **σ compose** $=\sqrt{\sigma_1^2+\sigma_2^2}$; separable; derivative-of-Gaussian.
- **Edges:** gradient mag $\sqrt{I_x^2+I_y^2}$, dir $\perp$ edge; **Sobel**; **Canny** = DoG→mag/orient→**NMS**→**hysteresis**.
- **Corners (Harris):** $M=\sum[I_x^2, I_xI_y; I_xI_y, I_y^2]$, $R=\det M - k\,\text{tr}^2M$; +corner / −edge / ~0 flat. Rotation-inv, not scale-inv. **SIFT** = scale-invariant 128-D descriptor.
- **RANSAC:** min-sample → fit → count inliers (vectorized) → keep best.
- **Segmentation:** semantic (per-class mask) vs instance (unique ids); threshold / k-means / mean-shift / **U-Net (skip connections, 1 dilation rate)** / **DeepLab (multi-rate dilated/atrous conv)**; segmentation-CNN = **dilated conv + upsampling, no FC**. Metrics: **IoU**, **Dice (DSC)**, pixel/voxel accuracy.
- **Detection:** **IoU**=overlap/union (TP if ≥ threshold; the verdict flips at the bar). **AP**=area under the PR curve, **mAP**=mean over classes, **COCO**=avg AP over IoU 0.50:0.95. Detector = **backbone + neck (FPN) + head**. **Two-stage:** R-CNN → Fast (RoI pooling) → Faster (RPN + anchors). **One-stage:** YOLO (grid, anchor-free, ~8,400 boxes) → **NMS**. **Focal loss** for background imbalance; feature pyramid for small objects. Pick 2 of {accuracy, latency, size}.
- **CNN:** $K_{eff}=D(K-1)+1$, out $=\lfloor(W+2P-K_{eff})/S\rfloor+1$; stride shrinks, padding protects borders; max/avg pooling; receptive field.
- **Backbones:** ResNet uses $F(x)+x$; EfficientNet balances scaling; MobileNet targets edge devices. Parameter count ≠ latency: benchmark hardware.
- **Transfer learning:** replace head → freeze/train head → progressively fine-tune from late blocks; use the pretrained weights' exact transforms.
- **Metrics:** precision $TP/(TP+FP)$, recall $TP/(TP+FN)$, $F_1=2PR/(P+R)$; macro treats classes equally, micro pools decisions.
- **Custom CNN:** params $=K_hK_wC_{in}C_{out}+C_{out}$; pick depth for receptive-field reach, then width for budget; two $3\times3$ layers cost $18C^2$ vs one $5\times5$ at $25C^2$; GAP keeps the head small.
- **Training:** BatchNorm uses batch stats in training/running stats at inference; sweep LR by decades; high train error = underfit, train–validation gap = overfit; regularise the gap, not inability to fit.
- **Evaluation:** augmentation must preserve labels; select on validation, test once; group near-duplicates before splitting. The split is part of the result.
- **ViT:** patches + self-attention → global RF, needs more data; CNN more translation-invariant & efficient.
- **Self-supervised:** SimCLR (contrastive, big batch), MoCo (momentum + queue), DINO (self-distillation, no negatives).

---

## ✅ Self-Test (cover the answers)
1. Why can't we recover depth from a single image? *(3D→2D projection loses the depth ray; under-constrained.)*
2. Convolve an image with σ=4 then σ=3 — equivalent single σ? *($\sqrt{16+9}=5$.)*
3. Correlation vs convolution on a 1D signal with an asymmetric filter — show both. *(Flip filter for convolution.)*
4. Give the 4 Canny steps and say what NMS and hysteresis each do.
5. Given $M$, compute Harris $R$ and classify the pixel (corner/edge/flat).
6. Conv output size for input 21×21, K=3, S=2, P=1. *(11×11.)*
7. Compute IoU for two boxes and decide TP at threshold 0.5.
8. Why are R-CNNs unsuitable for real-time; how does YOLO fix it?
9. Explain U-Net skip connections and contrast with k-means.
10. CNN vs ViT: which has a global receptive field, which is more translation-invariant, which needs more data?
11. Input $6\times6$, kernel 3, stride 1, no padding: output size? What if padding is 1? **Answer:** $4\times4$ without padding; $6\times6$ with padding 1.
12. Why can a model with fewer parameters still run slower? *(Latency also depends on activation sizes, operations, memory traffic, hardware, and implementation.)*
13. Choose a transfer-learning strategy for little data in a nearby domain, then for little data in a distant domain. *(Freeze/train head; collect more data or find a closer pretrained source.)*
14. With $TP=40,FP=10,FN=20$, calculate precision, recall, and $F_1$. *($0.80$, $0.667$, $0.727$.)*
15. Why do residual shortcuts help deep networks? *(They provide an identity/gradient path and let blocks learn residual corrections.)*
16. Count parameters in a $3\times3$ convolution from 32 to 64 channels, with bias. **Answer:** $3\cdot3\cdot32\cdot64+64=18{,}496$.
17. Why do two $3\times3$ layers usually beat one $5\times5$ layer at equal width? *(Same receptive field, $18C^2$ vs $25C^2$ weights, plus an extra non-linearity.)*
18. Training and validation errors are both high. Should you add dropout? *(No: this is underfitting; improve capacity or optimisation first.)*
19. Why is a random image split misleading when several photos show the same leaf? *(Near-duplicates leak identity across splits; group by leaf before splitting.)*
20. For mutually exclusive classes with PyTorch `CrossEntropyLoss`, should the model apply softmax? *(No; pass logits because the loss applies log-softmax.)*
21. Define **AP** and **mAP**; why does COCO average AP over ten IoU thresholds? *(AP = area under the PR curve; mAP = mean AP over classes; COCO averages 0.50–0.95 because "correct" depends on the IoU bar.)*
22. Name the three **R-CNN** generations and the bottleneck each removed. *(R-CNN 2,000 CNN passes → Fast R-CNN one pass + RoI pooling → Faster R-CNN learned RPN proposals.)*
23. What does **NMS** do, and when does it fail? *(Greedily deletes overlapping duplicates by confidence; fails when two real objects genuinely overlap.)*
24. Why does detection need **focal loss** and a **feature pyramid**? *(Down-weight easy background so rare objects count; predict at several strides so small objects survive downsampling.)*
25. A model leads at AP@0.5 but trails at AP@0.8, and is "real-time" on GPU but not CPU. Which two reporting rules does this illustrate? *(Quote the IoU threshold with every AP; quote hardware/image-size/batch — "real-time" is a threshold, not a property.)*
26. Semantic vs **instance** segmentation: which one separates person 1 from person 2? *(Instance — semantic gives every person the same label; instance adds unique ids.)*
27. Why use a **dilated** convolution instead of a larger kernel, and what's the cost? *(Larger field of view at constant parameters; cost = it skips pixels/approximates, so tune the rate or small objects are missed.)*
28. Write **IoU** and **Dice** for masks with intersection 20, pred 50, truth 30. *(IoU $=20/60=0.33$; Dice $=40/80=0.5$.)*

---

## 📈 How to extend this note
Append a dated `### Update Log — YYYY-MM-DD` below with each new lecture; add rows to the Syllabus map. Likely upcoming (per EC3): optical flow/tracking, stereo & depth, image captioning/VLMs, diffusion/generative vision, agentic vision (course finale).

## Update Log
- **2026-09-16** — Audited mathematical readability and correctness. Distinguished centred projection coordinates from pixels, stated sampling assumptions, defined Harris and mean-shift symbols, completed segmentation/detection metrics and losses, and expanded anchor, YOLO, NMS and convolution-parameter formulas with interpretations.
- **2026-09-12** — Integrated Lectures 7–8, the full numerical tutorial, the 12-Sep transcript, and exact EC2/EC3 papers. Added the output-first semantic/instance/panoptic mental model; U-Net vs DeepLab mechanism and evidence; Dice↔IoU conversion, imbalance-aware losses, boundary evaluation; Mask R-CNN, RoI Align, mask AP and SAM; the complete numerical drill ladder; and paper-derived question recognition/solving patterns.
- **2026-09-06 (Segmentation lecture)** — Expanded **Segmentation** into a full section: the classification→detection→segmentation task ladder, **semantic vs instance**, the **segmentation-CNN anatomy** (dilated conv + upsampling, no FC head), **dilated / atrous convolution** (larger field of view at constant parameters; rate tuning for object size), **U-Net (single rate) vs DeepLab (multiple parallel rates)**, and pixel-level evaluation — **IoU / Dice (DSC) / pixel & voxel accuracy** with foreground-vs-boundary and instance-overlap notes. Added cheat-sheet, syllabus-map, and self-test items.
- **2026-09-03 (Lecture 6 + webinar)** — Expanded **Object Detection** into a full lecture: task shapes, backbone/neck/head anatomy, sliding-window infeasibility, IoU / PR-curve / **AP** / **mAP** / COCO IoU-averaging, the **R-CNN** family (RoI pooling, RPN, anchors vs anchor-free, offset & log-space box regression, positive/negative assignment), **YOLO** one-stage (multi-scale grid, ~8,400 boxes, decoupled head), **NMS**, **focal loss**, feature pyramids for small objects, and accuracy/latency/size detector selection with annotation-quality caveats. Also captured the hands-on lab webinar (pixels/patches, RGB/gray/HSV, sampling/quantisation, H-matrix transforms, interactive Canny/Harris).
- **2026-08-22 (Lecture 5)** — Added custom CNN design from the fifth-class transcript and slides: parameter/receptive-field budgeting, depth–width–kernel–head trade-offs, BatchNorm and output semantics, learning rate/optimizers/batch size, underfit-vs-overfit diagnosis, regularisation, augmentation invariance, group-aware splits, and evidence-based custom-vs-pretrained selection.
- **2026-08-22** — Added the 21-Aug CNN lecture and transcript: feature hierarchies, stride/padding/dilation, pooling, receptive fields, compute budgets, ResNet degradation and shortcuts, EfficientNet/MobileNet trade-offs, transfer-learning strategy, pretrained transforms, and classification metrics.
- **2026-08-18** — Initial note from Lectures 1–3 (image basics, camera & transforms, edges/corners), class transcripts, and EC2/EC3 papers. Extended with segmentation, detection, CNN/ViT and self-supervised learning to match exam scope.
