# Multi-Head Attention Transformer for Filipino Sign Language Recognition

## Comprehensive System Methodology and Architecture Documentation

---

## Table of Contents

1. [Research Design](#1-research-design)
2. [Data Sources: FSL-105 Dataset](#2-data-sources-fsl-105-dataset)
3. [Research Instrument: System Architecture](#3-research-instrument-system-architecture)
4. [Data Generation Procedure](#4-data-generation-procedure)
5. [Data Analysis](#5-data-analysis)

---

## 1. Research Design

### 1.1 Research Framework

This study employs a **quantitative experimental framework** designed to systematically compare two competing model architectures for Filipino Sign Language Recognition:

1. **MHAM-based Transformer** (Proposed model)
2. **InceptionV3-GRU** (Baseline model)

Both models are evaluated under identical conditions using the same dataset, preprocessing pipeline, and training configuration to ensure a fair and reliable comparison.

### 1.2 Variables

#### Independent Variable

**Model Architecture** with two levels:

- Transformer with Multi-Head Attention Mechanism (MHAM)
- InceptionV3-GRU hybrid architecture

#### Dependent Variables

The study measures four distinct aspects of model performance:

**1. Recognition Performance (Without Occlusion)**

- Ability to identify correct glosses from clean sign videos
- Metrics: Precision, Recall, F₁-score

**2. Recognition Performance (With Occlusion)**

- Ability to identify correct glosses when facial landmarks are obscured
- Metrics: Precision, Recall, F₁-score

**3. Classification Performance (Without Occlusion)**

- Ability to categorize signs into semantic groups (clean videos)
- Metrics: Precision, Recall, F₁-score

**4. Classification Performance (With Occlusion)**

- Ability to categorize signs into semantic groups (occluded videos)
- Metrics: Precision, Recall, F₁-score

### 1.3 Experimental Conditions

**Training Phase:**

- Both models trained from scratch
- Same 80/20 train-test split
- Identical hyperparameters where applicable
- Same preprocessing pipeline
- Same optimization strategy (Adam optimizer)

**Evaluation Phase:**
The test set is automatically divided into two subsets based on keypoint visibility:

```
Test Set (426 samples)
├── Non-occluded subset
│   └── Videos where all facial landmarks remain visible
└── Occluded subset
    └── Videos where critical facial features (eyes, nose, mouth) are temporarily hidden
```

### 1.4 Hypothesis Testing

Four null hypotheses are tested:

**H₀₁:** No significant difference in recognition performance without occlusion  
**H₀₂:** No significant difference in recognition performance with occlusion  
**H₀₃:** No significant difference in classification performance without occlusion  
**H₀₄:** No significant difference in classification performance with occlusion

Statistical significance is assessed using:

- Paired t-tests (if differences are normally distributed)
- Wilcoxon signed-rank tests (if differences are non-normal)
- Significance level: α = 0.05
- Multiple comparison correction: Holm-Bonferroni method

---

## 2. Data Sources: FSL-105 Dataset

### 2.1 Dataset Overview

The FSL-105 dataset is a **publicly available, pre-annotated video corpus** specifically created for Filipino Sign Language recognition research. It was developed by Tupal (2023) and made available through Mendeley Data.

**Key Statistics:**

- **Total videos:** 2,130 samples
- **Sign vocabulary:** 105 distinct FSL glosses
- **Semantic categories:** 10 thematic groups
- **Signers:** 4 native Deaf FSL users
- **Recording duration:** 4 seconds per video
- **Recording quality:** 1920×1080 pixels at 60 FPS

### 2.2 Dataset Composition

#### Sign Distribution by Category

| Category          | Description             | Number of Signs | Example Signs                                                            |
| ----------------- | ----------------------- | --------------- | ------------------------------------------------------------------------ |
| **Greetings**     | Social interactions     | 10              | Good morning, Hello, How are you, Nice to meet you                       |
| **Survival**      | Essential communication | 10              | Yes, No, I understand, I know, Slow, Fast, Correct, Wrong                |
| **Numbers**       | Counting                | 10              | One, Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten                |
| **Calendar**      | Months                  | 9               | January, February, March, April, May, June, July, August, September      |
| **Days**          | Time references         | 10              | Monday, Tuesday, Wednesday, Thursday, Friday, Today, Tomorrow, Yesterday |
| **Family**        | Kinship terms           | 10              | Father, Mother, Son, Daughter, Grandfather, Grandmother, Parents, Cousin |
| **Relationships** | Social categories       | 10              | Boy, Girl, Woman, Man, Deaf, Hard of Hearing, Blind, Married, Single     |
| **Colors**        | Color vocabulary        | 10              | Red, Blue, Green, Yellow, Black, White, Orange, Brown, Pink, Purple      |
| **Food**          | Food items              | 10              | Rice, Bread, Egg, Fish, Meat, Chicken, Vegetables, Fruit, Spaghetti      |
| **Drinks**        | Beverages               | 16              | Coffee, Tea, Juice, Milk, Water, Hot, Cold, Sugar, No sugar, Beer, Wine  |

**Total:** 105 unique signs across 10 categories

#### Sample Distribution

Each sign was recorded multiple times by different signers to capture natural variation:

```
Per sign distribution:
- Signer 1: ~5-6 samples
- Signer 2: ~5-6 samples
- Signer 3: ~5-6 samples
- Signer 4: ~5-6 samples

Total per sign: ~20 samples
Total dataset: 105 signs × ~20 samples = 2,130 videos
```

### 2.3 Participant Information

#### Signer Demographics

**Number of Signers:** 4 individuals

**Gender Distribution:**

- Male: 2 signers (including 1 FSL expert)
- Female: 2 signers

**Language Proficiency:**

- All signers are **Deaf**
- All signers are **native FSL users**
- All signers are **fluent** in Filipino Sign Language

#### Expert Validation

**Sir Rey Alfred Lee** (FSL Expert) served dual roles:

1. **Participant:** Performed signs as one of the four signers
2. **Validator:** Ensured all signs reflect authentic, community-accepted FSL usage

This expert-guided approach ensures:

- **Linguistic authenticity:** Signs match real-world FSL usage
- **Cultural accuracy:** Signs respect Deaf community conventions
- **Consistency:** All signers use standardized sign forms

#### Ethical Considerations

**Informed Consent:**

- All participants were fully informed about the dataset's purpose
- Written consent was obtained from each signer
- A trusted FSL interpreter facilitated the consent process
- Participants understood their recordings would be used for research and education

**Community Engagement:**

- Selection prioritized ethical engagement with the Filipino Deaf community
- Participants were chosen through community-informed processes
- Emphasis on trust-building and cultural sensitivity

### 2.4 Recording Specifications

#### Hardware Setup

**Recording Device:** iPhone SE 2020

**Technical Specifications:**

- **Resolution:** 1920 × 1080 pixels (Full HD)
- **Frame Rate:** 60 frames per second (FPS)
- **Video Duration:** 4 seconds per clip
- **Color Space:** RGB
- **File Format:** .mp4 (H.264 encoding)

#### Recording Environment

**Controlled Conditions:**

- **Background:** Plain, solid color (typically blue or green)
- **Lighting:** Consistent, even illumination
- **Camera Position:** Front-facing, fixed position
- **Framing:** Upper body and hands fully visible
- **Distance:** Standardized signer-to-camera distance

**Why Controlled Environment?**

- Minimizes environmental distractions
- Reduces background variations that could bias the model
- Ensures consistent lighting across all recordings
- Facilitates background removal in preprocessing

### 2.5 Sign Language Features Captured

The FSL-105 dataset captures the **multimodal nature** of sign language communication:

#### Manual Features (Hand-based)

**Hand Shape:**

- Finger configurations (open palm, fist, pointing, etc.)
- Thumb position relative to fingers
- Hand orientation (palm up, palm down, sideways)

**Hand Movement:**

- Trajectory paths (straight, circular, arc-shaped)
- Movement direction (up, down, forward, backward)
- Movement speed and acceleration
- Repetition patterns

**Hand Location:**

- Position relative to body (chest, face, neutral space)
- Contact points (hand touching face, head, other hand)
- Spatial relationships between hands

#### Non-Manual Features (Beyond Hands)

**Facial Expressions:**

- Eyebrow position (raised, lowered, neutral)
- Eye wideness (wide open, squinted, normal)
- Mouth shape (open, closed, rounded, stretched)
- Cheek puffing or sucking

**Head Movements:**

- Nodding (yes, affirmation)
- Head shaking (no, negation)
- Tilting (questions, emphasis)
- Forward/backward movement

**Upper Body Posture:**

- Shoulder position
- Torso orientation
- Body leaning (forward for emphasis, back for negation)

**Why Non-Manual Features Matter:**

In FSL (and most sign languages), **facial expressions and body movements are grammatically essential**, not merely emotional displays. For example:

```
Same hand gesture + Different facial expression = Different meaning

Example:
Hand gesture: Pointing forward
+ Neutral face → "That" (demonstrative)
+ Eyebrows raised + Eyes wide → "Really?!" (question with surprise)
+ Eyebrows furrowed + Mouth tight → "That one!" (emphasis/insistence)
```

Many signs in the FSL-105 dataset differ **only** in non-manual markers, making facial landmark tracking critical for accurate recognition.

### 2.6 Data Split

The dataset is divided into **training** and **testing** sets using an 80/20 split:

#### Training Set

- **Total samples:** 1,704 videos (80%)
- **Per-sign samples:** ~16-17 videos per sign
- **Purpose:** Model learning and parameter optimization
- **Signer representation:** All 4 signers represented

#### Testing Set

- **Total samples:** 426 videos (20%)
- **Per-sign samples:** ~4 videos per sign
- **Purpose:** Model evaluation and performance assessment
- **Signer representation:** All 4 signers represented

**Why This Split?**

- 80/20 is standard in machine learning for moderate-sized datasets
- Ensures sufficient training data while maintaining a robust test set
- Balances the need for model learning with reliable evaluation

#### Split Strategy

**Stratified Sampling:**

```
For each of the 105 signs:
├── Take 80% of samples → Training set
└── Take 20% of samples → Testing set

This ensures:
✓ All 105 signs represented in both sets
✓ All 4 signers represented in both sets
✓ Class balance maintained
```

**No Temporal Leakage:**

- No overlap between training and testing samples
- Each video appears in exactly one set
- Models never see test data during training

---

## 3. Research Instrument: System Architecture

### 3.1 Overall System Design

The research instrument is the **complete pipeline** that transforms raw FSL videos into sign predictions. It consists of two major components:

1. **Preprocessing Pipeline:** Converts videos to normalized keypoint sequences
2. **Transformer Architecture:** Processes keypoint sequences to generate predictions

```
┌─────────────────────────────────────────────────────────┐
│                    SYSTEM OVERVIEW                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Raw Video (1920×1080, 60fps, 4 seconds)               │
│          ↓                                              │
│  ┌────────────────────────────────────────┐            │
│  │   PREPROCESSING PIPELINE               │            │
│  │   1. Downsampling                      │            │
│  │   2. Background Removal                │            │
│  │   3. Keypoint Extraction (78 points)   │            │
│  │   4. Coordinate Normalization          │            │
│  │   5. Sequence Formation (T×156)        │            │
│  └────────────────────────────────────────┘            │
│          ↓                                              │
│  ┌────────────────────────────────────────┐            │
│  │   TRANSFORMER ARCHITECTURE             │            │
│  │   1. Linear Embedding (156→768)        │            │
│  │   2. Positional Encoding               │            │
│  │   3. 6 Encoder Blocks                  │            │
│  │      - Multi-Head Attention (8 heads)  │            │
│  │      - Feed-Forward Networks           │            │
│  │   4. CLS Token Pooling                 │            │
│  │   5. Dual Classification Heads         │            │
│  └────────────────────────────────────────┘            │
│          ↓                                              │
│  Predictions: Gloss (1 of 105) + Category (1 of 10)   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Preprocessing Pipeline

The preprocessing pipeline transforms raw video into a structured sequence of normalized keypoint coordinates suitable for neural network processing.

#### Step 1: Video Downsampling

**Input:**

- Resolution: 1920×1080 pixels
- Frame rate: 60 FPS
- Duration: 4 seconds
- Total frames: 240 frames

**Process:**

```python
# Resize spatial dimensions
target_width = 640
target_height = 360
frame_resized = cv2.resize(frame, (target_width, target_height))

# Reduce temporal resolution
original_fps = 60
target_fps = 30
frame_skip = original_fps // target_fps  # Skip every 2nd frame
```

**Output:**

- Resolution: 640×360 pixels
- Frame rate: 30 FPS
- Duration: 4 seconds
- Total frames: 120 frames

**Rationale:**

1. **Computational Efficiency:** Reduces data by ~80% (240 frames → 120 frames, 1920×1080 → 640×360)
2. **Sufficient Temporal Resolution:** 30 FPS captures sign language dynamics (most gestures occur over 0.5-2 seconds)
3. **Keypoint Accuracy Maintained:** MediaPipe performs reliably at 640×360 resolution

#### Step 2: Background Removal

**Tool:** MediaPipe Selfie Segmentation

**Process:**

```
For each frame:
1. Input: 640×360 RGB frame
2. Run segmentation model
3. Output: Binary mask (person=1, background=0)
4. Apply mask to frame
5. Result: Signer isolated, background = solid color or transparent
```

**Technical Details:**

**Segmentation Model:**

- **Type:** Lightweight semantic segmentation CNN
- **Output:** Per-pixel probability of being "person"
- **Threshold:** Pixels with probability > 0.5 classified as person
- **Inference Speed:** ~10ms per frame on CPU

**Mask Application:**

```python
# Get segmentation mask
mask = selfie_segmentation.process(frame)

# Threshold to binary
binary_mask = (mask > 0.5).astype(np.uint8)

# Apply mask
frame_isolated = frame * binary_mask[:, :, np.newaxis]

# Optional: Replace background with solid color
background_color = [0, 0, 0]  # Black
frame_final = frame_isolated + (1 - binary_mask[:, :, np.newaxis]) * background_color
```

**Benefits:**

- **Focus on Signer:** Removes irrelevant environmental features
- **Generalization:** Model doesn't learn background-specific patterns
- **Invariance:** Same signer in different locations produces similar features
- **Reduced Complexity:** Simpler input for keypoint detection

#### Step 3: Keypoint Extraction

This is the **most critical step** in the preprocessing pipeline. Instead of processing raw pixels, we extract structured skeletal landmarks representing the signer's body.

##### 3.2.1 MediaPipe Pose (Upper Body Landmarks)

**Purpose:** Track major body joints and facial reference points

**Keypoints Extracted:** 25 upper-body landmarks

**Landmark Groups:**

**Face (11 points):**

```
0: Nose
1: Left eye (inner)
2: Left eye
3: Left eye (outer)
4: Right eye (inner)
5: Right eye
6: Right eye (outer)
7: Left ear
8: Right ear
9: Mouth (left)
10: Mouth (right)
```

**Torso & Arms (14 points):**

```
11: Left shoulder
12: Right shoulder
13: Left elbow
14: Right elbow
15: Left wrist
16: Right wrist
17: Left pinky (base)
18: Right pinky (base)
19: Left index (base)
20: Right index (base)
21: Left thumb (base)
22: Right thumb (base)
23: Left hip
24: Right hip
```

**Output Format:**

```python
pose_landmarks = {
    'landmark_0': {'x': 0.512, 'y': 0.234, 'z': -0.123, 'visibility': 0.98},
    'landmark_1': {'x': 0.498, 'y': 0.221, 'z': -0.118, 'visibility': 0.95},
    # ... 23 more landmarks
}
```

**Note:** We only use `(x, y)` coordinates, discarding `z` and `visibility` for this study.

##### 3.2.2 MediaPipe Hands (Hand Landmarks)

**Purpose:** Track detailed finger and palm positions

**Keypoints Per Hand:** 21 landmarks

**Hand Structure:**

```
Wrist (1 point):
0: Wrist base

Thumb (4 joints):
1: Thumb CMC (carpometacarpal)
2: Thumb MCP (metacarpophalangeal)
3: Thumb IP (interphalangeal)
4: Thumb tip

Index Finger (4 joints):
5: Index MCP
6: Index PIP (proximal interphalangeal)
7: Index DIP (distal interphalangeal)
8: Index tip

Middle Finger (4 joints):
9: Middle MCP
10: Middle PIP
11: Middle DIP
12: Middle tip

Ring Finger (4 joints):
13: Ring MCP
14: Ring PIP
15: Ring DIP
16: Ring tip

Pinky (4 joints):
17: Pinky MCP
18: Pinky PIP
19: Pinky DIP
20: Pinky tip
```

**Bilateral Tracking:**

- **Left hand:** 21 landmarks
- **Right hand:** 21 landmarks
- **Total hand keypoints:** 42 landmarks

**Output Format:**

```python
hands_landmarks = {
    'left_hand': [
        {'x': 0.423, 'y': 0.556},  # Wrist
        {'x': 0.431, 'y': 0.542},  # Thumb CMC
        # ... 19 more points
    ],
    'right_hand': [
        {'x': 0.587, 'y': 0.563},  # Wrist
        {'x': 0.579, 'y': 0.549},  # Thumb CMC
        # ... 19 more points
    ]
}
```

##### 3.2.3 Facial Landmarks (Critical Non-Manual Markers)

**Purpose:** Capture facial expressions essential for FSL grammar

**Keypoints:** 11 carefully selected facial landmarks

These 11 points are a **subset** of MediaPipe Face Mesh (which provides 468 points), specifically chosen to capture the most important non-manual features:

```
Eyes (4 points):
- Left eye center
- Right eye center
- Left eyebrow peak
- Right eyebrow peak

Nose (1 point):
- Nose tip

Mouth (4 points):
- Left mouth corner
- Right mouth corner
- Upper lip center
- Lower lip center

Chin (1 point):
- Chin tip

Forehead (1 point):
- Forehead reference
```

**Why These 11 Points?**

These landmarks efficiently capture the **key non-manual markers** in FSL:

1. **Eyebrow Raises:** Grammatical marker for yes/no questions
2. **Eye Widening:** Indicates surprise, emphasis, or intensity
3. **Mouth Shapes:** Phonetic components (mouthings) and emotional affect
4. **Head Position:** Tracked via nose-chin-forehead triangle

**Example: "Really?" vs "That"**

```
Sign: Point forward with index finger

"That" (neutral statement):
- Eyebrows: neutral position
- Eyes: normal width
- Mouth: closed/relaxed

"Really?!" (question with surprise):
- Eyebrows: raised high
- Eyes: wide open
- Mouth: slightly open
→ These 11 keypoints capture this distinction
```

##### 3.2.4 Total Keypoint Count

**Keypoint Summary:**

```
MediaPipe Pose:      25 landmarks
MediaPipe Hands:     42 landmarks (21 per hand)
Facial Landmarks:    11 landmarks
────────────────────────────────────
TOTAL:               78 landmarks
```

**Coordinate Representation:**

```
Each landmark: (x, y) coordinates
Total features per frame: 78 × 2 = 156 values
```

#### Step 4: Coordinate Normalization

Raw keypoint coordinates are in pixel space and vary based on:

- Signer's position in frame
- Distance from camera
- Frame dimensions

**Normalization** makes coordinates **translation-invariant** and **scale-invariant**.

##### Process

**Raw Coordinates:**

```
x ∈ [0, 640] pixels (frame width)
y ∈ [0, 360] pixels (frame height)
```

**Step 1: Center coordinates**

```python
frame_center_x = 640 / 2 = 320
frame_center_y = 360 / 2 = 180

x_centered = x - frame_center_x
y_centered = y - frame_center_y
```

Now coordinates are relative to frame center:

```
x_centered ∈ [-320, 320]
y_centered ∈ [-180, 180]
```

**Step 2: Scale to [-1, 1]**

```python
x_normalized = x_centered / frame_center_x
y_normalized = y_centered / frame_center_y
```

**Final normalized range:**

```
x_normalized ∈ [-1, 1]
y_normalized ∈ [-1, 1]
```

**Coordinate System:**

```
(-1, -1)  ────────────  (1, -1)
   │                        │
   │         (0, 0)         │  ← Frame center
   │                        │
(-1, 1)   ────────────  (1, 1)
```

**Example:**

```
Original keypoint (nose): (320, 180) pixels
After centering: (0, 0)
After normalization: (0.0, 0.0)  ← Nose at frame center

Original keypoint (left hand): (160, 270) pixels
After centering: (-160, 90)
After normalization: (-0.5, 0.5)  ← Hand at top-left
```

**Benefits:**

1. **Translation Invariance:** Signer can be anywhere in frame
2. **Scale Invariance:** Distance from camera doesn't matter
3. **Numerical Stability:** Smaller values → better gradient flow
4. **Generalization:** Model learns positional relationships, not absolute locations

#### Step 5: Sequence Formation

After processing all frames, we organize keypoints into a time-ordered sequence.

**Single Frame Vector:**

```python
frame_t = [
    # Pose landmarks (25 × 2 = 50 values)
    nose_x, nose_y,
    left_eye_inner_x, left_eye_inner_y,
    left_eye_x, left_eye_y,
    # ... 22 more pose landmarks

    # Left hand (21 × 2 = 42 values)
    left_wrist_x, left_wrist_y,
    left_thumb_cmc_x, left_thumb_cmc_y,
    # ... 19 more left hand landmarks

    # Right hand (21 × 2 = 42 values)
    right_wrist_x, right_wrist_y,
    right_thumb_cmc_x, right_thumb_cmc_y,
    # ... 19 more right hand landmarks

    # Face (11 × 2 = 22 values)
    left_eye_center_x, left_eye_center_y,
    # ... 10 more facial landmarks
]

Shape: (156,)  ← One-dimensional vector
```

**Video Sequence:**

```python
video_sequence = [
    frame_0,   # (156,)
    frame_1,   # (156,)
    frame_2,   # (156,)
    # ...
    frame_119  # (156,)
]

Shape: (120, 156)  ← Two-dimensional matrix
```

**Interpretation:**

- **Rows (120):** Time steps (frames)
- **Columns (156):** Spatial features (keypoint coordinates)

**This sequence is now ready** to be fed into the Transformer architecture.

### 3.3 Transformer Architecture

The Transformer is the **core learning component** that processes keypoint sequences and generates predictions.

#### 3.3.1 Input Embedding Layer

**Purpose:** Transform raw 156-dimensional keypoint vectors into rich 768-dimensional learned representations.

**Architecture:**

```python
embedding_layer = nn.Linear(in_features=156, out_features=768)
```

**Parameters:**

- Weight matrix: **W** ∈ ℝ^(768 × 156)
- Bias vector: **b** ∈ ℝ^768
- Total parameters: (768 × 156) + 768 = **120,576 parameters**

**Mathematical Operation:**

```
For each frame t:
e_t = W · x_t + b

where:
x_t ∈ ℝ^156  (input keypoint vector)
e_t ∈ ℝ^768  (output embedding)
```

**Sequence Processing:**

```
Input:  X ∈ ℝ^(120 × 156)  (120 frames, 156 features each)
Output: E ∈ ℝ^(120 × 768)  (120 frames, 768 embedding dims)
```

**Why 768 Dimensions?**

1. **Standard in Transformers:** BERT, GPT, and ViT use 768
2. **Expressiveness:** Larger space captures more nuanced patterns
3. **Divisibility:** 768 / 8 heads = 96 dimensions per head (clean division)
4. **Proven Performance:** Extensive research validates this dimension size

**What Does Embedding Learn?**

The embedding layer learns to **encode spatial relationships** between keypoints:

- Hand proximity to face
- Finger configurations
- Symmetry between left and right hands
- Body-hand-face coordination

Example learned patterns:

```
High embedding activation might indicate:
- "Both hands are at chest level" → Certain embedding dimensions activate
- "Right hand touching nose" → Different dimensions activate
- "Eyebrows raised + hands moving" → Another pattern of dimensions
```

#### 3.3.2 Positional Encoding

**Problem:** Transformers process sequences in parallel, so they have **no inherent notion of order**.

Without positional information:

```
Frame sequence: [0, 1, 2, 3, ..., 119]
Transformer sees: [?, ?, ?, ?, ..., ?]  ← All positions equivalent!
```

**Solution:** Add position-dependent patterns to embeddings.

##### Sinusoidal Positional Encoding

**Formula:**

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

where:
pos: Position in sequence (0, 1, 2, ..., 119)
i: Dimension index (0, 1, 2, ..., 383)
d_model: Embedding dimension (768)
```

**Intuition:**

Think of positional encoding as a **unique barcode** for each position:

```
Position 0:   [sin(0/10000^0), cos(0/10000^0), sin(0/10000^(2/768)), ...]
Position 1:   [sin(1/10000^0), cos(1/10000^0), sin(1/10000^(2/768)), ...]
Position 2:   [sin(2/10000^0), cos(2/10000^0), sin(2/10000^(2/768)), ...]
...

Each position gets a UNIQUE 768-dimensional pattern
```

**Why Sine and Cosine?**

1. **Periodic:** Repeating patterns at different frequencies
2. **Smooth:** Continuous functions → similar positions have similar encodings
3. **Relative Position:** Model can learn "frame X is Y steps from frame Z"

**Frequency Spectrum:**

```
Low dimensions (i=0, 1, 2):
- High frequency (changes rapidly with position)
- Captures fine-grained temporal order

High dimensions (i=380, 381, 382, 383):
- Low frequency (changes slowly)
- Captures coarse temporal structure
```

**Example Calculation (Position 0, First 4 Dimensions):**

```python
pos = 0
d_model = 768

# Dimension 0 (i=0, even → sine)
PE(0, 0) = sin(0 / 10000^(0/768)) = sin(0) = 0

# Dimension 1 (i=0, odd → cosine)
PE(0, 1) = cos(0 / 10000^(0/768)) = cos(0) = 1

# Dimension 2 (i=1, even → sine)
PE(0, 2) = sin(0 / 10000^(2/768)) = sin(0) = 0

# Dimension 3 (i=1, odd → cosine)
PE(0, 3) = cos(0 / 10000^(2/768)) = cos(0) = 1
```

**Adding Positional Encoding:**

```python
# Embedding from previous layer
E = [e_0, e_1, e_2, ..., e_119]  # (120, 768)

# Positional encodings
PE = [pe_0, pe_1, pe_2, ..., pe_119]  # (120, 768)

# Element-wise addition
Z = E + PE

# Now each frame has:
z_t = e_t + pe_t
```

**Result:** Each frame embedding now carries **both content** (from linear layer) and **position** (from positional encoding).

#### 3.3.3 Encoder Block Structure

The Transformer contains **6 identical encoder blocks** stacked sequentially. Each block has the same internal structure.

**Single Encoder Block:**

```
Input: X ∈ ℝ^(120 × 768)
         ↓
┌────────────────────────────────┐
│ Multi-Head Self-Attention      │
│ (8 heads, 96 dims each)        │
│ Captures frame-to-frame        │
│ relationships                  │
└────────────────────────────────┘
         ↓
    Residual Add
         ↓
    Layer Norm
         ↓
┌────────────────────────────────┐
│ Position-wise Feed-Forward     │
│ Layer 1: 768 → 3072 (ReLU)    │
│ Layer 2: 3072 → 768            │
│ Non-linear transformation      │
└────────────────────────────────┘
         ↓
    Residual Add
         ↓
    Layer Norm
         ↓
Output: Y ∈ ℝ^(120 × 768)
```

##### Sub-Component 1: Multi-Head Self-Attention

**Detailed explanation in Section 3.4**

High-level purpose:

- Each frame "attends to" every other frame
- Learns which frames are most relevant to each other
- Captures temporal dependencies and patterns

##### Sub-Component 2: Residual Connection

**Operation:**

```python
output = input + sublayer_output
```

**Why Residual Connections?**

Without residuals in a 6-layer network:

```
Input → Block1 → Block2 → Block3 → Block4 → Block5 → Block6 → Output
           ↓        ↓        ↓        ↓        ↓        ↓
        Gradients get weaker with each layer (vanishing gradient)
```

With residuals:

```
Input ─→ Block1 ─→ Block2 ─→ Block3 ─→ Block4 ─→ Block5 ─→ Block6 ─→ Output
  │        ↓         ↓         ↓         ↓         ↓         ↓
  └───────→ + ──────→ + ──────→ + ──────→ + ──────→ + ──────→ +

Direct gradient paths bypass each block
```

**Benefits:**

1. **Gradient Flow:** Gradients can flow directly from output to input
2. **Easier Training:** Network learns "refinements" rather than full transformations
3. **Deep Networks:** Enables training of 6+ layers without degradation

##### Sub-Component 3: Layer Normalization

**Formula:**

```
LayerNorm(x) = γ ⊙ (x - μ) / √(σ² + ε) + β

where:
μ = mean(x)           ← Average across feature dimension
σ² = variance(x)      ← Variance across feature dimension
γ = learnable scale   ← Initialized to 1
β = learnable shift   ← Initialized to 0
ε = 1e-5              ← Numerical stability constant
⊙ = element-wise multiplication
```

**Example (Single Frame):**

```
Input: x = [0.5, 1.2, -0.3, 0.8, ...]  (768 values)

Step 1: Compute statistics
μ = mean(x) = 0.4
σ² = var(x) = 0.25

Step 2: Normalize
x_norm = (x - 0.4) / √(0.25 + 1e-5)
x_norm = (x - 0.4) / 0.5
x_norm = [0.2, 1.6, -1.4, 0.8, ...]

Step 3: Scale and shift (γ=1, β=0 initially)
output = 1 · x_norm + 0 = x_norm
```

**Why Layer Norm?**

1. **Stabilizes Training:** Prevents activation values from growing too large or small
2. **Faster Convergence:** Reduces internal covariate shift
3. **Works with Variable Sequences:** Unlike batch norm, doesn't depend on batch size

**Layer Norm vs Batch Norm:**

```
Batch Norm: Normalizes across batch dimension
            [Sample 1, Sample 2, ..., Sample 16] → Compute mean/var
            Problem: Depends on batch size, doesn't work well with sequences

Layer Norm: Normalizes across feature dimension
            [Feature 1, Feature 2, ..., Feature 768] → Compute mean/var
            Advantage: Independent of batch size, perfect for sequences
```

##### Sub-Component 4: Position-wise Feed-Forward Network (FFN)

**Architecture:**

```python
FFN = nn.Sequential(
    nn.Linear(768, 3072),  # Expansion layer
    nn.ReLU(),              # Activation
    nn.Linear(3072, 768)    # Compression layer
)
```

**Mathematical Operation:**

```
FFN(x) = W_2 · ReLU(W_1 · x + b_1) + b_2

where:
W_1 ∈ ℝ^(3072 × 768)  ← Expansion weights
b_1 ∈ ℝ^3072           ← Expansion bias
W_2 ∈ ℝ^(768 × 3072)  ← Compression weights
b_2 ∈ ℝ^768            ← Compression bias
```

**Processing Flow:**

```
Input:  x ∈ ℝ^768      (one frame embedding)
         ↓
Layer 1: 768 → 3072    (expansion)
         ↓
ReLU:    max(0, x)     (non-linearity)
         ↓
Layer 2: 3072 → 768    (compression)
         ↓
Output:  y ∈ ℝ^768     (transformed embedding)
```

**Why This Architecture?**

**1. Expansion-Compression Creates Representational Bottleneck:**

```
768 dims → 3072 dims → 768 dims
         ↑             ↑
    Expand space    Compress back

In expanded space:
- More dimensions allow richer transformations
- Non-linear combinations of features
- Complex pattern learning

Back to 768:
- Forces model to compress learned patterns
- Retains only most important information
```

**2. Why 3072? (4× Rule)**

```
Standard ratio: 4 × model_dimension
768 × 4 = 3072

Proven in practice:
- BERT: 768 → 3072
- GPT: 768 → 3072
- ViT: 768 → 3072
```

**3. ReLU Activation:**

```
ReLU(x) = max(0, x)

Properties:
- Simple: Efficient to compute
- Non-linear: Enables complex function learning
- Sparse: Typically ~50% of neurons are zero
- Well-studied: Proven effective in deep networks
```

**What Does FFN Learn?**

The FFN learns **non-linear transformations** of the attended features:

```
After attention, frame embedding might represent:
"Hand near face + eyebrows raised"

FFN transforms this to:
"This is probably a question sign"

Or:
"Left hand + right hand symmetric" → "Two-handed sign"
```

##### Sub-Component 5: Dropout

**Application Points:**

```python
class EncoderBlock(nn.Module):
    def forward(self, x):
        # Multi-head attention
        attn_output = self.attention(x)
        attn_output = self.dropout(attn_output, p=0.1)  ← Dropout 1

        # Residual + Norm
        x = x + attn_output
        x = self.layer_norm1(x)

        # Feed-forward
        ffn_output = self.ffn(x)
        ffn_output = self.dropout(ffn_output, p=0.1)   ← Dropout 2

        # Residual + Norm
        x = x + ffn_output
        x = self.layer_norm2(x)

        return x
```

**Dropout Rate:** p = 0.1 (10% of neurons randomly set to zero)

**How Dropout Works:**

**During Training:**

```
Original output: [0.5, 1.2, -0.3, 0.8, ...]

Apply dropout (p=0.1):
Random mask: [1, 0, 1, 1, ...]  ← 10% are zeros
Scaled output: [0.556, 0, -0.333, 0.889, ...]  ← Scale by 1/(1-p)
```

**During Inference:**

```
No dropout applied
Output used as-is
```

**Why Dropout?**

1. **Prevents Overfitting:** Model can't rely on specific neurons
2. **Ensemble Effect:** Acts like training multiple models and averaging
3. **Regularization:** Reduces model capacity without reducing parameters

#### 3.3.4 Stacking 6 Encoder Blocks

**Architecture:**

```
Input (position-aware embeddings): Z ∈ ℝ^(120 × 768)
         ↓
Encoder Block 1 → Z¹ ∈ ℝ^(120 × 768)
         ↓
Encoder Block 2 → Z² ∈ ℝ^(120 × 768)
         ↓
Encoder Block 3 → Z³ ∈ ℝ^(120 × 768)
         ↓
Encoder Block 4 → Z⁴ ∈ ℝ^(120 × 768)
         ↓
Encoder Block 5 → Z⁵ ∈ ℝ^(120 × 768)
         ↓
Encoder Block 6 → Z⁶ ∈ ℝ^(120 × 768)
         ↓
Final Encoded Sequence: H ∈ ℝ^(120 × 768)
```

**Why 6 Layers?**

**Hierarchical Feature Learning:**

```
Layer 1-2: Low-level patterns
- Individual keypoint movements
- Simple temporal transitions

Layer 3-4: Mid-level patterns
- Hand trajectories
- Facial expression sequences

Layer 5-6: High-level patterns
- Complete sign gestures
- Multi-modal integration (hand+face)
- Semantic meaning
```

**Empirical Evidence:**

- Original Transformer (Vaswani et al., 2017): 6 layers
- BERT-base: 12 layers (larger model)
- ViT-base: 12 layers
- Our model: 6 layers (balances capacity with dataset size)

**Why Not More Layers?**

With FSL-105's ~1,700 training samples:

- 6 layers: ~90M parameters → Good balance
- 12 layers: ~180M parameters → Risk of overfitting
- More layers don't always improve performance on smaller datasets

#### 3.3.5 CLS Token Pooling

After 6 encoder blocks, we have a sequence of 120 frame representations. But we need a **single vector** to represent the entire video.

**Options:**

**1. Mean Pooling:**

```python
pooled = torch.mean(encoded_sequence, dim=0)
# Average all 120 frames → (768,)
```

**2. Max Pooling:**

```python
pooled, _ = torch.max(encoded_sequence, dim=0)
# Take maximum across all frames → (768,)
```

**3. Last Frame:**

```python
pooled = encoded_sequence[-1]
# Use final frame → (768,)
```

**4. CLS Token (Our Approach):**

```python
# Add learnable [CLS] token at position 0
cls_token = nn.Parameter(torch.randn(1, 768))
sequence_with_cls = torch.cat([cls_token, encoded_sequence], dim=0)
# Shape: (121, 768)

# After encoding, extract CLS token representation
pooled = final_output[0]  # (768,)
```

**Why CLS Token?**

Inspired by BERT:

- CLS token "attends to" all frames during encoding
- Learns to aggregate video-level information
- More flexible than fixed pooling strategies

**Visualization:**

```
Frame representations:
[f₀, f₁, f₂, ..., f₁₁₉]

After attention in each layer:
CLS token sees all frames and learns:
"This video is sign X because frames 10-30 show hand movement Y,
 frames 40-60 show facial expression Z, and frames 80-100 show both"

CLS representation ← Learned video-level summary
```

#### 3.3.6 Dual Classification Heads

The pooled representation branches into **two parallel classification heads**:

##### Recognition Head (Gloss Prediction)

**Purpose:** Identify which of the 105 glosses is being signed

**Architecture:**

```python
recognition_head = nn.Sequential(
    nn.Linear(768, 105),
    nn.Softmax(dim=-1)
)
```

**Process:**

```
Input: pooled ∈ ℝ^768 (CLS token representation)
         ↓
Linear transformation:
logits = W_gloss · pooled + b_gloss
         ↓
W_gloss ∈ ℝ^(105 × 768)
b_gloss ∈ ℝ^105
         ↓
logits ∈ ℝ^105 (raw scores for each gloss)
         ↓
Softmax:
P(gloss_i) = exp(logit_i) / Σ_j exp(logit_j)
         ↓
probabilities ∈ ℝ^105
```

**Output Interpretation:**

```python
probabilities = [0.001, 0.003, 0.847, 0.012, ..., 0.005]
                  ↑       ↑       ↑
                gloss0  gloss1  gloss2 (highest)

predicted_gloss = argmax(probabilities) = 2
confidence = probabilities[2] = 0.847 (84.7%)
```

**Gloss Index Mapping:**

```
0 → "good_morning"
1 → "hello"
2 → "how_are_you"  ← Predicted
3 → "nice_to_meet_you"
...
104 → "wine"
```

##### Category Head (Semantic Classification)

**Purpose:** Classify sign into one of 10 semantic categories

**Architecture:**

```python
category_head = nn.Sequential(
    nn.Linear(768, 10),
    nn.Softmax(dim=-1)
)
```

**Process:**

```
Input: pooled ∈ ℝ^768 (same CLS token)
         ↓
Linear transformation:
logits = W_category · pooled + b_category
         ↓
W_category ∈ ℝ^(10 × 768)
b_category ∈ ℝ^10
         ↓
logits ∈ ℝ^10
         ↓
Softmax:
P(category_i) = exp(logit_i) / Σ_j exp(logit_j)
         ↓
probabilities ∈ ℝ^10
```

**Output Interpretation:**

```python
probabilities = [0.921, 0.015, 0.008, ..., 0.003]
                  ↑
               category0 (highest)

predicted_category = argmax(probabilities) = 0
confidence = probabilities[0] = 0.921 (92.1%)
```

**Category Index Mapping:**

```
0 → "Greetings"  ← Predicted
1 → "Survival"
2 → "Numbers"
3 → "Calendar"
4 → "Days"
5 → "Family"
6 → "Relationships"
7 → "Colors"
8 → "Food"
9 → "Drinks"
```

**Multi-Task Learning:**

Training with both heads simultaneously:

```python
# Combined loss
loss = loss_gloss + 0.5 * loss_category

where:
loss_gloss = CrossEntropyLoss(pred_gloss, true_gloss)
loss_category = CrossEntropyLoss(pred_category, true_category)
```

**Why 0.5 Weight on Category Loss?**

- Category task is easier (10 classes vs 105)
- Downweighting prevents category task from dominating
- Helps balance gradient contributions

**Benefits of Dual Heads:**

1. **Multi-Task Learning:** Model learns both specific (gloss) and general (category) features
2. **Regularization:** Category task provides additional supervision
3. **Semantic Understanding:** Model learns that similar glosses belong to same category
4. **Evaluation Flexibility:** Can assess both fine-grained and coarse-grained performance

### 3.4 Multi-Head Attention Mechanism (MHAM)

This is the **core innovation** of the Transformer architecture. MHAM enables the model to simultaneously attend to different aspects of the input sequence.

#### 3.4.1 Self-Attention Intuition

**Concept:** Each frame asks, "Which other frames are relevant to me?"

**Example:**

```
Video frames: [0, 1, 2, ..., 50, 51, 52, ..., 119]
                            ↑
                     Frame 51 analysis

Frame 51 asks:
"I'm in the middle of a sign. What happened before? What comes after?"

Attention mechanism computes:
- Frame 51 is highly relevant to frames 48-54 (local context)
- Frame 51 also relevant to frame 10 (sign beginning)
- Frame 51 also relevant to frame 90 (sign ending)
```

#### 3.4.2 Scaled Dot-Product Attention

**Three Matrices:**

1. **Query (Q):** "What am I looking for?"
2. **Key (K):** "What do I contain?"
3. **Value (V):** "What information should I pass forward?"

**Step-by-Step Process:**

**Step 1: Compute Attention Scores**

```
Scores = Q · K^T

For frame i and frame j:
score_ij = dot_product(query_i, key_j)
```

**Intuition:** High score means frame i finds frame j relevant.

**Step 2: Scale Scores**

```
Scaled_Scores = Scores / √d_k

where d_k = dimension of keys (96 per head in our model)
```

**Why Scale?**

- Dot products grow with dimension
- Without scaling: softmax saturates (gradients vanish)
- Scaling keeps values in moderate range

**Step 3: Apply Softmax**

```
Attention_Weights = softmax(Scaled_Scores, dim=-1)

For each frame i:
weights_i = [w_i0, w_i1, w_i2, ..., w_i119]

Properties:
- All weights ≥ 0
- Sum of weights = 1.0
- Represents probability distribution
```

**Step 4: Weighted Sum of Values**

```
Output_i = Σ_j (weight_ij × value_j)

Frame i's output = weighted combination of all frames' values
```

**Complete Formula:**

```
Attention(Q, K, V) = softmax((Q · K^T) / √d_k) · V
```

#### 3.4.3 Multi-Head Attention

**Why Multiple Heads?**

Single attention can only learn **one type of relationship**.

```
Example limitation (1 head):
Head learns: "Attend to nearby frames for smooth motion"

But we also need to learn:
- Hand-to-face spatial alignment
- Left-hand to right-hand coordination
- Facial expression timing
- Sign boundaries
```

**Solution:** Run attention in parallel with **different learned projections**

**Architecture:**

```
Input: X ∈ ℝ^(120 × 768)
         ↓
For each head h (h = 1, 2, ..., 8):
├─ Q_h = X · W^Q_h  (120 × 768) · (768 × 96) = (120 × 96)
├─ K_h = X · W^K_h  (120 × 768) · (768 × 96) = (120 × 96)
└─ V_h = X · W^V_h  (120 × 768) · (768 × 96) = (120 × 96)
         ↓
head_h = Attention(Q_h, K_h, V_h)  ∈ ℝ^(120 × 96)
         ↓
Concatenate all heads:
multi_head_output = [head_1 | head_2 | ... | head_8]
                  = ℝ^(120 × 768)  (8 × 96 = 768)
         ↓
Final projection:
output = multi_head_output · W^O
       = (120 × 768) · (768 × 768)
       = (120 × 768)
```

**Head Specialization Example:**

```
Head 1 might learn:
High attention between frames where:
- Right hand is near face (spatial proximity)

Head 2 might learn:
High attention between frames where:
- Both hands move symmetrically (bilateral coordination)

Head 3 might learn:
High attention to:
- Frames with eyebrow raises (non-manual markers)

Head 4 might learn:
High attention between:
- Sign start and end frames (temporal boundaries)

Heads 5-8: Other learned patterns specific to FSL
```

**Parameter Count:**

```
Per head:
- W^Q_h: 768 × 96 = 73,728 parameters
- W^K_h: 768 × 96 = 73,728 parameters
- W^V_h: 768 × 96 = 73,728 parameters
- Subtotal: 221,184 parameters

8 heads: 8 × 221,184 = 1,769,472 parameters

Final projection W^O: 768 × 768 = 589,824 parameters

Total MHAM parameters: 2,359,296 parameters per attention layer
```

#### 3.4.4 Occlusion Handling via Attention

**Scenario:** Hand passes in front of face (frames 45-55)

**Traditional CNN-RNN Approach:**

```
Frame 45-55: Facial keypoints missing → [0, 0, 0, ...]
InceptionV3 extracts features → Degraded/corrupted
GRU processes sequence → Error propagates
Result: Reduced accuracy
```

**Transformer MHAM Approach:**

**Attention Weights Adapt:**

```
Frame 50 (occluded face):

Without occlusion:
weights_50 = [0.02, 0.03, ..., 0.15 (self), ..., 0.03, 0.02]
                                  ↑
                       High weight on self

With occlusion (face keypoints missing):
weights_50 = [0.08, 0.12, ..., 0.02 (self), ..., 0.14, 0.11]
                                  ↑
                       Low weight on self!
              ↑                                    ↑
         Higher weights on visible frames before and after
```

**Mechanism:**

1. **Attention scores decrease** for frames with missing keypoints
   (Query from occluded frame produces lower similarity with keys)

2. **Model redistributes attention** to frames with complete information
   (Nearby frames with visible face receive higher weights)

3. **Global context allows recovery:**

```
Frame 50 output =
  0.08 × frame_40 +
  0.12 × frame_41 +
  ...
  0.02 × frame_50 +  ← Low weight on occluded self
  ...
  0.14 × frame_58 +
  0.11 × frame_59

→ Model "borrows" facial information from visible frames
```

**Result:** Recognition accuracy maintained despite temporary occlusion

**Why CNN-RNN Cannot Do This:**

```
CNN: Processes each frame independently
     No cross-frame communication
     Cannot "look at" other frames when one is corrupted

RNN: Processes sequentially, one frame at a time
     Hidden state has fixed capacity
     Cannot selectively weight distant frames
     Error from occluded frame propagates forward
```

### 3.5 Baseline Model: InceptionV3-GRU

To evaluate the Transformer's performance, we compare against a **strong CNN-RNN baseline**.

#### 3.5.1 Architecture Overview

```
Input: Video (256×256 RGB frames)
         ↓
┌────────────────────────────┐
│ InceptionV3                │
│ (Pretrained on ImageNet)   │
│                            │
│ Per-frame feature extraction │
│ Output: 2048-dim vector    │
└────────────────────────────┘
         ↓
Frame Sequence (30 × 2048)
         ↓
┌────────────────────────────┐
│ GRU Layer 1                │
│ Hidden size: 16            │
│ Dropout: 0.3               │
└────────────────────────────┘
         ↓
┌────────────────────────────┐
│ GRU Layer 2                │
│ Hidden size: 12            │
│ Dropout: 0.3               │
└────────────────────────────┘
         ↓
Final Hidden State (12-dim)
         ↓
┌────────────────────────────┐
│ Dense Layer: 12 → 105      │
│ Softmax Activation         │
└────────────────────────────┘
         ↓
Gloss Prediction
```

#### 3.5.2 InceptionV3 (Spatial Feature Extractor)

**Purpose:** Extract spatial features from each video frame

**Architecture:**

- **Layers:** 48 convolutional layers
- **Parameters:** ~23.8 million
- **Pretrained:** ImageNet (1.2M images, 1000 classes)
- **Input:** 256×256 RGB image
- **Output:** 2048-dimensional feature vector

**Inception Module:**

Key innovation: Parallel convolutions at multiple scales

```
Input
  │
  ├─── 1×1 conv ────────────────┐
  │                             │
  ├─── 1×1 conv → 3×3 conv ─────┤
  │                             │
  ├─── 1×1 conv → 5×5 conv ─────┤
  │                             │
  └─── 3×3 maxpool → 1×1 conv ──┤
                                │
                        Concatenate
                                │
                            Output
```

**Benefits:**

- Captures features at multiple scales simultaneously
- Efficient parameter usage
- Proven performance on vision tasks

**Frame Processing:**

```python
for frame in video_frames:  # 30 frames
    # Resize frame to 256×256
    frame_resized = resize(frame, (256, 256))

    # Extract features
    features = inceptionv3(frame_resized)  # (2048,)

    feature_sequence.append(features)

# Result: (30, 2048) matrix
```

#### 3.5.3 GRU (Temporal Modeler)

**Purpose:** Model temporal dependencies between frames

**GRU (Gated Recurrent Unit):**

Simpler alternative to LSTM with similar performance

**GRU Cell Equations:**

```
Update gate:
z_t = σ(W_z · [h_{t-1}, x_t] + b_z)

Reset gate:
r_t = σ(W_r · [h_{t-1}, x_t] + b_r)

Candidate hidden state:
h̃_t = tanh(W_h · [r_t ⊙ h_{t-1}, x_t] + b_h)

Final hidden state:
h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

where:
σ = sigmoid function
⊙ = element-wise multiplication
```

**Intuition:**

- **Update gate (z_t):** How much of previous state to keep?
- **Reset gate (r_t):** How much of previous state to forget?
- **Candidate state (h̃_t):** New information to add
- **Final state (h_t):** Mixture of old and new

**Two-Layer Configuration:**

**Layer 1:**

```
Input: (30, 2048)  ← InceptionV3 features
Hidden size: 16
Output: (30, 16)   ← Sequence of hidden states
Dropout: 0.3
```

**Layer 2:**

```
Input: (30, 16)    ← From GRU Layer 1
Hidden size: 12
Output: (12,)      ← Final hidden state only
Dropout: 0.3
```

**Sequential Processing:**

```python
# Initialize hidden state
h_0 = zeros(16)

# Process sequence
for t in range(30):
    h_t = GRU_layer1(x_t, h_{t-1})
    h_t = dropout(h_t, p=0.3)

# Pass to second GRU layer
final_state = GRU_layer2(h_sequence)  # (12,)
```

#### 3.5.4 Classification Layer

```python
classifier = nn.Sequential(
    nn.Linear(12, 105),
    nn.Softmax(dim=-1)
)

probabilities = classifier(final_state)
predicted_gloss = argmax(probabilities)
```

#### 3.5.5 Key Differences: Transformer vs IV3-GRU

| Aspect                  | MHAM Transformer                           | InceptionV3-GRU                          |
| ----------------------- | ------------------------------------------ | ---------------------------------------- |
| **Input**               | Keypoint coordinates (78×2)                | Raw RGB frames (256×256)                 |
| **Feature Source**      | Structured skeletal data                   | Pixel-level visual features              |
| **Sequence Processing** | Parallel (all frames simultaneously)       | Sequential (one frame at a time)         |
| **Context**             | Global (any frame can attend to any other) | Local (limited by hidden state capacity) |
| **Occlusion**           | Dynamic attention reweighting              | Fixed per-frame extraction               |
| **Parameters**          | ~90M                                       | ~24M (IV3) + 50K (GRU)                   |
| **Interpretability**    | Attention weights visualizable             | Hidden states are black-box              |
| **Training**            | More data/compute intensive                | More efficient                           |
| **Inference**           | GPU-parallelizable                         | Sequential (slower)                      |

---

## 4. Data Generation Procedure

This section describes the **experimental workflow** used to train models, run inference, and collect performance data.

### 4.1 Pre-Experimentation Phase

#### 4.1.1 Dataset Preparation

**Step 1: Load FSL-105 Dataset**

```python
# Dataset structure
dataset_root = '/path/to/FSL-105/'
├── videos/
│   ├── greeting_goodmorning_signer1_rep1.mp4
│   ├── greeting_goodmorning_signer1_rep2.mp4
│   └── ...
├── labels.csv
└── metadata.json
```

**Step 2: Create Train-Test Split**

```python
# Load all samples
total_samples = 2130

# Stratified split (80/20)
train_samples = 1704  # 80%
test_samples = 426    # 20%

# Ensure all 105 glosses represented in both sets
for gloss in range(105):
    gloss_samples = get_samples_for_gloss(gloss)  # ~20 samples
    train_split = gloss_samples[:16]              # 80%
    test_split = gloss_samples[16:]               # 20%
```

**Step 3: Create Data Loaders**

```python
train_dataset = FSL105Dataset(
    video_paths=train_samples,
    transform=preprocessing_pipeline
)

test_dataset = FSL105Dataset(
    video_paths=test_samples,
    transform=preprocessing_pipeline
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)
```

#### 4.1.2 Pipeline Compatibility Validation

**Test End-to-End Processing:**

```python
def validate_pipeline():
    # Load one sample video
    sample_video = load_video('test_sample.mp4')
    print(f"✓ Video loaded: {sample_video.shape}")

    # Apply preprocessing
    keypoints = preprocess_video(sample_video)
    print(f"✓ Keypoints extracted: {keypoints.shape}")
    # Expected: (120, 156)

    # Convert to tensor
    input_tensor = torch.tensor(keypoints).unsqueeze(0)
    print(f"✓ Tensor created: {input_tensor.shape}")
    # Expected: (1, 120, 156)

    # Forward pass through model
    gloss_out, category_out = model(input_tensor)
    print(f"✓ Forward pass successful")
    print(f"  Gloss output shape: {gloss_out.shape}")    # (1, 105)
    print(f"  Category output shape: {category_out.shape}")  # (1, 10)

    # Verify output ranges
    assert torch.all(gloss_out >= 0) and torch.all(gloss_out <= 1)
    assert torch.allclose(torch.sum(gloss_out), torch.tensor(1.0))
    print(f"✓ Outputs are valid probability distributions")

    return True

# Run validation
if validate_pipeline():
    print("✓ Pipeline validation successful - ready for training")
```

#### 4.1.3 Model Training

**Initialize Models:**

```python
# Transformer model
transformer_model = TransformerFSLR(
    input_dim=156,
    d_model=768,
    nhead=8,
    num_encoder_layers=6,
    dim_feedforward=3072,
    num_glosses=105,
    num_categories=10
)

# InceptionV3-GRU model
iv3gru_model = InceptionV3GRU(
    num_glosses=105,
    gru_hidden_sizes=[16, 12],
    dropout=0.3
)
```

**Training Configuration:**

```python
# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5
)

# Loss functions
criterion_gloss = nn.CrossEntropyLoss()
criterion_category = nn.CrossEntropyLoss()
```

**Training Loop:**

```python
max_epochs = 50
early_stopping_patience = 10
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(max_epochs):
    # Training phase
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        videos, gloss_labels, category_labels = batch

        # Forward pass
        gloss_pred, category_pred = model(videos)

        # Compute loss
        loss_gloss = criterion_gloss(gloss_pred, gloss_labels)
        loss_category = criterion_category(category_pred, category_labels)
        loss = loss_gloss + 0.5 * loss_category

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            videos, gloss_labels, category_labels = batch
            gloss_pred, category_pred = model(videos)

            loss = criterion_gloss(gloss_pred, gloss_labels) + \
                   0.5 * criterion_category(category_pred, category_labels)
            val_loss += loss.item()

    # Calculate averages
    train_loss_avg = train_loss / len(train_loader)
    val_loss_avg = val_loss / len(val_loader)

    # Logging
    print(f"Epoch {epoch+1}/{max_epochs}")
    print(f"  Train Loss: {train_loss_avg:.4f}")
    print(f"  Val Loss: {val_loss_avg:.4f}")

    # Learning rate scheduling
    scheduler.step(val_loss_avg)

    # Save best model
    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        torch.save(model.state_dict(), 'best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

print("Training complete!")
```

### 4.2 Experimentation Phase

#### 4.2.1 Occlusion Detection

**Automatic Partitioning of Test Set:**

```python
def detect_occlusion(keypoints_sequence):
    """
    Detect if video contains occlusion of facial landmarks

    Args:
        keypoints_sequence: (T, 78, 2) array of keypoints

    Returns:
        bool: True if occlusion detected
    """
    # Facial landmark indices (11 face keypoints out of 78 total)
    face_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    occlusion_frames = []

    for t in range(len(keypoints_sequence)):
        # Extract facial keypoints for this frame
        face_keypoints = keypoints_sequence[t, face_indices, :]

        # Check if any are missing (all zeros or NaN)
        if np.any(np.isnan(face_keypoints)) or \
           np.any(np.all(face_keypoints == 0, axis=1)):
            occlusion_frames.append(t)

    # Apply 5-frame majority filter
    # (Reduces false positives from momentary tracking failures)
    filtered_frames = apply_majority_filter(occlusion_frames, window=5)

    # Count consecutive occluded frames
    consecutive_sequences = find_consecutive(filtered_frames)

    # Occlusion if any sequence ≥ 3 frames
    return any(len(seq) >= 3 for seq in consecutive_sequences)

# Partition test set
test_clean = []
test_occluded = []

for sample in test_dataset:
    video_path, label, category = sample
    keypoints = extract_keypoints(video_path)

    if detect_occlusion(keypoints):
        test_occluded.append(sample)
    else:
        test_clean.append(sample)

print(f"Test set partitioned:")
print(f"  Clean samples: {len(test_clean)}")
print(f"  Occluded samples: {len(test_occluded)}")
```

#### 4.2.2 Inference and Prediction Logging

**Run Inference on Both Models:**

```python
def evaluate_model(model, test_loader, condition_name):
    """
    Run inference and log all predictions

    Args:
        model: Trained model
        test_loader: DataLoader for test set
        condition_name: 'clean' or 'occluded'

    Returns:
        DataFrame with per-sample results
    """
    model.eval()
    results = []

    with torch.no_grad():
        for batch in test_loader:
            videos, true_glosses, true_categories, filenames = batch

            # Forward pass
            gloss_logits, category_logits = model(videos)

            # Get probabilities
            gloss_probs = F.softmax(gloss_logits, dim=-1)
            category_probs = F.softmax(category_logits, dim=-1)

            # Get predictions
            pred_glosses = torch.argmax(gloss_probs, dim=-1)
            pred_categories = torch.argmax(category_probs, dim=-1)

            # Get confidence scores
            gloss_confidences = gloss_probs.gather(1, pred_glosses.unsqueeze(1))
            category_confidences = category_probs.gather(1, pred_categories.unsqueeze(1))

            # Log each sample
            for i in range(len(filenames)):
                result = {
                    'filename': filenames[i],
                    'condition': condition_name,
                    'true_gloss': true_glosses[i].item(),
                    'pred_gloss': pred_glosses[i].item(),
                    'gloss_confidence': gloss_confidences[i].item(),
                    'gloss_correct': (pred_glosses[i] == true_glosses[i]).item(),
                    'true_category': true_categories[i].item(),
                    'pred_category': pred_categories[i].item(),
                    'category_confidence': category_confidences[i].item(),
                    'category_correct': (pred_categories[i] == true_categories[i]).item()
                }
                results.append(result)

    return pd.DataFrame(results)

# Evaluate Transformer model
transformer_clean = evaluate_model(transformer_model, test_clean_loader, 'clean')
transformer_occluded = evaluate_model(transformer_model, test_occluded_loader, 'occluded')

# Evaluate IV3-GRU model
iv3gru_clean = evaluate_model(iv3gru_model, test_clean_loader, 'clean')
iv3gru_occluded = evaluate_model(iv3gru_model, test_occluded_loader, 'occluded')

# Save results
transformer_clean.to_csv('transformer_clean_results.csv', index=False)
transformer_occluded.to_csv('transformer_occluded_results.csv', index=False)
iv3gru_clean.to_csv('iv3gru_clean_results.csv', index=False)
iv3gru_occluded.to_csv('iv3gru_occluded_results.csv', index=False)
```

#### 4.2.3 Compute Per-Sample Confusion Matrix Components

```python
def compute_confusion_components(true_labels, pred_labels, num_classes):
    """
    Compute TP, FP, TN, FN for each class

    Returns:
        tp, fp, tn, fn: arrays of shape (num_classes,)
    """
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    tn = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for c in range(num_classes):
        # True Positives: Predicted c AND actually c
        tp[c] = ((pred_labels == c) & (true_labels == c)).sum()

        # False Positives: Predicted c BUT actually NOT c
        fp[c] = ((pred_labels == c) & (true_labels != c)).sum()

        # False Negatives: Predicted NOT c BUT actually c
        fn[c] = ((pred_labels != c) & (true_labels == c)).sum()

        # True Negatives: Predicted NOT c AND actually NOT c
        tn[c] = ((pred_labels != c) & (true_labels != c)).sum()

    return tp, fp, tn, fn

# Compute for all conditions
tp_t_clean, fp_t_clean, tn_t_clean, fn_t_clean = compute_confusion_components(
    transformer_clean['true_gloss'].values,
    transformer_clean['pred_gloss'].values,
    num_classes=105
)
```

### 4.3 Post-Experimentation Phase

#### 4.3.1 Metric Computation

**Per-Class Metrics:**

```python
def compute_metrics(tp, fp, tn, fn):
    """
    Compute precision, recall, F1 for each class
    """
    # Avoid division by zero
    epsilon = 1e-10

    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp + epsilon)

    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn + epsilon)

    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)

    return precision, recall, f1

# Compute per-class metrics
prec_per_class, rec_per_class, f1_per_class = compute_metrics(
    tp_t_clean, fp_t_clean, tn_t_clean, fn_t_clean
)
```

**Macro-Averaged Metrics:**

```python
# Equal weight to each class
macro_precision = np.mean(prec_per_class)
macro_recall = np.mean(rec_per_class)
macro_f1 = np.mean(f1_per_class)

print(f"Transformer (Clean):")
print(f"  Macro Precision: {macro_precision:.4f}")
print(f"  Macro Recall: {macro_recall:.4f}")
print(f"  Macro F1: {macro_f1:.4f}")
```

#### 4.3.2 Statistical Testing

**Step 1: Check Normality**

```python
from scipy.stats import shapiro

# Compute paired differences (per-class)
f1_differences = f1_transformer_clean - f1_iv3gru_clean

# Shapiro-Wilk test
statistic, p_value = shapiro(f1_differences)

print(f"Shapiro-Wilk Test:")
print(f"  Statistic: {statistic:.4f}")
print(f"  p-value: {p_value:.4f}")

if p_value > 0.05:
    print("  → Data is normally distributed (use paired t-test)")
    use_parametric = True
else:
    print("  → Data is NOT normally distributed (use Wilcoxon test)")
    use_parametric = False
```

**Step 2: Perform Appropriate Test**

```python
if use_parametric:
    from scipy.stats import ttest_rel
    t_stat, p_val = ttest_rel(f1_transformer_clean, f1_iv3gru_clean)
    print(f"Paired t-test:")
    print(f"  t-statistic: {t_stat:.4f}")
else:
    from scipy.stats import wilcoxon
    w_stat, p_val = wilcoxon(f1_transformer_clean, f1_iv3gru_clean)
    print(f"Wilcoxon signed-rank test:")
    print(f"  W-statistic: {w_stat:.4f}")

print(f"  p-value: {p_val:.4f}")

if p_val < 0.05:
    print("  → Reject H0: Significant difference")
else:
    print("  → Fail to reject H0: No significant difference")
```

**Step 3: Apply Holm-Bonferroni Correction**

```python
def holm_bonferroni(p_values, alpha=0.05):
    """
    Apply Holm-Bonferroni correction for multiple comparisons
    """
    m = len(p_values)

    # Sort p-values and track original indices
    sorted_indices = np.argsort(p_values)
    sorted_pvalues = p_values[sorted_indices]

    # Adjust p-values
    adjusted = np.zeros(m)
    for i, p in enumerate(sorted_pvalues):
        adjusted[sorted_indices[i]] = min(p * (m - i), 1.0)

    # Decisions
    reject = adjusted < alpha

    return adjusted, reject

# Apply to multiple tests
p_values = np.array([
    p_precision,  # Test 1: Precision difference
    p_recall,     # Test 2: Recall difference
    p_f1          # Test 3: F1 difference
])

adjusted_p, decisions = holm_bonferroni(p_values)

for i, (orig_p, adj_p, decision) in enumerate(zip(p_values, adjusted_p, decisions)):
    print(f"Test {i+1}:")
    print(f"  Original p: {orig_p:.4f}")
    print(f"  Adjusted p: {adj_p:.4f}")
    print(f"  Decision: {'Reject H0' if decision else 'Fail to reject H0'}")
```

---

## 5. Data Analysis

### 5.1 Performance Metric Formulas

**Precision:**

```
Precision = TP / (TP + FP)

Interpretation: Of all predicted positives, how many were correct?
```

**Recall:**

```
Recall = TP / (TP + FN)

Interpretation: Of all actual positives, how many did we find?
```

**F1-Score:**

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Interpretation: Harmonic mean balancing precision and recall
```

**Macro-Averaging:**

```
Macro_Precision = (1/N) × Σ Precision_i

where N = number of classes (105 for gloss, 10 for category)

Gives equal weight to each class regardless of frequency
```

### 5.2 Statistical Test Selection

**Decision Tree:**

```
Are paired differences normally distributed?
├─ Yes → Use paired t-test
└─ No → Use Wilcoxon signed-rank test
```

**Paired t-Test Formula:**

```
t = (mean_diff) / (std_diff / √n)

where:
mean_diff = mean of paired differences
std_diff = standard deviation of differences
n = sample size
```

**Wilcoxon Test:**

```
Rank absolute differences
Compute sum of positive ranks (W+)
Compute sum of negative ranks (W-)
Test statistic: W = min(W+, W-)
```

### 5.3 Multiple Comparison Correction

**Holm-Bonferroni Method:**

```
For m tests with p-values p₁ ≤ p₂ ≤ ... ≤ pₘ:

Adjusted p-value:
p'ᵢ = pᵢ × (m - i + 1)

Reject H0 if p'ᵢ < α
```

**Example:**

```
3 tests: Precision, Recall, F1
α = 0.05

p-values: [0.023, 0.041, 0.156]
Sorted:   [0.023, 0.041, 0.156]

Adjusted:
p'₁ = 0.023 × (3 - 0) = 0.069  → Fail to reject (0.069 > 0.05)
p'₂ = 0.041 × (3 - 1) = 0.082  → Fail to reject
p'₃ = 0.156 × (3 - 2) = 0.156  → Fail to reject
```

### 5.4 Mapping Research Questions to Analysis

| Research Question              | Data Source               | Analysis Method                      |
| ------------------------------ | ------------------------- | ------------------------------------ |
| RQ1: Recognition (Clean)       | Test clean predictions    | Macro-averaged metrics + paired test |
| RQ2: Recognition (Occluded)    | Test occluded predictions | Macro-averaged metrics + paired test |
| RQ3: Classification (Clean)    | Test clean categories     | Macro-averaged metrics + paired test |
| RQ4: Classification (Occluded) | Test occluded categories  | Macro-averaged metrics + paired test |

---

This comprehensive documentation provides a complete, step-by-step explanation of the system methodology and architecture for the Multi-Head Attention Transformer for Filipino Sign Language Recognition.
