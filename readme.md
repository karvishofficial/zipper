# Karvish — Autonomous On-Device Mobile Action Intelligence

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Target Platform: Android](https://img.shields.io/badge/Platform-Android%2014%2B%20%7C%20OneUI%206%2B-green.svg)](https://developer.android.com)
[![Hardware Verified: Samsung Galaxy A26](https://img.shields.io/badge/Device-Samsung%20Galaxy%20A26-orange.svg)]()
[![Inference Runtime: ONNX Runtime Mobile FP16](https://img.shields.io/badge/Runtime-ONNX%20Mobile%20FP16%20%28%3C50ms%29-blueviolet.svg)]()
[![Constitution: AGENTS.md Certified](https://img.shields.io/badge/Contract-AGENTS.md%20Strict-red.svg)](AGENTS.md)
[![Frozen Control: V54-LOCKED](https://img.shields.io/badge/Frozen%20Control-V54--LOCKED--CONTROL-brightgreen.svg)](tools/training/karvish_mobile_action/v54/V54_LOCKED_CONTROL.json)

**Karvish** is an experimental on-device mobile action intelligence framework. It enables compact neural action policies (10M–27M parameters) to observe live mobile user interfaces, interpret natural-language goals, ground spatial candidates, select executable actions, maintain multi-turn recurrent task memory, and physically verify task outcomes directly on edge hardware (smartphones) with **zero cloud dependencies** during autonomous execution.

---

## Table of Contents
- [1. Core Research Philosophy & Constitutional Contract](#1-core-research-philosophy--constitutional-contract)
  - [The Motors and Brakes Rule](#the-motors-and-brakes-rule)
  - [Failure-to-Train Protocol](#failure-to-train-protocol)
- [2. Model Evolution & Frozen Baselines](#2-model-evolution--frozen-baselines)
  - [Lineage Overview](#lineage-overview)
  - [2.1 V51.7-GOLD (Historical Control)](#21-v517-gold-historical-control)
  - [2.2 V52-LOCKED-GOLD (Authoritative Frozen Baseline)](#22-v52-locked-gold-authoritative-frozen-baseline)
  - [2.3 V54-LOCKED-CONTROL (Universal Search & Contact Baseline)](#23-v54-locked-control-universal-search--contact-baseline)
  - [2.4 V55 Next-Gen (Continual Learning & Multimodal Verification)](#24-v55-next-gen-continual-learning--multimodal-verification)
- [3. Neural Policy Architecture](#3-neural-policy-architecture)
  - [Full Architecture Dataflow](#full-architecture-dataflow)
  - [3.1 Text Backbone & Continuous 2D Spatial Geometry](#31-text-backbone--continuous-2d-spatial-geometry)
  - [3.2 Foundational 0/1/2 Role Embedding Contract](#32-foundational-012-role-embedding-contract)
  - [3.3 4-Way Interaction Candidate Ranking Tensor](#33-4-way-interaction-candidate-ranking-tensor)
  - [3.4 Recurrent Task State Engine & Bounded Unit Normalization](#34-recurrent-task-state-engine--bounded-unit-normalization)
  - [3.5 Multimodal Observation Delta Verifier](#35-multimodal-observation-delta-verifier)
  - [3.6 Action-Conditioned Dual Pointer Span Extraction Head](#36-action-conditioned-dual-pointer-span-extraction-head)
- [4. Android Runtime Architecture](#4-android-runtime-architecture)
  - [Runtime Dataflow Pipeline](#runtime-dataflow-pipeline)
  - [4.1 Accessibility Distillation (`UiStateDistiller`)](#41-accessibility-distillation-uistatedistiller)
  - [4.2 Generic Accessibility-Tree Normalization](#42-generic-accessibility-tree-normalization)
  - [4.3 On-Device Neural Execution (`OnDeviceRankerEngine`)](#43-on-device-neural-execution-ondevicerankerengine)
  - [4.4 Closed-Loop Orchestration (`AgentRuntime`)](#44-closed-loop-orchestration-agentruntime)
  - [4.5 Evidence-Gated Completion (Outcome Verification)](#45-evidence-gated-completion-outcome-verification)
  - [4.6 Real-Time Visual Feedback HUD (`KarvishControlOverlay`)](#46-real-time-visual-feedback-hud-karvishcontroloverlay)
- [5. Physical Evaluation & Anchor Benchmark Matrix](#5-physical-evaluation--anchor-benchmark-matrix)
  - [Target Hardware Setup](#target-hardware-setup)
  - [Physical Anchor Benchmark Tasks](#physical-anchor-benchmark-tasks)
  - [Empirical Forensics & Logcat Diagnosis](#empirical-forensics--logcat-diagnosis)
- [6. Senior Continual Learning & Distillation Roadmap](#6-senior-continual-learning--distillation-roadmap)
  - [The 10-Step Continual Learning Protocol](#the-10-step-continual-learning-protocol)
  - [Scorecard Certification & Promotion Gate](#scorecard-certification--promotion-gate)
- [7. Repository Structure](#7-repository-structure)
- [8. Developer Quick Start, Build & Deployment Guide](#8-developer-quick-start-build--deployment-guide)
  - [Prerequisites](#prerequisites)
  - [Build and Install Debug APK](#build-and-install-debug-apk)
  - [Grant Accessibility Permissions](#grant-accessibility-permissions)
  - [Real-Time Logcat Monitoring](#real-time-logcat-monitoring)
  - [Python Training & ONNX FP16 Export](#python-training--onnx-fp16-export)
- [9. License](#9-license)

---

## 1. Core Research Philosophy & Constitutional Contract

Karvish operates under an immutable project constitution defined in [`AGENTS.md`](AGENTS.md). 

### The Motors and Brakes Rule:
* **Legitimate Platform Plumbing (Motors & Brakes):** Deterministic runtime code is strictly reserved for platform mechanics: Android Accessibility service dispatch, coordinate translation, safe viewport clamping, IME synchronization, process lifecycle, transport, and safety checks.
* **Task Intelligence in the Weights:** Task decisions (which button advances the goal, what query to extract, when a task is completed, how to recover from dead ends) belong **exclusively in the neural model**.
* **Zero Heuristic Spoonfeeding:** Hardcoding task-specific keywords, app package heuristics, fixed coordinates, or oracle progress injection is strictly forbidden. 

```text
    USER GOAL
       │
       ▼
    OBSERVE ENVIRONMENT (Accessibility Tree / Continuous Geometry)
       │
       ▼
    NEURAL MODEL INFERENCE (On-Device ONNX FP16, <50ms)
       │
       ▼
    EXECUTE NEXT ACTION (Tap, TypeText, Scroll, Submit, Back)
       │
       ▼
    OBSERVE RESULTING STATE TRANSITION (Observation Delta)
       │
       ▼
    UPDATE RECURRENT TASK STATE (h_prev → h_next)
       │
       ▼
    OUTCOME-BASED EVIDENCE VERIFICATION
       │
       ▼
    COMPLETE OR ADAPT / RECOVER
```

### Failure-to-Train Protocol:
When a model fails on a physical device, the failure is **never** resolved by adding a runtime heuristic (e.g. `if (package == WhatsApp) boost search`). Instead, the mandatory research protocol is:

$$\text{Reproduce Failure} \longrightarrow \text{Diagnose Root Cause} \longrightarrow \text{Collect Hard Negatives} \longrightarrow \text{Retrain / Distill} \longrightarrow \text{Physical Device Revalidation}$$

---

## 2. Model Evolution & Frozen Baselines

To safeguard against catastrophic forgetting and maintain strict scientific reproducibility, all Karvish research is anchored against immutable frozen controls.

### Lineage Overview:

```text
┌────────────────────────────────────────┐
│        V51.7-GOLD (10.88M)             │  Historical physical baseline
│ Checkpoint: v51_phase6_7/v51_7_gold.pt │  First de-spoonfeeding control group
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│      V52-LOCKED-GOLD (26.78M)          │  Authoritative frozen control (Locked Sep 30, 2026)
│ SHA256: fa38aa02c290f8839ee4c1b1621... │  12/12 Physical anchor pass rate on Samsung A26
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│     V54-LOCKED-CONTROL (26.78M)        │  Universal Search & Contact Baseline
│ SHA256: a6decb1129df2bc9f622d4a13b5... │  Container Attention, Entity Gating, Price Grounding
│ ONNX: karvish_v54_fp16.onnx (50.88 MB) │  Locked Control for V55 Teacher Distillation
└───────────────────┬────────────────────┘
                    │
                    ▼
┌────────────────────────────────────────┐
│             V55 NEXT-GEN               │  Goal-Conditioned Multimodal Attention,
│      (Under Continual Learning)        │  Recurrent State Memory, Delta Verification,
│ ONNX: karvish_v55_fp16.onnx (50.88 MB) │  Dual Pointer Span Extraction Head
└────────────────────────────────────────┘
```

### Baseline Specifications:

| Model Baseline | Parameters | Artifact / Checkpoint | SHA-256 Checksum | Anchor Qualification | Key Architectural Features |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **V51.7-GOLD** | 10.88M | `tools/training/karvish_mobile_action/v51_phase6_7/checkpoints/v51_7_gold.pt` | *(Repository Record)* | 7/8 Tasks | Historical control baseline for de-spoonfeeding ablations. |
| **V52-LOCKED-GOLD** | 26.78M | `tools/training/karvish_mobile_action/v52_locked_gold/v52_locked_gold.pt` | `fa38aa02c290f883...` | **12/12 Anchor Tasks** | Continuous 2D geometry, InfoNCE contrastive loss, 4-way interaction. |
| **V54-LOCKED-CONTROL** | 26.78M | `tools/training/karvish_mobile_action/v54/checkpoints/v54_universal_search_best.pt` | `a6decb1129df2bc9...` | **60%+ Messaging Pass** | Container attention, numerical price grounding, search affordance grounding. |
| **V55 NEXT-GEN** | 26.85M | `tools/training/karvish_mobile_action/v55/checkpoints/v55_best.pt` | *(Active Branch)* | Active Continual Learning | Recurrent memory chaining, observation delta verifier, pointer span head. |

---

## 3. Neural Policy Architecture

### Full Architecture Dataflow:

```text
                  USER GOAL TOKENS [B, L]
                             │
                             ▼
                 MiniLM-L6 Text Encoder (384-d)
                             │
             ┌───────────────┴───────────────┐
             ▼                               ▼
       Goal Embedding u                Span Token Features
             │                               │
             │                 ┌─────────────┴─────────────┐
             │                 ▼                           ▼
             │           Span Start Head             Span End Head
             │                 │                           │
             │                 └─────────────┬─────────────┘
             │                               ▼
             │                      Extracted Argument Text
             │
             │   CANDIDATE FEATURES [B, N, D]
             │   ├── Candidate Labels (MiniLM Text Encoder)
             │   ├── Continuous 2D Geometry (10-dim MLP → 64-d)
             │   └── Foundational Role Embedding (0/1/2 → 32-d)
             │                 │
             │                 ▼
             │      Hierarchical Candidate Fusion
             │                 │
             ▼                 ▼
      4-WAY INTERACTION TENSOR: [u, v, diff, prod] ∈ R^[B, N, 4×384]
             │
             ├───────────────────────────────────────┐
             ▼                                       ▼
     Candidate Ranking Head                 Action Type Head
     Logits for each candidate             (Tap, TypeText, Scroll...)
             │
             ▼
     RECURRENT TASK STATE ENGINE
     h_next = GRU(h_prev, [u, v_selected, action_emb])
     Stabilized via Bounded Unit Normalization: (h / ||h||) * 8.0
             │
             ▼
     MULTIMODAL OBSERVATION DELTA VERIFIER
     Cross-Attention(obs_before, obs_after) → P(stall)
```

### 3.1 Text Backbone & Continuous 2D Spatial Geometry
* **Semantic Backbone:** Lightweight 6-layer MiniLM transformer mapping natural language token sequences into 384-dimensional dense semantic vectors $\mathbf{u} \in \mathbb{R}^{384}$.
* **Continuous 2D Spatial Geometry Tensor:** Every candidate node is parameterized by a continuous 10-dimensional spatial geometry vector:
  $$\mathbf{g} = \left[ x_{\text{left}}, y_{\text{top}}, x_{\text{right}}, y_{\text{bottom}}, x_{\text{center}}, y_{\text{center}}, \text{width}, \text{height}, \text{area}, \text{aspect\_ratio} \right]$$
  Normalized continuously to $[0.0, 1.0]$ relative to the physical screen resolution and projected through a 2-layer MLP with GELU activations into a 64-dimensional spatial embedding.

### 3.2 Foundational 0/1/2 Role Embedding Contract
The foundational pretrained action policy relies on a strictly partitioned 3-class role contract:
* **Role `2L` (Input / Editable):** Text fields, search bars, edit boxes, query inputs.
* **Role `1L` (Button / Clickable):** Action buttons, list rows, cards, icons, tabs, toggles.
* **Role `0L` (Display / Other):** Non-actionable background text, decorative images, metric counters.

> [!IMPORTANT]
> The runtime strictly enforces the `0/1/2` role contract. Assigning unlearned role IDs (e.g. `3..31`) routes inputs into untrained embedding rows, destroying attention sharpness and producing random candidate selection.

### 3.3 4-Way Interaction Candidate Ranking Tensor
Candidate ranking combines goal intent $\mathbf{u}$ and fused candidate representation $\mathbf{v}_i$ through four complementary interaction modes:
$$\mathbf{I}_i = \left[ \mathbf{u}, \mathbf{v}_i, \mathbf{u} - \mathbf{v}_i, \mathbf{u} \odot \mathbf{v}_i \right] \in \mathbb{R}^{4 \times 384}$$
This representation simultaneously measures:
1. Pure goal context ($\mathbf{u}$)
2. Pure candidate semantic & spatial identity ($\mathbf{v}_i$)
3. Direct directional difference vector ($\mathbf{u} - \mathbf{v}_i$)
4. Element-wise multiplicative feature alignment ($\mathbf{u} \odot \mathbf{v}_i$)

The interaction tensor is processed through LayerNorm, GELU, and a linear projection to output real-valued ranking logits.

### 3.4 Recurrent Task State Engine & Bounded Unit Normalization
To track multi-turn task progress across sequential screens without compounding drift or half-precision overflow, the recurrent hidden state $\mathbf{h}_{\text{task}} \in \mathbb{R}^{128}$ is stabilized using **Bounded Unit Normalization**:
$$\hat{\mathbf{h}}_{\text{next}} = \frac{\mathbf{h}_{\text{next}}}{\max(\|\mathbf{h}_{\text{next}}\|_2, 1e-6)} \times 8.0, \quad \text{clamped to } [-50.0, 50.0]$$
This guarantees the recurrent norm remains stably anchored at $\approx 8.0$ across 40+ consecutive turns, completely eliminating exploding activations, underflow, and NaNs.

### 3.5 Multimodal Observation Delta Verifier
A dedicated cross-attention head that compares the observation representations immediately before and after action execution:
$$\Delta_{\text{obs}} = \left[ \mathbf{u}_{\text{after}} - \mathbf{u}_{\text{before}}, \mathbf{g}_{\text{after}} - \mathbf{g}_{\text{before}} \right]$$
Predicts $P(\text{stall}) \in [0.0, 1.0]$: the probability that the previous action produced a NO-OP, failed transition, or modal trap, allowing the runtime to initiate active recovery without hardcoded timeouts.

### 3.6 Action-Conditioned Dual Pointer Span Extraction Head
Rather than using fragile regular expressions, Karvish employs dual pointer networks over goal tokens:
$$P(\text{start} = j) = \text{Softmax}(\mathbf{W}_s \mathbf{h}_{\text{token}, j}), \quad P(\text{end} = k) = \text{Softmax}(\mathbf{W}_e \mathbf{h}_{\text{token}, k})$$
Enables direct, end-to-end extraction of target entities, search queries, and message payloads (e.g. `"send hi to Manoj on WhatsApp"` $\rightarrow$ argument `"hi"`, recipient `"Manoj"`).

---

## 4. Android Runtime Architecture

The production Android client (`app/`) is built in Kotlin and executes natively on device via Android Accessibility APIs.

### Runtime Dataflow Pipeline:

```text
┌─────────────────────────────────┐
│   AccessibilityService Event    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│       UiStateDistiller          │  Extracts visible AccessibilityNodeInfo hierarchy
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     ToolSchemaGenerator         │  Applies Generic Normalization:
│                                 │  • Parent-Child Action Binding
│                                 │  • Multi-Signal Informative Metric Demotion
│                                 │  • Foundational 0/1/2 Role Assignment
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     OnDeviceRankerEngine        │  Runs ONNX Runtime Mobile FP16 (<50ms)
│                                 │  • Tokenizes goal & candidate text
│                                 │  • Computes 4-Way Interaction & Ranking
│                                 │  • Updates Recurrent State with Unit Normalization
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│         AgentRuntime            │  Closed-Loop Orchestration:
│                                 │  • Checks Evidence-Gated Completion
│                                 │  • Rejects premature CompleteGoal
│                                 │  • Dispatches Action (Tap / TypeText / Scroll)
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│     KarvishControlOverlay       │  Renders real-time HUD with target bounding box,
│                                 │  confidence score, latency, and recovery state
└─────────────────────────────────┘
```

### 4.1 Accessibility Distillation (`UiStateDistiller`)
Captures the live `AccessibilityNodeInfo` hierarchy and constructs an optimized snapshot containing:
* Bounding rectangles normalized to device display metrics
* Text and content descriptions
* State flags: `isClickable`, `isEditable`, `isFocused`, `isSelected`, `isScrollable`
* View hierarchy depth and parent/child structural relationships

### 4.2 Generic Accessibility-Tree Normalization
In strict accordance with AGENTS.md Section 14 (*"No App-Specific Workflow Logic"*), tree normalization uses universal structural principles:
1. **Parent-Child Action Binding:** When an actionable container (`isClickable=true`, e.g. a WhatsApp contact row or social feed item) encloses child visual elements (`ImageView`, avatar thumbnail), the parent container absorbs actionability. The child thumbnail is demoted to display-only (`role="display"`), preventing unwanted profile picture preview popups.
2. **Multi-Signal Informational Metric Demotion:** Pure metric counters (e.g. `"45.7K"`, `"266 posts"`) are detected via multi-signal layout reasoning (non-editable `TextView`, no action verbs, enclosed in card/header) and assigned role `0L`, preventing the model from tapping view counts instead of action buttons.

### 4.3 On-Device Neural Execution (`OnDeviceRankerEngine`)
* Loads optimized FP16 ONNX models via `onnxruntime-android`.
* Tokenizes text using an embedded SentencePiece / WordPiece tokenizer.
* Feeds candidate labels, 10-dim geometry vectors, and `0/1/2` role IDs.
* Maintains recurrent task state $\mathbf{h}_{\text{task}}$ across execution turns.
* Average end-to-end inference latency: **32ms – 48ms** on Samsung Galaxy A26.

### 4.4 Closed-Loop Orchestration (`AgentRuntime`)
* Manages the complete autonomous action cycle (Observe $\rightarrow$ Ground $\rightarrow$ Act $\rightarrow$ Verify).
* Features automated loop-stall detection: if the screen transition does not advance after execution, recovery logic is triggered.
* Enforces target freshness to protect against stale node touches during UI animations.

### 4.5 Evidence-Gated Completion (Outcome Verification)
Tasks are **never** declared complete based on keyword heuristics or superficial entity discovery. Completion requires observable physical outcome evidence:
$$\text{Locate Recipient} \longrightarrow \text{Enter Message Text} \longrightarrow \text{Dispatch Send} \longrightarrow \text{Verify Message Sent on Screen} \longrightarrow \text{COMPLETE}$$
Premature completion attempts emitted before the intended outcome has taken place are rejected by the runtime, prompting the agent to complete the remaining steps.

### 4.6 Real-Time Visual Feedback HUD (`KarvishControlOverlay`)
A non-intrusive floating HUD rendered via Android's `WindowManager`:
* Real-time bounding box highlighting the currently targeted element
* Decision latency (ms), confidence percentage, and active model tag
* Visual indicators of NO-OP stall detection and recovery reflections

---

## 5. Physical Evaluation & Anchor Benchmark Matrix

Offline benchmarks are diagnostic; **physical device execution is authoritative**. All releases are qualified on a real Samsung Galaxy A26:

### Target Hardware Setup:
* **Device:** Samsung Galaxy A26 (`RZCY51MQP9K`)
* **OS:** Android 14 / OneUI 6.1
* **Execution Environment:** Full on-device standalone (Zero cloud, local ONNX Mobile FP16 runtime)

### Physical Anchor Benchmark Tasks:

| Task Family | Natural Language Goal | Target Physical Action Flow | Acceptance Verification Standard |
| :--- | :--- | :--- | :--- |
| **WhatsApp Contact** | *"send hi to Manoj on WhatsApp"* | Open WhatsApp $\rightarrow$ Search / Select Manoj $\rightarrow$ Type "hi" $\rightarrow$ Tap Send | Message bubble confirmed on screen; zero avatar popups. |
| **Instagram Search** | *"search photography on Instagram"* | Open Instagram $\rightarrow$ Tap Search $\rightarrow$ Enter "photography" | Search input focused; feed distractors ignored. |
| **YouTube Media** | *"search lo-fi hip hop on YouTube"* | Open YouTube $\rightarrow$ Tap Search icon $\rightarrow$ Type query $\rightarrow$ Submit | Search bar targeted over video recommendations. |
| **Zepto E-Commerce** | *"buy cheapest fresh tomatoes on Zepto"* | Open Zepto $\rightarrow$ Search "fresh tomatoes" $\rightarrow$ Add lowest ₹ item | Scalar price grounding selects lowest cost item. |
| **System Settings** | *"open settings and check battery"* | Open Settings $\rightarrow$ Scroll/Select "Battery" | Exact deep-menu navigation without mis-clicks. |
| **Calculator** | *"calculate 45 * 12"* | Open Calculator $\rightarrow$ Tap `4`, `5`, `×`, `1`, `2`, `=` | Precise keypad grounding; correct calculation result. |

### Empirical Forensics & Logcat Diagnosis:
Recent on-device forensic analysis revealed:
1. **Recurrent Stability:** Bounded unit normalization maintains stable recurrent norm ($8.000$) across 40+ consecutive turns without single NaN or overflow.
2. **Completion Gating:** Outcome verifier successfully catches and rejects premature `CompleteGoal` emissions.
3. **Learned Grounding Weakness:** When a target entity (e.g. "Ajay") is absent from the visible contact list, the V55 rank head exhibited confusion, tapping "You" or avatar containers instead of targeting the Search icon or FAB. In accordance with AGENTS.md, this grounding failure is resolved via **continual learning and teacher distillation**, never via keyword heuristics.

---

## 6. Senior Continual Learning & Distillation Roadmap

To upgrade V55 beyond V54 while preventing catastrophic forgetting, development follows the Senior Engineer's 10-Step Continual Learning Roadmap:

### The 10-Step Continual Learning Protocol:
1. **Runtime Role Contract Restoration (Completed):** Lock `candRoles` to foundational `0/1/2` contract in `OnDeviceRankerEngine.kt`.
2. **Parent-Child Action Normalization (Completed):** Container absorption demotes child thumbnails in `ToolSchemaGenerator.kt`.
3. **Recurrent State Bounded Normalization (Completed):** Stabilize `h_next` to target norm $8.0$ clamped to $[-50.0, 50.0]$.
4. **Outcome-Based Completion Verification (Completed):** Gate `CompleteGoal` on observable outcome evidence.
5. **Physical Device Baseline Diagnostic (Completed):** Forensic analysis of live device logcat on Samsung Galaxy A26.
6. **V54 Teacher Distillation & Contact Search Replay (Current Step):**
   * Initialize V55 with strict parity verification: $\text{MSE}(V54, V55) < 1e-5$ on shared layers.
   * Freeze shared feature extraction backbones.
   * Fine-tune upper interaction and ranking layers using V54 teacher distillation loss:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{distill}} \mathcal{L}_{\text{KL}}(\mathbf{z}_{V55} \,\|\, \mathbf{z}_{V54}) + \lambda_{\text{replay}} \mathcal{L}_{\text{replay}}$$
   * Train on contact search hard-negatives (when target entity is absent, target MUST be Search affordance).
7. **Scorecard Regression Suite:** Evaluate checkpoint against 6-suite regression harness (`eval_v55_regression.py`).
8. **Sanitized FP16 ONNX Export:** Export with dynamic axes, zero NaNs, and graph optimization (`export_v55_fp16.py`).
9. **Physical Device A/B Benchmarking:** Execute identical anchor suite on Samsung Galaxy A26 comparing V54 vs V55.
10. **Promotion / Rollback Gate:** Promote V55 only if it matches or exceeds V54 pass rate across all physical anchors.

### Scorecard Certification & Promotion Gate:
```text
┌────────────────────────────────────────────────────────┐
│               6-SUITE SCORECARD GATES                  │
├────────────────────────────────────────────────────────┤
│ 1. Search Affordance Grounding Rate     ≥ 95.0%        │
│ 2. Feed Distractor Suppression Rate     ≥ 92.0%        │
│ 3. Contact Entity Grounding Rate        ≥ 90.0%        │
│ 4. Recurrent State Drift Across 20 Turns ≤ 0.05        │
│ 5. Observation Delta Accuracy           ≥ 88.0%        │
│ 6. Argument Pointer Span F1 Score       ≥ 92.0%        │
└────────────────────────────────────────────────────────┘
```

---

## 7. Repository Structure

```text
zipper/
├── app/                                    # Android Production Application
│   ├── src/main/assets/models/             # On-device quantized ONNX models
│   │   ├── karvish_v54_fp16.onnx           # Locked V54 control model (50.88 MB)
│   │   ├── karvish_v55.onnx                # V55 deployment candidate (50.88 MB)
│   │   ├── tokenizer.model                 # Embedded SentencePiece tokenizer
│   │   └── vocab.txt                       # WordPiece vocabulary
│   └── src/main/java/com/zipper/app/core/
│       ├── agent/
│       │   ├── ActionDispatcher.kt         # Android Accessibility action dispatch
│       │   ├── AgentRuntime.kt             # Closed-loop task orchestration & recovery
│       │   ├── ExternalOracleVerifier.kt   # Observable outcome verification
│       │   ├── OnDeviceRankerEngine.kt     # ONNX Runtime mobile inference engine
│       │   └── ToolSchemaGenerator.kt      # Generic accessibility-tree candidate builder
│       ├── distiller/
│       │   └── UiStateDistiller.kt         # Accessibility tree extractor
│       └── overlay/
│           └── KarvishControlOverlay.kt    # Real-time visual feedback HUD
│
├── tools/training/karvish_mobile_action/   # Model R&D and Training Infrastructure
│   ├── v52_locked_gold/                    # Frozen V52 physical baseline (26.78M)
│   ├── v54/                                # Frozen V54 search baseline & control JSON
│   │   ├── checkpoints/
│   │   │   └── v54_universal_search_best.pt
│   │   └── V54_LOCKED_CONTROL.json
│   └── v55/                                # V55 Active Research Branch
│       ├── model_v55.py                    # V55 PyTorch architecture definition
│       ├── init_v55_from_v54.py            # Weight transfer & parity verification
│       ├── train_v55_distilled.py          # Teacher distillation & continual learning
│       ├── eval_v55_regression.py          # 6-suite behavioral regression harness
│       ├── eval_v55_scorecard.py           # Multi-metric certification scorecard
│       └── export_v55_fp16.py              # Sanitized FP16 ONNX exporter
│
├── AGENTS.md                               # Authoritative Karvish Constitution
└── README.md                               # This documentation
```

---

## 8. Developer Quick Start, Build & Deployment Guide

### Prerequisites
* **Git + Git LFS:** Required to pull bundled ONNX/native assets (`git lfs install && git lfs pull`)
* **Android SDK:** Platform-Tools 34+, Android Studio Ladybug / Koala
* **Python:** 3.10+ with PyTorch 2.1+, ONNX 1.15+, Transformers (`pip install -r requirements.txt`)
* **Target Hardware:** Physical Android device (Developer Options & USB Debugging enabled)

### Local Secrets & SDK Configuration:
```bash
# 1. Copy secrets template and fill in optional cloud API keys (Groq, Firebase, Tencent, VoIPBin, SMTP)
cp secrets.properties.example secrets.properties

# 2. Ensure local.properties points to your Android SDK (sdk.dir=...)
# 3. For web/tracking-dashboard, copy config.example.js -> config.js
```
See [`MODEL_MANIFEST.md`](MODEL_MANIFEST.md) for the inventory of bundled Git LFS assets and optional external retraining checkpoints.

### Build and Install Debug APK:
```bash
# 1. Build debug APK using Gradle wrapper
./gradlew assembleDebug

# 2. Install APK to connected device via ADB
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

### Grant Accessibility Permissions:
```bash
# Enable Karvish Accessibility Service
adb shell settings put secure enabled_accessibility_services \
  com.zipper.app/com.zipper.app.accessibility.KarvishAccessibilityService

# Grant overlay permission
adb shell appops set com.zipper.app SYSTEM_ALERT_WINDOW allow
```

### Real-Time Logcat Monitoring:
Filter live inference events, candidate ranking, bounded norms, and overlay feedback:
```bash
adb logcat -v time -s "OnDeviceRankerEngine:*" "AgentRuntime:*" "KarvishControlOverlay:*" "ToolSchemaGenerator:*"
```

### Python Training & ONNX FP16 Export:
```bash
# Step 1: Initialize V55 with V54 parity verification
python tools/training/karvish_mobile_action/v55/init_v55_from_v54.py

# Step 2: Run Teacher Distillation with Search Replay
python tools/training/karvish_mobile_action/v55/train_v55_distilled.py

# Step 3: Run the 6-Suite Regression Scorecard
python tools/training/karvish_mobile_action/v55/eval_v55_scorecard.py

# Step 4: Export Sanitized Mobile ONNX FP16
python tools/training/karvish_mobile_action/v55/export_v55_fp16.py
```

---

## 9. License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
