# KARVISH UNIVERSAL V56 — AUTHORITATIVE IMPLEMENTATION PLAN
**Target Architecture**: Karvish V56 Universal Action Policy  
**Governing Standard**: Karvish Agent Constitution (`AGENTS.md`)  
**Frozen Baseline Control**: `tools/training/karvish_mobile_action/v56_locked_baseline/` (10/12 Physical Qualification, 83.3%)  
**Status**: Ready for Phase 1 Execution  

---

## 1. Governance & Baseline Protection Invariants

1. **Frozen Baseline Preservation**:
   * The baseline model qualified on the physical Samsung Galaxy A26 (`RZCY51MQP9K`) is permanently frozen in:
     `tools/training/karvish_mobile_action/v56_locked_baseline/`
   * Checkpoint: `v56_locked_distilled_best.pt` (SHA256: `685CFEE618B0CF950757299C7A954876CAF26CFFCE45D945CB93FDF1E8FC38C7`)
   * ONNX Model: `karvish_v56_locked.onnx` (SHA256: `3274910C072C27303A5E9C51046B8DAE0C1730642E84D81AEC0EBC1047AD7D47`)
   * **Mandate**: This baseline directory is immutable. It will never be overwritten, retrained, or altered. All V56 training, synthesis, and evaluation is strictly isolated in `tools/training/karvish_mobile_action/v56/`.
2. **Zero Heuristic Task-Planning (`AGENTS.md` Sections 0, 4, 5)**:
   * No app-specific branching in runtime or neural decoder (`if package == ...` or `if label == ...`).
   * No hardcoded coordinate hacks or oracle subgoals.
   * Affordance grounding, multi-step navigation, and completion confidence must be 100% learned through neural representation.

---

## 2. Four Behavioral Target Classes (Device Evidence-Driven)

Formulated directly from the 12 physical device episodes on the Samsung Galaxy A26:

### Class A: Universal Communication Lifecycle Grounding
* **Observed Device Failure**: In LinkedIn (`Manoj`), the model searched `"Manoj"`, opened the profile screen, but repeatedly looped back to re-click the top search field across 30 turns instead of selecting the prominent `Message` button affordance. In Gmail (`Karthik`), the model searched `"Karthik"` inside `Search in emails` rather than tapping the floating action button `Compose (+)`.
* **Learned Sequence**:
  $$\text{SEARCH\_RESULTS} \longrightarrow \text{RECIPIENT\_SELECTION} \longrightarrow \text{PROFILE/INBOX} \longrightarrow \text{COMPOSE/MESSAGE} \longrightarrow \text{INPUT} \longrightarrow \text{SEND}$$
* **Cross-App Scope**: 10+ communication apps (WhatsApp, Telegram, Signal, LinkedIn, Gmail, Slack, Discord, Instagram Direct, Messages, WeChat, Outlook) across 32 diverse recipient names and 25 payloads.
* **Target Margins**:
  * Profile `Message` button over search bar: $> +2.0$ pts
  * Inbox `Compose (+)` button over search bar: $> +2.0$ pts
  * Search Results recipient card over re-searching: $> +2.0$ pts

### Class B: Navigation Traps vs. Open Active Input
* **Observed Device Failure**: On Instagram Explore and store search landings, a bottom navigation tab (`Search and explore`) is already active (`is_selected=True`), causing the model to repeatedly tap the selected tab rather than the top editable input field (`Search`).
* **Principle**: $\text{SELECTED TAB } (\text{is\_selected=True}) \ne \text{ACTIVE EDITABLE INPUT } (\text{is\_editable=True})$.
* **Target Margin**: Active search input over selected tab: $> +2.0$ pts.

### Class C: Dual-Slot Action-Conditioned Span Extraction
* **Core Function**: Disentangle recipient entities from message payloads using neural pointer heads conditioned on candidate semantic roles without regex or heuristics:
  * **Slot 0**: `RECIPIENT_OR_QUERY` (conditioned on `SEARCH_INPUT` or `CONTACT_ITEM`).
  * **Slot 1**: `MESSAGE_PAYLOAD` (conditioned on `MESSAGE_INPUT` or `COMPOSE`).
* **Target Metric**: Span extraction exact match $> 95\%$.

### Class D: Evidence-Gated Completion (Learned BCE Calibration)
* **Senior's Mandate**: Avoid hardcoding arbitrary negative penalties (`-12/-15`) or probability tables. Instead, formulate completion as a supervised binary classification conditioned on generic observable UI evidence.
* **Generic Observable State Evidence**:
  * `SEARCH_RESULTS` $\longrightarrow$ `is_final: False`, `verification_evidence: 0.0`
  * `RECIPIENT_SELECTED` $\longrightarrow$ `is_final: False`, `verification_evidence: 0.0`
  * `COMPOSER_OPEN` $\longrightarrow$ `is_final: False`, `verification_evidence: 0.0`
  * `MESSAGE_TYPED` $\longrightarrow$ `is_final: False`, `verification_evidence: 0.0`
  * `SEND_DISPATCHED` $\longrightarrow$ `is_final: False`, `verification_evidence: 0.0` (unconfirmed transmission)
  * `OUTCOME_VERIFIED` (timestamp/checkmark, cart badge count $> 0$) $\longrightarrow$ `is_final: True`, `verification_evidence: 1.0`
* **Calibrated Output Gates**:
  * Intermediate non-final screens: $P(\text{complete}) < 0.05$
  * Terminal verified screens: $P(\text{complete}) > 0.90$

---

## 3. Canonical Record Schema

Every sample in the unified training corpus strictly follows the schema:

```python
# Intermediate step (e.g. inside profile or chat view)
{
    "goal": "open LinkedIn and message Manoj: congratulations on the new role",
    "current_phase": "PROFILE_VIEW",
    "candidate_set": [
        {"label": "Message", "role_idx": 1, "is_clickable": True, "bounds": [...]},
        {"label": "Search Manoj", "role_idx": 3, "is_clickable": True, "bounds": [...]},
        {"label": "Connect", "role_idx": 1, "is_clickable": True, "bounds": [...]},
    ],
    "correct_candidate": "Message",
    "action": "CLICK",
    "argument_slot": -1,
    "argument_value": "",
    "expected_state": "COMPOSER_OPENED",     # Observable UI concept, NOT package name
    "is_final": False,                       # Binary supervision label
    "verification_evidence": 0.0
}

# Terminal step (outcome verified on physical UI)
{
    "goal": "open LinkedIn and message Manoj: congratulations on the new role",
    "current_phase": "VERIFY_TRANSMISSION",
    "candidate_set": [
        {"label": "Sent 5:30 PM", "role_idx": 23, "is_clickable": False, "bounds": [...]},
        {"label": "complete", "role_idx": 1, "is_clickable": True, "bounds": [...]},
    ],
    "correct_candidate": "complete",
    "action": "WAIT",
    "argument_slot": -1,
    "argument_value": "",
    "expected_state": "OUTCOME_VERIFIED",
    "is_final": True,                        # Terminal verified turn
    "verification_evidence": 1.0
}
```

---

## 4. Execution Phases

### Phase 1: Dataset Generation (`build_v56_generalized_dataset.py`)
* Ingest frozen V54 replay anchors (invariance shield + Zepto price grounding).
* Generate full Class A communication lifecycles across 10 apps.
* Generate Class B search landing traps.
* Annotate Class C dual-slot spans (Slot 0 vs Slot 1).
* Generate Class D binary `is_final` labels with observable evidence.
* Synthesize candidate permutations ($p_0, p_1$) for 100% order invariance.

### Phase 2: Distillation Training (`train_v56_distilled.py`)
* Model: `KarvishModelV56` with asymmetric cross-attention, dual-slot span head, and sigmoid completion head:
  $$\hat{y}_{\text{final}} = \sigma(W_{\text{term}} h_{\text{task}} + b)$$
* Multi-task loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rank}} + 0.5 \mathcal{L}_{\text{action}} + 0.3 \mathcal{L}_{\text{span}} + 0.5 \mathcal{L}_{\text{KD}} + 0.4 \mathcal{L}_{\text{BCE}}(\hat{y}_{\text{final}}, y_{\text{final}})$$
* Teacher: Distill from frozen V54 teacher (`tools/training/karvish_mobile_action/v54/checkpoints/v54_universal_search_best.pt`).
* Target Checkpoint: `tools/training/karvish_mobile_action/v56/checkpoints/v56_distilled_best.pt`.

### Phase 3: Pre-Export Regression Suite (`eval_v56_regression.py`)
* **Gate 1**: 12/12 Frozen baseline anchors pass (100%).
* **Gate 2**: Candidate permutation invariance (100% identical top-1 across shuffled order).
* **Gate 3**: Class A margins ($> +2.0$ pts for Message, Compose, and Recipient).
* **Gate 4**: Class B margin ($> +2.0$ pts for Search Input over Selected Tab).
* **Gate 5**: Class C span extraction accuracy $> 95\%$.
* **Gate 6**: Class D calibration ($P < 0.05$ on intermediate, $P > 0.90$ on verified).
* **Gate 7**: Zepto price grounding (₹13 beats ₹19).

### Phase 4: ONNX Mobile Export & Parity (`export_v56_onnx.py`)
* Export dynamic-axes ONNX model to `tools/training/karvish_mobile_action/v56/export/karvish_v56.onnx`.
* Verify numerical parity against PyTorch ($< 10^{-4}$ max absolute error).
* Ensure model size $< 55 \text{ MB}$.

### Phase 5: APK Deployment & Closed-Loop Physical Device Qualification
* Copy ONNX artifact to `app/src/main/assets/karvish_v56.onnx`.
* Build and install debug APK onto Samsung Galaxy A26 (`RZCY51MQP9K`).
* Re-evaluate the two target failure tasks on physical device:
  1. **LinkedIn Message to Manoj**: `open LinkedIn and message Manoj: congratulations on the new role` (verify finding `Message` button without looping).
  2. **Gmail Compose to Karthik**: `open Gmail and send mail to Karthik: meeting tomorrow` (verify selecting `Compose (+)`).
* Verify zero regressions on baseline tasks (Zepto tomato, Telegram message, YouTube lo-fi).
