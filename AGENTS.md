# KARVISH — AGENT CONSTITUTION
# ============================================================
# AUTHORITATIVE PROJECT CONTRACT
#
# Current control:
#   V51.7-GOLD (Historical physical control baseline)
#   V52-LOCKED-GOLD (Authoritative frozen baseline for next-gen V53+ training)
#
# Current development direction:
#   V53+ Spatial Layout, Goal-Cached Asymmetric Attention & Universal Agent
#
# Primary objective:
#   Transfer task intelligence from deterministic heuristics into
#   the learned Karvish policy while preserving physical reliability.
#
# This file is mandatory project policy.
# The coding agent MUST follow it before modifying code, datasets,
# training pipelines, runtime logic, evaluation harnesses, or model
# architecture.
# ============================================================


# ============================================================
# 0. NON-NEGOTIABLE DEVELOPMENT LAWS
# ============================================================

1. V51.7-GOLD and V52-LOCKED-GOLD are frozen physical control baselines.

2. NEVER modify, overwrite, retrain, or silently replace V51.7-GOLD or
   V52-LOCKED-GOLD.

3. Every V53+ / next-gen experiment MUST be explicitly compared against
   V52-LOCKED-GOLD, V51.7-GOLD, or the immediately preceding frozen candidate.
   All future training MUST branch from or benchmark against V52-LOCKED-GOLD.

4. NEVER solve a learned-intelligence failure by silently adding a
   task-specific runtime heuristic.

5. Task intelligence belongs in:
   - model architecture
   - training data
   - training objectives
   - learned representations
   - learned state
   - learned argument grounding
   - learned recovery behavior

6. Deterministic runtime code is allowed for:
   - Android/platform interaction
   - action execution
   - safety
   - authorization
   - validity checking
   - stale-target protection
   - observation collection
   - transport
   - timing
   - logging
   - test infrastructure
   - evidence verification

7. Deterministic runtime code MUST NOT secretly decide the user's task.

8. NEVER use goal keywords, package names, fixed coordinates, known UI
   labels, or predetermined step sequences to manufacture apparent
   model intelligence.

9. NEVER inject:
   - correct target
   - correct candidate
   - correct next action
   - future UI state
   - evaluator state
   - oracle progress
   - hidden subgoal
   - success label
   into autonomous inference.

10. If a failure is caused by insufficient learned capability, prefer:
      failure reproduction
      -> diagnosis
      -> training data
      -> model/loss improvement
      -> retraining
      -> physical evaluation

    instead of:
      failure
      -> heuristic patch

11. NEVER call a capability "learned", "autonomous", "general", or
    "model-driven" unless the evidence actually supports that claim.

12. Physical-device behavior is higher-value evidence than synthetic
    benchmark performance.

13. Offline metrics are diagnostic. They are NOT substitutes for
    closed-loop physical evaluation.

14. A passing benchmark must not be manufactured through the evaluator.

15. When uncertain, preserve the existing working baseline and run a
    controlled experiment instead of making a broad irreversible change.


# ============================================================
# 1. PROJECT IDENTITY
# ============================================================

Karvish is an agentic AI/action-intelligence research project.

The long-term objective is a compact universal action policy capable of:

    USER GOAL
       ->
    OBSERVE ENVIRONMENT
       ->
    UNDERSTAND CURRENT STATE
       ->
    SELECT NEXT ACTION
       ->
    EXECUTE
       ->
    OBSERVE RESULT
       ->
    UPDATE TASK STATE
       ->
    VERIFY PROGRESS
       ->
    RECOVER IF NECESSARY
       ->
    CONTINUE
       ->
    VERIFY COMPLETION

The current primary environment is Android.

Future environments may include:
- desktop
- web
- other devices
- visual environments
- robotics
- games

Future environments MUST NOT distract current development from the
Android-first objective unless explicitly requested.


# ============================================================
# 2. KARVISH IS NOT
# ============================================================

Do NOT turn Karvish into:

- a chatbot
- a generic conversational assistant
- a collection of app automations
- a macro recorder
- a coordinate-clicking system
- a scripted demo
- a package-specific automation framework
- a collection of hardcoded workflows
- a benchmark-specific solver
- an evaluator-specific solver
- a cloud-only action system
- a runtime wrapper around another model that secretly performs the task
- a system where deterministic code chooses the action and the neural
  model merely approves it


# ============================================================
# 3. CORE AUTONOMY CONTRACT
# ============================================================

At inference time, the action policy must make decisions from information
that is legitimately available at that moment.

Allowed policy inputs:

- user goal
- current live observation
- currently available candidates/actions
- relevant action history
- previous observable outcomes
- learned task state
- observable UI/environment state
- model memory that was legitimately accumulated from prior observations

Forbidden policy inputs:

- future screen
- future action
- evaluator success flag
- hidden task state
- oracle target index
- oracle candidate ID
- ground-truth progress
- future subgoal
- runtime-selected correct objective
- benchmark answer
- test-specific expected action

The production policy must select the next action dynamically.

NEVER precompute:

    Goal
      ->
    Action 1
      ->
    Action 2
      ->
    Action 3
      ->
    ...

and then ask the model to approve those actions.

The model must genuinely participate in next-action selection.


# ============================================================
# 4. MODEL RESPONSIBILITY VS RUNTIME RESPONSIBILITY
# ============================================================

## 4.1 MODEL RESPONSIBILITY

The learned model should progressively learn:

- goal interpretation
- candidate relevance
- UI grounding
- action selection
- action type
- action arguments
- task progress
- subgoal estimation
- state transition understanding
- recovery decisions
- termination confidence
- cross-app task state
- adaptation to changed UI
- recognition of whether an action actually advanced the goal

## 4.2 RUNTIME RESPONSIBILITY

The runtime may handle:

- AccessibilityService
- AccessibilityNodeInfo
- observation capture
- candidate extraction
- action dispatch
- Android APIs
- coordinate conversion
- safe viewport bounds
- stale-target validation
- action validity
- permission handling
- timeouts
- transport
- process management
- persistence
- logging
- crash handling
- safety authorization
- deterministic evidence verification
- test setup

The runtime MUST NOT secretly perform task planning.

## 4.3 SAFETY BOUNDARY

The model may request:

    TAP(candidate)
    TYPE_TEXT(candidate, argument)
    SCROLL(candidate/direction)
    SUBMIT(candidate)
    BACK
    WAIT
    OPEN_APP(target)
    COMPLETE

The runtime may reject an invalid or unsafe action.

That rejection is a safety decision, not model intelligence.

The runtime MUST NOT replace the rejected action with the correct
task-specific action unless the behavior is explicitly classified as
test-only infrastructure.


# ============================================================
# 5. THE MOTORS AND BRAKES RULE
# ============================================================

Deterministic code is NOT automatically bad.

Karvish explicitly distinguishes:

    MOTORS / BRAKES
        vs
    TASK SPOONFEEDING

## 5.1 LEGITIMATE PLATFORM PLUMBING

Allowed:

- Accessibility APIs
- Android action dispatch
- coordinate conversion
- safe viewport clamping
- gesture safety
- stale node detection
- candidate validity checks
- focus validation
- IME synchronization
- timeout handling
- permission handling
- transport
- logging
- process supervision
- crash recovery
- test reset
- evidence collection
- independent verification

Example:

    "Do not send a touch into Android's gesture-navigation region."

This is platform safety.

## 5.2 TASK SPOONFEEDING

Forbidden in the final autonomous policy:

    if "search" in goal:
        boost search

    if previous_action == TYPE_TEXT:
        boost submit

    if package == YouTube:
        perform known sequence

    if package == LinkedIn:
        suppress feed cards

    if goal contains "SpaceX":
        type "SpaceX"

    if goal contains "then press back":
        force BACK

    if candidate label == known benchmark target:
        boost candidate

These are task-intelligence shortcuts.

Temporary research versions may contain such logic ONLY when:
- explicitly documented
- isolated
- named
- measurable
- excluded from claims of pure learned autonomy
- scheduled for ablation/removal

No temporary heuristic may silently become permanent production logic.


# ============================================================
# 6. CURRENT FROZEN CONTROL
# ============================================================

## 6.1 V51.7-GOLD

V51.7-GOLD is the immutable physical control baseline.

Artifact:

    tools/training/karvish_mobile_action/v51_phase6_7/checkpoints/v51_7_gold.pt

Architecture:

    KarvishModelV51_7

Approximate parameter count:

    10.88M

The exact checkpoint hash stored in the repository is authoritative.

V51.7-GOLD MUST NOT be overwritten.

Its:
- weights
- decoder configuration
- evaluation configuration
- benchmark definition
- recorded results

must remain reproducible.

## 6.2 WHY V51.7-GOLD EXISTS

V51.7-GOLD is the control group for de-spoonfeeding.

It answers:

    "What does the working system do before we remove a heuristic?"

Every V52+ experiment must preserve a reproducible comparison against it.

## 6.3 HISTORICAL BASELINES

V42, V48, V49, V50 and other historical versions may be used for:
- regression analysis
- architectural comparison
- data provenance
- failure analysis

They MUST NOT be silently substituted for V51.7-GOLD.

## 6.4 V52-LOCKED-GOLD

V52-LOCKED-GOLD is the authoritative frozen physical control baseline for the
de-spoonfed universal agent, locked on September 30, 2026.

Artifacts:

    Checkpoint (PyTorch weights):
    tools/training/karvish_mobile_action/v52_locked_gold/v52_locked_gold.pt
    SHA256: fa38aa02c290f8839ee4c1b162136a32bd5618f5d3f7a11bb3b824f288c1d04c

    Runtime ONNX:
    tools/training/karvish_mobile_action/v52_locked_gold/karvish_v52_locked_gold.onnx
    SHA256: 247a4ee82e23f3859e8582eea2ba02d0935078ff057dd90545dab1fa91d6810e

    Architecture & Standalone Git Tracking:
    tools/training/karvish_mobile_action/v52_locked_gold/
    Git Tag: v52-locked-gold
    Commit: 44b638f

Architecture:

    KarvishModelV52

Parameter count:

    26,778,590 (26.78M)

Qualification:

    Validation Top-1: 100.0%
    Device-verified: Samsung Galaxy A26 (RZCY51MQP9K)
    Anchor task pass rate: 12/12 (YouTube, Instagram, Settings, Calculator, Zepto)

Constitutional Rules:

1. V52-LOCKED-GOLD MUST NOT be modified, retrained, or overwritten.
2. All future training (V53+, Goal Caching, 2D Spatial Geometry, InfoNCE Contrastive
   Loss, FP16/W8A16 QAT) MUST branch from or be compared directly against
   V52-LOCKED-GOLD.
3. If any experimental candidate regresses or fails on device, V52-LOCKED-GOLD
   remains the pristine, unaffected, working control model that can be restored
   instantly.


# ============================================================
# 7. CURRENT DE-SPOONFEEDING PROGRAM
# ============================================================

The purpose of de-spoonfeeding is NOT to delete working code blindly.

The purpose is:

    HEURISTIC INTELLIGENCE
          ->
    TRAINING EVIDENCE
          ->
    LEARNED MODEL CAPABILITY
          ->
    HEURISTIC REMOVED
          ->
    PHYSICAL REVALIDATION

Never remove all heuristics simultaneously.

Each capability must be transferred independently.

## 7.1 PHASE A — SEARCH / FEED GROUNDING

Removed:

    +3.5 search affordance boost
    -5.0 feed distractor penalty

These are now forbidden in the de-spoonfed decoder.

Phase A physical evidence:

    V51.7-GOLD: 7/8 overall
    Phase A:     6/8 overall

The important conclusion is NOT:

    "75% of the system is pure neural."

The correct conclusion is:

    "The search/feed decoder heuristics were unnecessary for the
     majority of tested tasks but remained useful on the LinkedIn
     distractor case."

Observed Phase A behavior:

- Calculator: preserved
- YouTube SpaceX: preserved
- Settings Battery: preserved/improved
- Keep: preserved/improved
- Play Store: remained partial
- YouTube lo-fi: preserved
- Instagram photography: preserved/improved
- LinkedIn software engineer: regressed

The LinkedIn failure MUST be treated as a learned grounding weakness,
not as justification for immediately restoring the removed heuristic.

## 7.2 LINKEDIN / FEED HARD NEGATIVE REQUIREMENT

When a prominent feed card competes with a search affordance, training should
teach candidate relevance directly.

Useful hard negatives include:

- sponsored cards
- content cards
- company cards
- profile cards
- Like
- Share
- Follow
- View image
- recommendation tiles
- large visual feed elements
- unrelated navigation elements

The training objective is NOT:

    "Search always beats feed."

The objective is:

    "Select the candidate that best advances the current goal."

Hard-negative training MUST contain cross-app variation.

Do not replace a LinkedIn-specific heuristic with a LinkedIn-specific
learned shortcut.

## 7.3 PHASE B — INPUT FIELD HEURISTICS

Target:

    -6.0 editable-field penalty after text entry

Before removing it, train:

- active-field state
- populated-field state
- focus transitions
- multi-turn editing
- text replacement
- cursor/focus changes
- post-input UI mutation
- repeated-input prevention

The model should learn when an editable field remains relevant and when
another action has become appropriate.

## 7.4 PHASE C — SUBMIT / COMMIT HEURISTIC

Target:

    +4.0 submit/IME action boost

The model must learn the transition:

    INPUT
      ->
    OBSERVE UPDATED UI
      ->
    RECOGNIZE COMMIT AFFORDANCE
      ->
    SUBMIT

The decoder must not force submission merely because the previous action
was TYPE_TEXT.

## 7.5 PHASE D — REGEX ENTITY EXTRACTION

Target:

    extract_entity_from_goal
    regex query extraction
    task-critical keyword extraction

Final objective:

    Goal tokens
       ->
    learned span/copy/pointer head
       ->
    argument

Examples:

    "Search SpaceX on YouTube"
                  ->
              "SpaceX"

    "Message Rahul: meeting at 5"
                  ->
              "Rahul"
              "meeting at 5"

The learned argument mechanism must generalize to:

- search queries
- names
- phone numbers
- URLs
- Unicode
- message bodies
- numerical values
- form values

Regex may remain temporarily for:
- data collection
- diagnostic comparison
- legacy compatibility
- explicitly marked teacher-only infrastructure

It must not remain the final source of task-critical intelligence.


# ============================================================
# 8. HEURISTIC LEDGER
# ============================================================

Every runtime heuristic that affects action selection MUST be documented.

Each entry must contain:

    HEURISTIC ID
    PURPOSE
    CURRENT LOCATION
    INPUTS USED
    EXACT EFFECT
    WHY IT EXISTS
    WHETHER IT IS PLATFORM PLUMBING OR TASK INTELLIGENCE
    ABLATION PLAN
    LEARNED REPLACEMENT
    VALIDATION PLAN
    REMOVAL STATUS

Any heuristic that changes candidate ranking based on:
- goal words
- app identity
- UI labels
- previous actions
- benchmark identity
- task identity

is presumed to be TASK INTELLIGENCE until proven otherwise.

If a heuristic is discovered during code review and is not documented,
STOP and classify it before proceeding.


# ============================================================
# 9. FAILURE-TO-TRAIN RULE
# ============================================================

When the model fails:

DO NOT immediately patch the decoder.

First classify the failure.

Possible categories:

1. Observation failure
2. Candidate extraction failure
3. Candidate availability failure
4. Grounding failure
5. Action-type failure
6. Argument extraction failure
7. Task-state failure
8. Progress-estimation failure
9. Transition-model failure
10. Recovery failure
11. Termination failure
12. Runtime execution failure
13. Safety rejection
14. Evaluation/verifier failure
15. Data-quality failure
16. Distribution-shift failure

Then determine ownership.

Example:

    Search candidate exists
    Model chooses feed card
    Android executes correctly

Classification:

    learned grounding failure

Correct response:

    collect hard negatives
    retrain
    evaluate

Incorrect response:

    add "search" score boost

Another example:

    Model requests TAP on stale node
    Runtime rejects it

Classification:

    legitimate runtime safety

No neural patch is necessarily required.


# ============================================================
# 10. LEARNED TASK STATE
# ============================================================

Task state is learned intelligence.

The model may learn:

- current objective
- completed requirements
- pending requirements
- active subgoal
- task progress
- previous action outcome
- recovery state
- termination confidence
- relevant observed values
- cross-app dependencies

The model MUST NOT receive:

- oracle subgoal
- oracle progress
- future state
- correct next objective
- evaluator state
- hidden completion label

Training labels may supervise these variables.

Inference must derive them from legitimate observations.

The preferred loop is:

    GOAL
      ->
    OBSERVATION
      ->
    TASK STATE
      ->
    ACTION
      ->
    OBSERVATION DELTA
      ->
    UPDATED TASK STATE
      ->
    NEXT ACTION

Repeated screens must not automatically imply progress.

An action that does not produce expected observable change must be treated
as evidence that the previous decision may have failed.


# ============================================================
# 11. ACTION-CANDIDATE ALIGNMENT
# ============================================================

The model must select actions that correspond to currently available
candidates.

For candidate-conditioned actions:

    candidate_i
       +
    current task state
       +
    goal
       ->
    action type

The model MUST NOT output a target candidate that is absent from the
current candidate set.

Candidate IDs must not become hidden oracle labels.

Candidate order must not be treated as semantic truth.

Training and evaluation should use candidate permutation where possible.

The system must distinguish:

- candidate availability
- candidate ranking
- action type
- action argument
- global actions
- termination


# ============================================================
# 12. UNIVERSAL UI REPRESENTATION
# ============================================================

The model must learn reusable UI concepts rather than app workflows.

Important universal concepts include:

- clickable affordance
- editable field
- selected option
- selectable option
- submit/commit action
- navigation affordance
- back action
- scroll container
- modal
- dialog
- permission prompt
- result item
- list item
- tab
- menu
- toggle
- slider
- checkbox
- radio option
- confirmation
- completion evidence

Do not encode these concepts using package-specific rules.

A UI concept must remain meaningful when:
- label changes
- app changes
- layout changes
- candidate order changes
- screen density changes
- irrelevant distractors increase


# ============================================================
# 13. ARGUMENT HANDLING
# ============================================================

Arguments are part of task intelligence.

The final architecture should distinguish:

    ACTION
    TARGET
    ARGUMENT

Example:

    TYPE_TEXT
    target = current search field
    argument = "SpaceX"

The model should eventually determine the argument through learned
goal-to-span/entity grounding.

Do not use regex extraction as a substitute for learned argument grounding
in the final autonomous policy.

Never silently alter user text.

Arguments must preserve:
- capitalization where relevant
- Unicode
- spaces
- punctuation
- URLs
- phone numbers
- message contents
- numerical values


# ============================================================
# 14. APP GENERALIZATION
# ============================================================

No app-specific workflow logic.

Forbidden:

    if package == "com.google.android.youtube":
        do sequence X

    if package == "com.linkedin.android":
        suppress candidate Y

    if package == Calculator:
        use keypad sequence

Allowed:

    generic UI representation
    generic semantic roles
    generic candidate ranking
    generic task state
    generic action types
    generic recovery
    platform APIs

Application identity may be observable information, but it must not become
a shortcut that directly determines the action.

Evaluation must separate:

- known app / known task family
- known app / unseen task
- unseen app / known task family
- unseen app / unseen task
- multi-app tasks


# ============================================================
# 15. TRAINING DATA CONTRACT
# ============================================================

Training data must prioritize correctness and provenance over quantity.

Preferred data:

1. verified human demonstrations
2. verified real-device trajectories
3. verified interactive environment trajectories
4. verified teacher-generated trajectories
5. hard-negative grounding data
6. recovery trajectories
7. counterfactual data
8. carefully controlled synthetic data

Never blindly trust generated data.

Each trajectory should preserve, where available:

    goal
    observation_before
    candidate_set
    history
    action
    target
    argument
    observation_after
    expected_transition
    observed_transition
    outcome
    verification
    provenance
    confidence

Missing information must remain missing.

Never silently convert unknown into correct.

Never silently infer an oracle target from evaluator labels.


# ============================================================
# 16. DATA PROVENANCE
# ============================================================

Every external or internally derived dataset must record:

- source
- license/permission
- version
- acquisition method
- preprocessing version
- deduplication policy
- filtering policy
- task family
- app identity where legitimately observable
- teacher model/version if applicable
- whether observation was static or interactive
- whether action was actually executed
- whether outcome was verified
- whether labels are human, teacher, synthetic, or inferred

Data without acceptable provenance must be quarantined.

Do not mix incompatible sources without normalization.

Do not silently change dataset versions between experiments.


# ============================================================
# 17. NO ORACLE LEAKAGE
# ============================================================

The following must NEVER enter autonomous model input:

- correct_candidate
- target_index
- oracle_target
- target_node_index
- expected_action
- evaluator_state
- ground_truth_progress
- future_screen
- future_action
- success_flag
- hidden completion state
- test answer
- teacher-selected future action

Training supervision is allowed.

Inference leakage is not.

Before every major training run, audit the complete inference input path.

Before every physical evaluation, audit:
- runtime request
- model input
- model output
- dispatcher
- verifier

No hidden oracle information may pass through any layer.


# ============================================================
# 18. TRAINING OBJECTIVES
# ============================================================

Use objectives appropriate to the capability.

Potential objectives:

- candidate ranking
- pairwise hard-negative ranking
- action-type classification
- argument/span extraction
- task-state prediction
- progress estimation
- transition prediction
- termination prediction
- recovery classification
- calibrated uncertainty
- preference/advantage learning for verified interactive trajectories

Do not train action loss against targets unavailable in the candidate set.

Do not optimize synthetic metrics at the expense of physical behavior.

A metric may be reported only with:
- definition
- dataset
- split
- horizon
- candidate availability
- evaluation protocol


# ============================================================
# 19. HARD NEGATIVES
# ============================================================

Hard negatives are strongly encouraged for grounding.

Useful negatives include:

- visually prominent irrelevant items
- duplicate labels
- semantically related but wrong controls
- feed cards
- sponsored cards
- wrong search affordances
- wrong text fields
- navigation elements
- disabled controls
- stale candidates
- repeated controls
- nearby distractors

Hard negatives must vary across applications.

Do not create a learned shortcut where:
- one label always wins
- one app always wins
- one screen position always wins
- one candidate index always wins


# ============================================================
# 20. RECOVERY
# ============================================================

A failed action must not automatically repeat forever.

The loop is:

    ACT
      ->
    OBSERVE
      ->
    DETECT OUTCOME
      ->
    CLASSIFY FAILURE
      ->
    UPDATE TASK STATE
      ->
    REPLAN
      ->
    ACT AGAIN

Recovery cases must include:

- wrong candidate
- stale target
- no-op
- screen mutation
- popup
- permission dialog
- keyboard change
- loading delay
- navigation interruption
- duplicate labels
- disabled control
- unexpected screen
- accidental distractor tap

Repeated identical actions without state progress must be treated as a
potential loop.


# ============================================================
# 21. TERMINATION
# ============================================================

The model may predict termination confidence.

The model MUST NOT be allowed to claim success solely because:

- an action was dispatched
- a package is foreground
- a generic word is visible
- a screen changed
- a timeout did not occur

Whenever possible:

    ACTION
      ->
    OBSERVE
      ->
    POSTCONDITION
      ->
    VERIFY
      ->
    COMPLETE / CONTINUE / RECOVER

Runtime verification may remain deterministic.

Runtime verification is evidence infrastructure, not learned intelligence.


# ============================================================
# 22. PHYSICAL EVALUATION INTEGRITY
# ============================================================

Physical evaluation is the authoritative test for actual Android behavior.

Every physical task must record:

- task ID
- goal
- device
- Android version
- model version
- decoder version
- runtime version
- start state
- action trace
- candidate trace
- model outputs
- arguments
- timestamps
- observations
- final state
- verification result
- failure classification

Every action attributed to the agent must correspond to a model inference
and a legitimate runtime dispatch.

ADB MUST NOT perform the actual task behind the model's back.


# ============================================================
# 23. PHYSICAL RESULT CLASSIFICATION
# ============================================================

Every physical result must be classified as exactly one of:

## CLEAN TRUE PASS

The task was completed through the intended action sequence without
meaningful accidental actions, loops, or evaluator assistance.

## MESSY PASS

The task was eventually completed after an incorrect intermediate action,
distractor interaction, or recovery.

## PARTIAL

Some requested state was reached, but the complete task/postcondition
was not established.

## TRUE FAILURE

The task was not completed, timed out, looped, terminated incorrectly,
or failed verification.

Do NOT silently convert PARTIAL or MESSY PASS into CLEAN PASS.


# ============================================================
# 24. NO-SHAM VERIFICATION
# ============================================================

Never mark a task successful merely because:

- target package is foreground
- generic UI text exists
- app header exists
- action count is nonzero
- screen changed
- model confidence was high

A strong physical pass should establish:

1. Correct action alignment
2. Correct target
3. Correct argument where applicable
4. Observable state mutation
5. Required final postcondition
6. No evaluator action causing the success

For example, a search task should establish that:
- the intended field received the intended query
- the query was submitted
- the resulting state supports successful search
- the required final state was reached


# ============================================================
# 25. TEST HARNESS SEPARATION
# ============================================================

Test infrastructure is separate from agent intelligence.

Allowed test setup:

- install APK
- launch app
- reset state
- clear app data
- configure environment
- collect logs
- capture screenshots
- collect Accessibility trees
- reset device state

Forbidden test behavior:

- entering the answer
- clicking the target
- submitting the query
- selecting the required item
- modifying the task state
- injecting hidden completion state

If a harness performs an action, label it TEST SETUP.

Never attribute harness behavior to Karvish.


# ============================================================
# 26. ADB POLICY
# ============================================================

ADB may be used for:

- installation
- launch
- reset
- diagnostics
- logs
- screenshots
- test setup
- cleanup
- transport

ADB MUST NOT execute the user's actual task while the result is attributed
to the model.

If ADB is used for a task during debugging, explicitly label that run as
NON-AUTONOMOUS and exclude it from autonomous evaluation.


# ============================================================
# 27. REGRESSION DISCIPLINE
# ============================================================

Every candidate model must be evaluated against:

1. V51.7-GOLD
2. immediate parent checkpoint
3. relevant historical baselines
4. current de-spoonfeeding benchmark
5. physical sealed tasks
6. historical regression tasks

Do not declare improvement from one metric.

A model that improves candidate accuracy but breaks physical execution is
not automatically an improvement.

A model that improves one app while regressing another must report both.

Never hide primitive regressions behind aggregate averages.


# ============================================================
# 28. EXPERIMENT DESIGN
# ============================================================

Controlled comparisons must keep constant where appropriate:

- dataset
- split
- initialization
- seed
- optimizer
- training budget
- checkpoint selection
- evaluation suite

When a variable is intentionally changed, document it.

Do not compare:
- one model trained for 3 epochs
- another model trained for 30 epochs

and attribute the difference solely to architecture.

Do not compare unmatched datasets and claim architectural causality.


# ============================================================
# 29. ABLATION PROTOCOL
# ============================================================

Every intelligence heuristic removal must follow:

    1. Freeze baseline
    2. Identify exact heuristic
    3. Record exact effect
    4. Remove ONLY that heuristic
    5. Keep everything else fixed
    6. Run offline diagnostics
    7. Run physical evaluation
    8. Classify failures
    9. Train learned replacement if needed
    10. Re-evaluate
    11. Freeze successful candidate
    12. Proceed to next heuristic

Never perform a multi-variable deletion and then claim causal attribution.


# ============================================================
# 30. CURRENT DE-SPOONFEEDING ORDER
# ============================================================

The preferred order is:

    A. Search/feed candidate biases
       ->
    B. Input-field suppression
       ->
    C. Submit/commit promotion
       ->
    D. Regex argument extraction
       ->
    E. Remaining task-specific decoder intelligence

Each stage must be physically evaluated before proceeding.

If a stage fails:
- preserve the failed candidate
- preserve logs
- diagnose
- train
- retest

Do NOT restore the removed heuristic merely because the first ablation
regressed.


# ============================================================
# 31. ONE PRIMARY RUNTIME POLICY
# ============================================================

The long-term runtime target is:

    ONE compact primary learned action policy
    +
    deterministic Android execution
    +
    deterministic safety
    +
    deterministic evidence verification

Additional learned models may exist only when measured evidence shows
specialization is necessary.

Training-time teachers are allowed.

Large external models may be used for:
- trajectory generation
- critique
- ranking
- distillation
- hard-negative generation
- reasoning supervision

Teacher models must NOT silently become required runtime inference assets.


# ============================================================
# 32. SPECIALIST MODEL RULE
# ============================================================

A runtime specialist must have a measurable reason to exist.

Before adding one, report:

- capability gap
- primary-model failure rate
- specialist benefit
- latency cost
- RAM cost
- storage cost
- invocation rate
- failure behavior
- fallback behavior

Do NOT run many models on every action and vote by default.

Do NOT hide a required learned model outside the deployment asset budget.


# ============================================================
# 33. VISUAL MODEL POLICY
# ============================================================

Accessibility information is the primary Android observation mechanism.

Visual models may be introduced when evidence shows a genuine gap.

Do not add a visual model merely because it sounds more intelligent.

A visual model must be evaluated against:

    Accessibility-only baseline
    vs
    Accessibility + visual capability

The visual capability must demonstrate measurable benefit.

The final deployed visual asset counts toward the learned-asset budget.


# ============================================================
# 34. MODEL SIZE AND DEPLOYMENT
# ============================================================

The long-term offline learned-asset budget is:

    < 50,000,000 bytes

This includes ALL required learned inference assets:

- primary model
- specialists
- tokenizer assets
- learned embeddings
- auxiliary heads
- learned lookup tables
- adapters
- other learned runtime components

Do NOT hide learned assets outside the declared budget.

Parameter count alone does not determine compliance.

Actual exported bytes are authoritative.


# ============================================================
# 35. PERFORMANCE
# ============================================================

Target device qualification:

- warm p95 decision latency < 300 ms
- cold initialization < 2 seconds
- incremental agent RAM < 250 MB

Also measure:

- thermal behavior
- sustained performance
- battery impact
- interruption handling
- crash behavior
- fail-closed behavior

Desktop CUDA latency is NOT Android-device performance evidence.


# ============================================================
# 36. TRAINING SCALE
# ============================================================

Training should grow based on validation evidence.

Suggested progression:

    2k–5k
        pipeline validation

    20k–50k
        serious policy development

    100k–300k
        only when learning curves justify expansion

Do NOT increase dataset size simply because more data exists.

Before expanding data, determine whether the bottleneck is:

- representation
- label quality
- task diversity
- long horizon
- recovery
- candidate grounding
- argument grounding
- task state
- objective mismatch


# ============================================================
# 37. LONG-HORIZON TRAINING
# ============================================================

Short one-step examples are insufficient for universal autonomy.

Training must progressively include:

- 1-step
- 2–4-step
- 5–9-step
- 10–19-step
- 20–49-step
- 50+ where justified

Later training should include:

- cross-app state transfer
- interruptions
- recovery
- stale targets
- no-progress states
- repeated screens
- changing layouts
- delayed UI
- dependencies between actions

Static replay MUST NOT be described as closed-loop execution.


# ============================================================
# 38. GENERALIZATION
# ============================================================

Training must vary:

- application
- UI layout
- labels
- wording
- candidate ordering
- distractors
- screen structure
- task length
- interaction history
- failure conditions

Evaluation must contain held-out:

- applications
- task families
- trajectories
- goal wording
- UI layouts

Never call the model "universal" because it succeeded on several known apps.


# ============================================================
# 39. MEMORY
# ============================================================

Memory must preserve information that was genuinely observed or verified.

Memory must NOT become:

- hidden oracle state
- predetermined workflow
- app-specific action macro
- future-state cache

Stored information should have provenance where possible.

Memory retrieval must not directly dictate an action without the current
observation supporting that action.


# ============================================================
# 40. SAFETY
# ============================================================

Consequential actions require independent safety controls.

Examples:

- sending messages
- purchases
- deletion
- external communication
- sensitive settings
- account changes
- irreversible operations

Never bypass:
- permissions
- confirmations
- authorization
- safety checks

merely to improve benchmark scores.

A benchmark win is worthless if achieved by unsafe execution.


# ============================================================
# 41. CLAIM DISCIPLINE
# ============================================================

The agent MUST use precise terminology.

DO NOT say:

    "100% neural"

unless every task-critical decision in the claimed scope is demonstrated
to come from learned inference.

DO NOT say:

    "fully autonomous"

if runtime code determines task actions.

DO NOT say:

    "general"

from a small known-app evaluation.

DO NOT say:

    "learned"

when behavior is supplied by deterministic code.

DO NOT say:

    "passed"

for PARTIAL or MESSY results without classification.

Every report must distinguish:

    MODEL DECISION
    RUNTIME DECISION
    SAFETY DECISION
    TEST-HARNESS OPERATION
    VERIFIER DECISION
    HARDCODED FALLBACK


# ============================================================
# 42. REPORTING REQUIREMENTS
# ============================================================

Every major experiment report should include:

- model version
- parent checkpoint
- parameter count
- checkpoint hash
- learned asset size
- dataset version
- dataset counts
- train/validation/heldout split
- training configuration
- seeds
- objective
- offline metrics
- physical metrics
- baseline comparison
- heuristic configuration
- failures
- failure classification
- recovery behavior
- false completion
- wrong target
- loop rate
- argument correctness
- latency
- memory
- known limitations

Never report only the best metric.


# ============================================================
# 43. BEFORE MODIFYING CODE
# ============================================================

Before a major change, the coding agent MUST answer internally:

1. What problem is being solved?
2. Is this model intelligence, runtime plumbing, safety, or evaluation?
3. Which component owns the responsibility?
4. Does the change inject task knowledge?
5. Does it depend on an app name?
6. Does it depend on goal keywords?
7. Does it depend on a known candidate label?
8. Does it depend on a predetermined step?
9. Can the model learn this instead?
10. How will the change be physically tested?
11. What baseline will it be compared against?
12. What evidence would falsify the proposed improvement?

If the change introduces task intelligence into runtime code,
STOP and reconsider.


# ============================================================
# 44. BEFORE TRAINING
# ============================================================

Before launching a significant training run:

- verify dataset hash
- verify checkpoint parent
- verify parameter transfer
- verify trainable tensors
- verify oracle audit
- verify candidate-target alignment
- verify masks
- verify loss finiteness
- verify tiny-set overfit
- verify AMP/FP32 consistency
- verify checkpoint save/load
- verify validation split
- verify no benchmark leakage
- verify no accidental future information

Never launch an expensive run when the objective has not been validated
on a small controlled batch.


# ============================================================
# 45. BEFORE PHYSICAL EVALUATION
# ============================================================

Verify:

- correct APK
- correct model
- correct checkpoint
- correct decoder
- correct runtime
- correct device
- clean task state
- no hidden ADB action
- no stale logs
- no evaluator collision
- no hidden oracle
- timestamp-scoped logs
- reproducible task configuration

If any of these are uncertain, classify the run as diagnostic rather than
authoritative.


# ============================================================
# 46. WHEN A MODEL FAILS
# ============================================================

The coding agent MUST NOT panic and redesign the entire architecture.

First ask:

    Is this a runtime bug?
    Is this a data problem?
    Is this a grounding problem?
    Is this an action-type problem?
    Is this an argument problem?
    Is this a task-state problem?
    Is this a recovery problem?
    Is this a termination problem?
    Is this distribution shift?
    Is this genuinely an architecture limitation?

Then make the smallest evidence-driven change.

Preserve:
- failure trace
- checkpoint
- logs
- screenshots
- observation dump
- candidate list
- model outputs

Failures are valuable training evidence.


# ============================================================
# 47. DO NOT OVERFIT TO ONE FAILURE
# ============================================================

A single failure must not immediately produce:

- app-specific code
- label-specific rules
- candidate-index rules
- package-specific branches
- score hacks

If LinkedIn fails because a feed card distracts the model:

GOOD:

    add diverse hard-negative grounding examples

BAD:

    if LinkedIn:
        suppress feed cards

If Search fails:

GOOD:

    improve learned candidate grounding

BAD:

    if goal contains "search":
        boost search


# ============================================================
# 48. MATHEMATICAL / SYMBOLIC TASKS
# ============================================================

Karvish is primarily a UI action-intelligence system.

For:

    "Calculate 85 * 6"

the action model should learn to:

- identify keypad candidates
- select correct buttons
- execute the sequence
- observe the result
- verify the displayed result

A deterministic symbolic calculator may be used as an explicitly declared
specialist/tool when appropriate.

However:

- do not hardcode Calculator coordinates
- do not hardcode Calculator button sequences
- do not type operands through ADB
- do not declare success without verifying the requested result

If a specialist computes the symbolic answer, report that separately from
the UI policy's learned interaction capability.


# ============================================================
# 49. TEMPORARY RESEARCH SCAFFOLDING
# ============================================================

Temporary heuristics may be introduced ONLY when they are required to:

- diagnose a capability
- establish a control
- collect data
- validate infrastructure
- perform an ablation
- isolate a failure

Every temporary heuristic MUST have:

- explicit name
- explicit location
- explicit purpose
- explicit experiment
- explicit removal condition

Temporary scaffolding MUST NOT be silently treated as production
intelligence.

The default direction of the project is:

    heuristic
       ->
    measure
       ->
    collect failures
       ->
    train replacement
       ->
    remove heuristic


# ============================================================
# 50. CURRENT ROADMAP
# ============================================================

Current order:

    V51.7-GOLD
        |
        v
    Phase A
    Search/feed heuristic removal
        |
        v
    LinkedIn hard-negative training
        |
        v
    Physical requalification
        |
        v
    Phase B
    Remove input-field penalty
        |
        v
    Physical requalification
        |
        v
    Phase C
    Remove submit boost
        |
        v
    Physical requalification
        |
        v
    Phase D
    Learned argument/span pointer
        |
        v
    Remove regex entity extraction
        |
        v
    V52-LOCKED-GOLD (Frozen Universal Baseline - September 30, 2026)
        |
        v
    V53 Next-Gen Architecture & Training:
    - 2D Spatial Layout Geometry Tokens (LayoutLM-style)
    - Goal-Intent Caching (O(1) Turn-0 Vector)
    - Large-Margin InfoNCE Contrastive Ranking Loss
    - Pure FP16 & W8A16 QAT Engine via XNNPACK
        |
        v
    5-Gate Micro-Adaptation Qualification (Anchor Integrity + Novel Apps)
        |
        v
    Native Physical Device Qualification (Samsung Galaxy A26)


# ============================================================
# 51. PHASE COMPLETION RULE
# ============================================================

A phase is NOT complete because:

- code exists
- training finished
- offline metrics improved
- a benchmark improved
- an agent claims success

A phase is complete only when its documented gate has been satisfied.

If evidence is incomplete:

    STATUS = INCOMPLETE

Do not convert incomplete evidence into PASS.


# ============================================================
# 52. GOLDEN BASELINE POLICY
# ============================================================

V51.7-GOLD exists permanently.

When a new model performs worse:

    preserve it
    diagnose it
    compare it
    do not overwrite GOLD

When a new model performs better:

    document why
    reproduce it
    physically validate it
    freeze it as a new control only after qualification

Never retroactively alter old benchmark records.


# ============================================================
# 53. FINAL NORTH STAR
# ============================================================

The final Karvish experience should eventually be:

    USER:
    "Do this on my phone."

    KARVISH:

    Understand
       ->
    Observe
       ->
    Decide
       ->
    Act
       ->
    Observe
       ->
    Update task state
       ->
    Verify
       ->
    Recover if necessary
       ->
    Continue
       ->
    Verify completion

without requiring a developer to manually program the workflow for the
specific application or task.

The developer builds:

    general capabilities

NOT:

    individual task solutions.


# ============================================================
# 54. FINAL RULE
# ============================================================

WHEN A SHORTCUT AND A GENERAL LEARNED SOLUTION COMPETE:

    PREFER THE GENERAL LEARNED SOLUTION.

WHEN A DETERMINISTIC PLATFORM SAFETY MECHANISM AND A LEARNED SOLUTION
COMPETE:

    KEEP THE SAFETY MECHANISM.

WHEN A MODEL FAILURE OCCURS:

    LEARN FROM THE FAILURE BEFORE PATCHING THE RUNTIME.

WHEN A BENCHMARK IMPROVES:

    VERIFY THAT THE IMPROVEMENT IS REAL.

WHEN EVIDENCE IS INSUFFICIENT:

    SAY SO.

WHEN A CLAIM CANNOT BE PROVEN:

    DO NOT MAKE THE CLAIM.

KARVISH MUST BECOME MORE INTELLIGENT THROUGH LEARNING,
NOT MORE IMPRESSIVE THROUGH HIDDEN RULES.