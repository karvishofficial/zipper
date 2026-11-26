# KARVISH / DITTO — ACTION PLAN SHEET
**Date:** October 6, 2026  
**Target Device:** Samsung Galaxy A26 (`RZCY51MQP9K`)  
**Core Baseline:** Karvish Universal V57 (`karvish_v57.onnx`, 28.88M parameters)

---

## 📋 Executive Summary
Today we resolved the fundamental routing between Instagram Search (`Search and explore` tab) and Messaging (`Message` inbox), and confirmed our search-first contact resolution on live devices. 

Tomorrow's work targets two high-impact refinements:
1. **Network Settle Check for Instagram Disambiguation**: Ensuring transient `'Searching…'` network delays never cause the Disambiguation Card to be skipped.
2. **Ditto Chrome Web Tree Hit-Testing**: Ensuring web navigation elements (like hamburger menu icons on `plutoailabs.com`) are accurately captured during Ditto demonstration recording.

---

## 🎯 Task 1: Instagram Search Settle & Disambiguation Card Stability

### Background:
In today's test for *"open Instagram and send hi to Manoj"*, after typing `'Manoj'` into the DM search bar, Instagram displayed a transient loading state:
```text
Text changed: row_search_for_x_textview: 'Recent' -> 'Searching…'
Accessibility Node Count: 7 nodes
```
Because the turn evaluated immediately while Instagram was still fetching search results from the server, `distinctContacts` was temporarily empty for ~300ms, causing the Disambiguation Card to not trigger before the user cancelled.

### Proposed Code Changes in `AgentRuntime.kt`:
- **File**: [AgentRuntime.kt](file:///c:/Users/Karthik%20S/AndroidStudioProjects/zipper/app/src/main/java/com/zipper/app/core/agent/AgentRuntime.kt)
- **Implementation**:
  ```kotlin
  // If search was just typed and the screen shows a transient search spinner/text,
  // settle for up to 600ms until network contacts populate:
  if (runtimeTaskState.hasExecutedSearch && distinctContacts.isEmpty()) {
      val isSearchingTransient = currentState.nodes.any { n ->
          val txt = (n.text ?: n.contentDescription).orEmpty().lowercase()
          txt.contains("searching") || txt.contains("loading")
      } || currentState.nodes.size < 10

      if (isSearchingTransient) {
          Log.i(TAG, "[DISAMBIGUATION] Transient search loading detected ('Searching…'). Settling for network results...")
          delay(500)
          // Re-capture live candidates after results populate
      }
  }
  ```
- **Validation**:
  - Test *"open Instagram and send hi to Manoj"* on phone.
  - Verify that all 12 contacts pop up in the numbered Disambiguation Card every single run, regardless of network lag.

---

## 🎯 Task 2: Ditto Chrome Web Tree Hit-Testing & Menu Capture

### Background:
During demonstration recording on `plutoailabs.com`, the hamburger menu (☰) tap was dropped because:
1. **Header Debounce**: [DittoRecorder.kt](file:///c:/Users/Karthik%20S/AndroidStudioProjects/zipper/app/src/main/java/com/zipper/app/ditto/DittoRecorder.kt#L266) had an aggressive `(rect.top < 450 && lastTapRect.top < 450)` check that treated top-bar taps as duplicate clicks.
2. **Chromium Virtual Node Accessibility**: Modern web hamburger menus are often HTML `<div>` or `<svg>` icons with JavaScript touch listeners. Chromium handles them internally and frequently does not emit an Android OS `AccessibilityEvent.TYPE_VIEW_CLICKED`.

### Proposed Code Changes in `DittoRecorder.kt`:
- **File**: [DittoRecorder.kt](file:///c:/Users/Karthik%20S/AndroidStudioProjects/zipper/app/src/main/java/com/zipper/app/ditto/DittoRecorder.kt)
- **Implementation**:
  1. **Refine Header Debounce**: Replace blanket `rect.top < 450` with physical Euclidean distance check:
     ```kotlin
     // Only debounce if the two clicks are within 60px of each other and under 300ms
     val isCoincident = !rect.isEmpty && !lastTapRect.isEmpty &&
         hypot((rect.centerX() - lastTapRect.centerX()).toFloat(), 
               (rect.centerY() - lastTapRect.centerY()).toFloat()) < 60f
     ```
  2. **Active Hit-Testing on Chrome's Virtual Tree**:
     When recording in `com.android.chrome` or WebViews:
     - Query `service.rootInActiveWindow` at the tap's physical `(touchX, touchY)` coordinates.
     - Extract the exact web button, SVG, or navigation container from Chrome's virtual DOM.
     - Record the step with its web role (`BUTTON` / `NAVIGATION`) and calibrated bounding box.

---

## 🎯 Task 3: End-to-End Validation of `plutoailabs.com` Flow

### Target User Scenario:
1. **Record**:
   - Start Ditto: *"Google Search: Search plutoailabs.com and open portfolio"*.
   - Open Google ➔ Type `plutoailabs.com` ➔ Open site ➔ Tap Hamburger Menu (☰) ➔ Tap "Portfolio".
2. **Verify Recorded Steps**:
   - Step 1: Launcher tap (Google).
   - Step 2: Type `plutoailabs.com`.
   - Step 3: Click search result link / open Chrome.
   - Step 4: **TAP on Navigation Menu (☰)** (Confirmed captured!).
   - Step 5: **TAP on Portfolio**.
3. **Replay Execution**:
   - Run deterministic replay via `DittoPlayer`.
   - Verify menu opens smoothly on device and navigates to the Portfolio page with 100% success.

---

## 🎯 Task 4: WhatsApp Full Communication Regression Check

### Tasks to verify:
- *"open WhatsApp and send hi to Ajay"* ➔ Verifies disambiguation between `Ajay` and `Ajay sir`.
- *"open WhatsApp and send hi to Shobha"* ➔ Verifies single partial match confirmation (`Shobha Aunty`).
- *"open WhatsApp and send hi to Manoj"* ➔ Verifies search-first resolution and final message send.

---

## 🛠️ Verification Checklist for Tomorrow:
- [ ] Implement network settle wait in `AgentRuntime.kt`.
- [ ] Fix `DittoRecorder.kt` header debounce & add Chrome tree hit-testing.
- [ ] `./gradlew assembleDebug` (compile check).
- [ ] `adb install -r app-debug.apk` onto `RZCY51MQP9K`.
- [ ] Live test Instagram Manoj search with network lag.
- [ ] Live test Ditto recording on `plutoailabs.com` menu navigation.
