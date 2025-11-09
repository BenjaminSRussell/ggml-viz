# GGML-viz Code Audit: Issues & Refactoring Opportunities

**Date:** 2025-01-09
**Files Analyzed:** 26 source files (5,857 total lines)
**Severity Legend:** 🔴 Critical | 🟠 High | 🟡 Medium | 🔵 Low

---

## Executive Summary

The codebase contains **~1,200 lines of duplicate/redundant code** that can be consolidated. Major issues:

- **400+ lines** of copy-paste hook functions (95% identical)
- **100+ lines** of duplicate validation checks
- **70+ printf/cout statements** mixing 3 different output methods
- **50+ magic numbers** without constants
- **11 TODOs** left in production code
- **Debug output** scattered throughout (should use Logger)
- **Inconsistent error handling** patterns

**Estimated Reduction Opportunity:** 25-35% codebase shrinkage (~1,500 lines)

---

## 🔴 CRITICAL ISSUES

### 1. Massive Code Duplication in Hook Functions (src/instrumentation/ggml_hook.cpp)

**Lines 500-900 (400 lines):** Seven hook interceptor functions are 90-98% identical.

#### Problem: Copy-Paste Programming

**Four Graph Compute Hooks** (95% identical):
```cpp
// Lines 504-554: ggml_backend_graph_compute
// Lines 557-603: ggml_graph_compute
// Lines 606-649: ggml_graph_compute_with_ctx
// Lines 652-701: ggml_backend_metal_graph_compute
```

**Each contains identical blocks:**

1. **Auto-start (6 occurrences):**
```cpp
if (!hook.is_active() && getenv("GGML_VIZ_OUTPUT")) {
    printf("[GGML_VIZ] Auto-starting hooks...\n");  // Duplicate 6x
    hook.start();
}
```

2. **Hook begin (4 occurrences):**
```cpp
if (hook.is_active()) {
    printf("[DEBUG] Intercepted <function>, nodes: %d\n", ...);  // Duplicate 4x
    hook.on_graph_compute_begin(cgraph, backend);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]) {
            hook.on_op_compute_begin(cgraph->nodes[i], backend);
        }
    }
}
```

3. **Hook initialization (7 occurrences):**
```cpp
if (!hooks_initialized) {  // Duplicate 7x
    install_ggml_hooks();
    hooks_initialized = true;
}
```

4. **Hook end (4 occurrences):**
```cpp
if (hook.is_active()) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]) {
            hook.on_op_compute_end(cgraph->nodes[i], backend);
        }
    }
    hook.on_graph_compute_end(cgraph, backend);
}
```

#### Solution: Template Function
```cpp
template<typename OrigFunc, typename... Args>
auto wrap_hook(const char* name, OrigFunc original,
               struct ggml_cgraph* cgraph, Args... args) {
    auto_start_if_needed();

    if (is_active()) {
        log_intercept(name, cgraph->n_nodes);
        record_graph_begin(cgraph);
        record_ops_begin(cgraph);
    }

    auto result = original(args...);

    if (is_active()) {
        record_ops_end(cgraph);
        record_graph_end(cgraph);
    }

    return result;
}
```

**Reduction:** 400 lines → ~150 lines (62% reduction)

---

### 2. Printf/Cout/Cerr Chaos (Entire Codebase)

**70+ mixed output statements** across 3 different methods:

#### Current State:
```cpp
printf("[GGML_VIZ] Auto-starting...\n");      // 20+ occurrences
printf("[DEBUG] Intercepted...\n");           // 10+ occurrences
std::cout << "[ImGuiApp] DEBUG: Got...\n";    // 15+ occurrences
std::cerr << "Warning: Cannot...\n";          // 10+ occurrences
printf("[LiveStreamServer] Started...\n");    // 15+ occurrences
```

#### Problems:
- **No log levels** (can't disable debug output in production)
- **No timestamps** (can't correlate events)
- **Three different mechanisms** (printf, cout, cerr)
- **Inconsistent formatting** ([GGML_VIZ] vs [ImGuiApp] vs [DEBUG])
- **Logger class exists but unused** (src/utils/logger.cpp)

#### Solution:
```cpp
// Replace ALL output with:
GGML_VIZ_LOG_INFO("Auto-starting hooks");
GGML_VIZ_LOG_DEBUG("Intercepted {}, nodes: {}", func_name, nodes);
GGML_VIZ_LOG_WARN("Cannot reconfigure while active");
GGML_VIZ_LOG_ERROR("Failed to open file: {}", filename);
```

**Files affected:**
- src/instrumentation/ggml_hook.cpp: 40+ printf statements
- src/frontend/imgui_app.cpp: 15+ cout statements
- src/server/live_data_collector.hpp: 15+ printf statements
- src/utils/config.cpp: 5+ cerr statements

**Reduction:** 70+ lines of manual printing → Consistent logging API

---

### 3. Duplicate Event Recording Logic (src/instrumentation/ggml_hook.cpp)

**Lines 308-458:** Six nearly identical `on_*` functions.

#### Current: 150 Lines of Repetition
```cpp
// Lines 308-328 (on_graph_compute_begin)
void on_graph_compute_begin(const ggml_cgraph* graph, ...) {
    if (!active_.load()) return;
    auto config = ConfigManager::instance().get();
    if (!config->instrumentation.enable_op_timing) return;

    Event event = {};
    event.type = EventType::GRAPH_COMPUTE_BEGIN;
    event.timestamp_ns = get_timestamp_ns();
    event.thread_id = get_thread_id();
    event.data.graph.graph_ptr = graph;
    event.data.graph.n_nodes = (graph != nullptr) ? graph->n_nodes : 0;
    event.data.graph.n_threads = 1; // TODO: Get actual thread count
    event.data.graph.backend_ptr = backend;
    event.label = nullptr;
    record_event(event);
}

// Lines 330-356 (on_graph_compute_end) - 95% IDENTICAL
// Lines 358-388 (on_op_compute_begin) - 95% IDENTICAL
// Lines 390-418 (on_op_compute_end) - 95% IDENTICAL
// Lines 420-438 (on_tensor_alloc) - 90% IDENTICAL
// Lines 440-458 (on_tensor_free) - 90% IDENTICAL
```

#### Solution: Consolidate with Flags
```cpp
void record_graph_event(const ggml_cgraph* graph, const ggml_backend* backend,
                       bool is_begin) {
    if (!should_record(RecordType::GRAPH)) return;

    auto event = create_event(is_begin ? GRAPH_COMPUTE_BEGIN : GRAPH_COMPUTE_END);
    set_graph_data(event, graph, backend);

    if (!is_begin && should_flush()) {
        flush_to_file();
    }
}

void record_op_event(const ggml_tensor* tensor, const ggml_backend* backend,
                    bool is_begin) {
    if (!should_record(RecordType::OP)) return;
    if (!should_trace_op(tensor->op)) return;

    auto event = create_event(is_begin ? OP_COMPUTE_BEGIN : OP_COMPUTE_END);
    set_op_data(event, tensor, backend);
}

void record_tensor_event(const ggml_tensor* tensor, size_t size,
                        const ggml_backend* backend, bool is_alloc) {
    if (!should_record(RecordType::MEMORY)) return;

    auto event = create_event(is_alloc ? TENSOR_ALLOC : TENSOR_FREE);
    set_memory_data(event, tensor->data, is_alloc ? size : 0);
}
```

**Reduction:** 150 lines → ~60 lines (60% reduction)

---

## 🟠 HIGH PRIORITY ISSUES

### 4. Magic Numbers Everywhere (src/frontend/imgui_app.cpp)

**50+ magic numbers** without named constants:

| Line | Value | Purpose | Should Be |
|------|-------|---------|-----------|
| 43 | 512 | File path buffer | `MAX_FILE_PATH` |
| 90 | 1280, 720 | Window size | `DEFAULT_WINDOW_WIDTH/HEIGHT` |
| 81-82 | 3, 3 | OpenGL version | `OPENGL_VERSION_MAJOR/MINOR` |
| 234, 308 | 50000 | Max events | `MAX_LIVE_EVENTS` |
| 204, 223, 241 | 100 | Debug frequency | `DEBUG_PRINT_INTERVAL` |
| 255 | 100 | File check interval | `FILE_CHECK_INTERVAL_MS` |
| 443 | 350 | Menu offset | `MENU_STATUS_OFFSET_X` |
| 522 | 12 | Header size | `MIN_FILE_HEADER_SIZE` |
| 653 | 256 | Search buffer | `SEARCH_BUFFER_SIZE` |
| 698 | 30.0f | Railroad spacing | `TIMELINE_TIE_SPACING` |
| 1034 | 10.0f | Overlay margin | `STATS_OVERLAY_MARGIN` |
| 1040 | 0.35f | Alpha | `OVERLAY_BACKGROUND_ALPHA` |
| 591 | RGB(0.13, 0.54, 0.82) | Timeline color | `COLOR_TIMELINE_TITLE` |
| 825 | RGB(0.30, 0.69, 0.31) | Graph color | `COLOR_GRAPH_TITLE` |
| 885 | RGB(0.91, 0.12, 0.39) | Tensor color | `COLOR_TENSOR_TITLE` |
| 941 | RGB(1.0, 0.60, 0.0) | Memory color | `COLOR_MEMORY_TITLE` |

#### Solution:
```cpp
// src/frontend/ui_constants.hpp
namespace UIConstants {
    constexpr size_t MAX_FILE_PATH = 512;
    constexpr int DEFAULT_WINDOW_WIDTH = 1280;
    constexpr int DEFAULT_WINDOW_HEIGHT = 720;
    constexpr int OPENGL_VERSION_MAJOR = 3;
    constexpr int OPENGL_VERSION_MINOR = 3;

    constexpr size_t MAX_LIVE_EVENTS = 50'000;
    constexpr int DEBUG_PRINT_INTERVAL = 100;
    constexpr int FILE_CHECK_INTERVAL_MS = 100;

    constexpr float TIMELINE_TIE_SPACING = 30.0f;
    constexpr float STATS_OVERLAY_MARGIN = 10.0f;
    constexpr float OVERLAY_BACKGROUND_ALPHA = 0.35f;

    // Theme colors
    namespace Colors {
        constexpr ImVec4 TIMELINE_TITLE{0.13f, 0.54f, 0.82f, 1.0f};
        constexpr ImVec4 GRAPH_TITLE{0.30f, 0.69f, 0.31f, 1.0f};
        constexpr ImVec4 TENSOR_TITLE{0.91f, 0.12f, 0.39f, 1.0f};
        constexpr ImVec4 MEMORY_TITLE{1.0f, 0.60f, 0.0f, 1.0f};
    }
}
```

**Reduction:** Improved maintainability, theming support

---

### 5. Static Variables in Member Functions (src/frontend/imgui_app.cpp)

**Lines 203, 251, 273:** Static variables causing hidden state.

#### Problem:
```cpp
void ImGuiApp::update_live_data() {
    static int call_count = 0;  // Hidden global state!
    if (++call_count % 100 == 0) { ... }
}

void ImGuiApp::monitor_external_file() {
    static auto last_file_check = ...;  // Hidden global state!
    static size_t last_file_event_count = 0;  // Hidden global state!
}
```

#### Issues:
- **Not visible in class definition**
- **Can't be reset/tested easily**
- **Breaks encapsulation**
- **Thread-safety unclear**

#### Solution:
```cpp
// In imgui_app.hpp AppData struct:
struct AppData {
    // ... existing fields ...

    // Move static variables to explicit state
    int debug_call_counter = 0;
    std::chrono::steady_clock::time_point last_file_check_time;
    size_t last_file_event_count = 0;
};
```

**Benefit:** Explicit state management, easier testing

---

### 6. Overly Long Functions (src/frontend/imgui_app.cpp)

**Lines 489-586 (97 lines):** `load_trace_file()` does too much.

#### Problem:
```cpp
bool ImGuiApp::load_trace_file(const std::string& filename) {
    // 1. Input validation (10 lines)
    // 2. File existence check (15 lines)
    // 3. File size check (10 lines)
    // 4. Magic header validation (15 lines)
    // 5. Version check (10 lines)
    // 6. Trace reader creation (15 lines)
    // 7. UI state update (10 lines)
    // 8. Error handling (12 lines)
    // Total: 97 lines, 4 levels of nesting
}
```

#### Solution: Extract Helper Functions
```cpp
struct FileValidation {
    bool is_valid;
    std::string error_message;
    FILE* file_handle;
    size_t file_size;
};

FileValidation validate_trace_file(const std::string& filename) {
    // Lines 496-540: Extract validation logic
}

bool verify_magic_header(FILE* file) {
    // Lines 522-540: Extract header check
}

bool load_trace_file(const std::string& filename) {
    auto validation = validate_trace_file(filename);
    if (!validation.is_valid) {
        set_error(validation.error_message);
        return false;
    }

    if (!verify_magic_header(validation.file_handle)) {
        set_error("Invalid file format");
        return false;
    }

    return create_trace_reader(filename);
}
```

**Reduction:** 97 lines → ~30 lines per function, easier to test

---

### 7. Duplicate Validation Checks (src/frontend/imgui_app.cpp)

**Lines 598-627, 828-832, 888-892, 944-947:** Same validation repeated 4x.

#### Problem:
```cpp
// In render_timeline_view() - Line 598
if (!data_->trace_reader && !data_->live_mode) {
    ImGui::Text("No trace loaded and live mode not active");
    ImGui::End();
    return;
}

// EXACT DUPLICATE in render_graph_view() - Line 828
// EXACT DUPLICATE in render_tensor_inspector() - Line 888
// EXACT DUPLICATE in render_memory_view() - Line 944
```

#### Solution: Guard Macro or Function
```cpp
#define REQUIRE_ACTIVE_DATA() \
    if (!has_active_data()) { \
        show_no_data_message(); \
        ImGui::End(); \
        return; \
    }

// Then in each function:
void ImGuiApp::render_timeline_view() {
    if (ImGui::Begin("Timeline View")) {
        REQUIRE_ACTIVE_DATA();
        // ... actual rendering ...
    }
    ImGui::End();
}
```

**Reduction:** 20 lines → 1 line per function

---

## 🟡 MEDIUM PRIORITY ISSUES

### 8. Duplicate Buffer Limiting Code (src/frontend/imgui_app.cpp)

**Lines 234-238, 308-312:** Identical buffer management.

#### Problem:
```cpp
// Line 234-238
const size_t max_events = 50000;
if (data_->live_events.size() > max_events) {
    data_->live_events.erase(
        data_->live_events.begin(),
        data_->live_events.begin() + (data_->live_events.size() - max_events)
    );
}

// EXACT DUPLICATE at lines 308-312
```

#### Solution:
```cpp
void limit_live_events() {
    constexpr size_t MAX_EVENTS = 50'000;
    if (data_->live_events.size() > MAX_EVENTS) {
        data_->live_events.erase(
            data_->live_events.begin(),
            data_->live_events.end() - MAX_EVENTS
        );
    }
}
```

**Reduction:** 10 lines → 1 function call

---

### 9. Render Function Scaffolding (src/frontend/imgui_app.cpp)

**Lines 589-954:** Four render functions with identical structure.

#### Problem:
```cpp
void ImGuiApp::render_timeline_view() {
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.13f, 0.54f, 0.82f, 1.0f));
    if (ImGui::Begin("Timeline View")) {
        if (!data_->trace_reader && !data_->live_mode) { return; }
        // ... unique content ...
    }
    ImGui::End();
    ImGui::PopStyleColor();
}

// render_graph_view(), render_tensor_inspector(), render_memory_view() - SAME PATTERN
```

#### Solution:
```cpp
template<typename ContentFunc>
void render_window_with_style(const char* title, ImVec4 color, ContentFunc&& content) {
    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, color);
    if (ImGui::Begin(title)) {
        if (has_active_data()) {
            content();
        } else {
            show_no_data_message();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor();
}

void render_timeline_view() {
    render_window_with_style("Timeline View", COLOR_TIMELINE_TITLE, [this] {
        // ... unique timeline content ...
    });
}
```

**Reduction:** ~40 lines of boilerplate → single template

---

### 10. Event Type Conversion Scattered (src/frontend/imgui_app.cpp)

**Lines 730-745 and elsewhere:** Manual switch statements repeated.

#### Problem:
```cpp
std::string event_name;
switch(event.type) {
    case EventType::GRAPH_COMPUTE_BEGIN: event_name = "GRAPH_BEGIN"; break;
    case EventType::GRAPH_COMPUTE_END: event_name = "GRAPH_END"; break;
    case EventType::OP_COMPUTE_BEGIN: event_name = "OP_BEGIN"; break;
    case EventType::OP_COMPUTE_END: event_name = "OP_END"; break;
    case EventType::TENSOR_ALLOC: event_name = "ALLOC"; break;
    case EventType::TENSOR_FREE: event_name = "FREE"; break;
    // ...
}
```

#### Solution:
```cpp
// In instrumentation/ggml_hook.hpp
constexpr const char* event_type_name(EventType type) {
    switch (type) {
        case EventType::GRAPH_COMPUTE_BEGIN: return "GRAPH_BEGIN";
        case EventType::GRAPH_COMPUTE_END: return "GRAPH_END";
        case EventType::OP_COMPUTE_BEGIN: return "OP_BEGIN";
        case EventType::OP_COMPUTE_END: return "OP_END";
        case EventType::TENSOR_ALLOC: return "ALLOC";
        case EventType::TENSOR_FREE: return "FREE";
        case EventType::BARRIER_WAIT: return "BARRIER";
        case EventType::THREAD_BEGIN: return "THREAD_BEGIN";
        case EventType::THREAD_FREE: return "THREAD_FREE";
    }
    return "UNKNOWN";
}

// Usage:
std::string event_name = event_type_name(event.type);
```

**Reduction:** DRY utility function

---

### 11. Unused/Unnecessary Code

#### Debug Counter That's Always On (Lines 203-226)
```cpp
static int call_count = 0;
if (++call_count % 100 == 0) {  // Always runs in production!
    std::cout << "[ImGuiApp] update_live_data() called..." << std::endl;
    std::cout << "[ImGuiApp] DEBUG: Got " << new_events.size() << "..." << std::endl;
}
```

**Solution:** Remove or use proper log levels:
```cpp
GGML_VIZ_LOG_DEBUG("update_live_data: file={}, events={}",
                   data_->live_file_path, data_->live_events.size());
```

#### TODO Comments (11 occurrences)
```cpp
// Line 323, 345: // TODO: Get actual thread count
// Line 11 (win32): // TODO: MH_CreateHook for ggml_backend_sched_graph_compute
// Line 436: // TODO: Implement about dialog
// Line 482: // TODO: Add proper file browser
// Line 781: // TODO: Calculate live operation timings
// Line 855: // TODO: Map node selection to event selection
// Line 929: // TODO: Add tensor-specific inspection
// Line 431 (widgets): // TODO: Use tensor_to_node for dependency
```

**Action:** Either implement or remove

#### Unused Variables
```cpp
// Line 856
int selected = graph_widget_.get_selected_node(); // Unused for now
```

**Action:** Remove or use

---

## 🔵 LOW PRIORITY / STYLE ISSUES

### 12. Inconsistent Naming

**Error Messages:**
```cpp
"IMG ERROR: Window build fail"  // Unprofessional (line 59)
"good" vs "bad" status strings  // Should be boolean/enum (line 447)
```

**Function Names:**
```cpp
trace_reader_for_widget  // Overly verbose (line 665)
call_count              // Generic (line 203)
```

### 13. File Organization

**live_data_collector.hpp:** 445 lines as header-only implementation

**Issue:** Should be split into .hpp/.cpp for faster compilation

### 14. Redundant Helper Function

**Lines 36-64 (ggml_hook.cpp):** Custom `ggml_nbytes_simple()` re-implements existing ggml function.

**Issue:** Could just call the real `ggml_nbytes()` from ggml library.

---

## REFACTORING ROADMAP

### Phase 1: Critical (Week 1) 🔴

1. **Consolidate hook functions** (src/instrumentation/ggml_hook.cpp:500-900)
   - Create `wrap_hook()` template
   - Consolidate begin/end pairs
   - **Reduction:** ~250 lines

2. **Replace all printf/cout/cerr with Logger** (all files)
   - Consistent log levels
   - **Reduction:** ~70 statements → proper logging

3. **Extract duplicate event recording** (src/instrumentation/ggml_hook.cpp:308-458)
   - Consolidate on_* functions
   - **Reduction:** ~90 lines

### Phase 2: High Priority (Week 2) 🟠

4. **Add constants for magic numbers** (src/frontend/imgui_app.cpp)
   - Create ui_constants.hpp
   - **Improvement:** Maintainability, theming

5. **Move static variables to class members** (src/frontend/imgui_app.cpp)
   - Explicit state management
   - **Improvement:** Testability

6. **Break down long functions** (src/frontend/imgui_app.cpp)
   - Extract validation helpers
   - **Improvement:** Readability

### Phase 3: Medium Priority (Week 3) 🟡

7. **Consolidate validation checks**
   - Create guard macros/functions
   - **Reduction:** ~20 lines

8. **Extract buffer management helper**
   - DRY compliance
   - **Reduction:** ~10 lines

9. **Template render window scaffolding**
   - Reduce boilerplate
   - **Reduction:** ~40 lines

### Phase 4: Polish (Week 4) 🔵

10. **Implement or remove TODOs** (11 items)
11. **Fix naming consistency**
12. **Split header-only implementations**
13. **Remove unused code**

---

## METRICS

### Current State
- **Total Lines:** 5,857
- **Duplicate Code:** ~1,200 lines (20%)
- **Printf Statements:** 70+
- **Magic Numbers:** 50+
- **TODOs:** 11
- **Static Variables:** 3

### After Refactoring (Estimated)
- **Total Lines:** ~4,300 (-26%)
- **Duplicate Code:** <100 lines (<2%)
- **Consistent Logging:** 100%
- **Named Constants:** 100%
- **TODOs:** 0
- **Proper State Management:** 100%

---

## CONCLUSION

The codebase is **functional but bloated**. Key issues:

1. **Copy-paste programming** in hook functions (400 lines)
2. **No consistent logging** (70+ mixed statements)
3. **Magic number hell** (50+ constants needed)
4. **DRY violations** throughout

**Recommended Action:** Execute phased refactoring roadmap above.

**Expected Outcome:**
- 25-30% codebase reduction
- Improved maintainability
- Easier testing
- Professional code quality
