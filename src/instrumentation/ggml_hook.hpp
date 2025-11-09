#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <functional>
#include <atomic>
#include <chrono>
#include <mutex>

struct ggml_tensor;
struct ggml_cgraph;
struct ggml_compute_params;
struct ggml_backend;

#ifndef GGML_VIZ_TEST_MODE
  #ifdef _WIN32
    #define GGML_VIZ_API __declspec(dllexport)
  #else
    #define GGML_VIZ_API __attribute__((visibility("default")))
  #endif
#else
  #define GGML_VIZ_API
#endif

namespace ggml_viz {

enum class EventType : uint8_t {
    GRAPH_COMPUTE_BEGIN,
    GRAPH_COMPUTE_END,
    OP_COMPUTE_BEGIN,
    OP_COMPUTE_END,
    TENSOR_ALLOC,
    TENSOR_FREE,
    BARRIER_WAIT,
    THREAD_BEGIN,
    THREAD_FREE
};

constexpr const char* event_type_name(EventType type) {
    switch (type) {
        case EventType::GRAPH_COMPUTE_BEGIN: return "GRAPH_BEGIN";
        case EventType::GRAPH_COMPUTE_END:   return "GRAPH_END";
        case EventType::OP_COMPUTE_BEGIN:    return "OP_BEGIN";
        case EventType::OP_COMPUTE_END:      return "OP_END";
        case EventType::TENSOR_ALLOC:        return "TENSOR_ALLOC";
        case EventType::TENSOR_FREE:         return "TENSOR_FREE";
        case EventType::BARRIER_WAIT:        return "BARRIER_WAIT";
        case EventType::THREAD_BEGIN:        return "THREAD_BEGIN";
        case EventType::THREAD_FREE:         return "THREAD_FREE";
        default:                             return "UNKNOWN";
    }
}

struct Event {
    EventType type;
    uint64_t timestamp_ns;
    uint32_t thread_id;

    union {
        struct {
            const void* tensor_ptr;
            uint32_t op_type;
            size_t op_size;
            const void* backend_ptr;
        } op;

        struct {
            const void* graph_ptr;
            uint32_t n_nodes;
            uint32_t n_threads;
            const void* backend_ptr;
        } graph;

        struct {
            const void* ptr;
            size_t size;
        } memory;
    } data;

    const char* label;
};

struct HookConfig {
    bool enable_op_timing = true;
    bool enable_memory_tracking = false;
    bool enable_thread_tracking = false;
    bool enable_tensor_names = true;

    bool write_to_file = true;
    std::string output_filename = "ggml_trace.bin";

    std::vector<uint32_t> op_types_to_trace;
    size_t max_events = 1000000;
};

class GGMLHook {
public:
    static GGMLHook& instance();

    void configure(const HookConfig& config);

    void start();
    void stop();
    bool is_active() const { return active_.load(); }

    size_t event_count() const { return event_count_.load(); }
    void reset_stats();
    std::vector<Event> get_events_size(uint64_t timestamp_ns);
    std::vector<Event> consume_available_events();
    Event* get_ring_buffer() { return event_buffer_; }
    size_t get_buffer_size() const { return BUFFER_SIZE; }
    size_t get_current_write_pos() const { return write_pos_.v.load(std::memory_order_acquire); }
    size_t get_current_read_pos() const { return read_pos_.v.load(std::memory_order_acquire); }
    size_t get_dropped_events() const { return dropped_events_.load(std::memory_order_relaxed); }

    void on_graph_compute_begin(const ggml_cgraph* graph, const ggml_backend* backend = nullptr);
    void on_graph_compute_end(const ggml_cgraph* graph, const ggml_backend* backend = nullptr);
    void on_op_compute_begin(const ggml_tensor* tensor, const ggml_backend* backend = nullptr);
    void on_op_compute_end(const ggml_tensor* tensor, const ggml_backend* backend = nullptr);
    void on_tensor_alloc(const ggml_tensor* tensor, size_t size, const ggml_backend* backend = nullptr);
    void on_tensor_free(const ggml_tensor* tensor, const ggml_backend* backend = nullptr);

    ~GGMLHook();

private:
    GGMLHook();
    GGMLHook(const GGMLHook&) = delete;
    GGMLHook& operator=(const GGMLHook&) = delete;

    void record_event(const Event& event);
    void flush_to_file();

    std::atomic<bool> active_{false};
    std::atomic<size_t> event_count_{0};

    struct alignas(64) IndexPad {
        std::atomic<uint64_t> v{0};
        char _pad[64 - sizeof(std::atomic<uint64_t>)];
    };

    static constexpr size_t BUFFER_SIZE = 65536;
    Event event_buffer_[BUFFER_SIZE];

    IndexPad write_pos_;
    IndexPad read_pos_;

    std::atomic<uint64_t> dropped_events_{0};

    std::mutex file_mutex_;
    FILE* output_file_ = nullptr;
    std::chrono::steady_clock::time_point start_time_;
};

extern "C" {
    GGML_VIZ_API void ggml_viz_hook_graph_compute_begin(const ggml_cgraph* graph, const ggml_backend* backend = nullptr);
    GGML_VIZ_API void ggml_viz_hook_graph_compute_end(const ggml_cgraph* graph, const ggml_backend* backend = nullptr);
    GGML_VIZ_API void ggml_viz_hook_op_compute_begin(const ggml_tensor* tensor, const ggml_backend* backend = nullptr);
    GGML_VIZ_API void ggml_viz_hook_op_compute_end(const ggml_tensor* tensor, const ggml_backend* backend = nullptr);
    GGML_VIZ_API void ggml_viz_hook_tensor_alloc(const ggml_tensor* tensor, size_t size, const ggml_backend* backend = nullptr);
    GGML_VIZ_API void ggml_viz_hook_tensor_free(const ggml_tensor* tensor, const ggml_backend* backend = nullptr);
    GGML_VIZ_API bool ggml_viz_is_initialized();
    GGML_VIZ_API void ggml_viz_print_status();
}

GGML_VIZ_API bool install_ggml_hooks();
GGML_VIZ_API bool uninstall_ggml_hooks();

} // namespace ggml_viz