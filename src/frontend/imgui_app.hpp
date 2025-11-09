#pragma once

#include <string>
#include <memory>
#include <vector>
#include "imgui_widgets.hpp"
#include "../instrumentation/ggml_hook.hpp"

namespace ggml_viz {

class TraceReader;

class ImGuiApp {
public:
    ImGuiApp();
    ~ImGuiApp();

    int run();
    bool load_trace_file(const std::string& filename);
    void enable_live_mode(bool no_hook = false, const std::string& trace_file = "");
    void disable_live_mode();
    bool is_live_mode() const;

private:
    struct AppData;
    std::unique_ptr<AppData> data_;

    bool initialize();
    void shutdown();
    void render_frame();

    void render_main_menu_bar();
    void render_hook_status_notification();
    void render_stats_overlay();
    void render_timeline_view();
    void render_graph_view();
    void render_tensor_inspector();
    void render_memory_view();
    void render_file_browser();

    void update_live_data();

    bool show_demo_window_ = false;
    bool show_timeline_ = true;
    bool show_graph_ = true;
    bool show_tensor_inspector_ = true;
    bool show_memory_view_ = true;
    bool show_file_browser_ = false;

    TimelineWidget timeline_widget_;
    TimelineWidget::TimelineConfig timeline_config_;

    GraphWidget graph_widget_;
    GraphWidget::GraphConfig graph_config_;
};

} // namespace ggml_viz