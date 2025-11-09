#include "imgui_app.hpp"
#include "utils/trace_reader.hpp"
#include "instrumentation/ggml_hook.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <iostream>
#include <memory>
#include <chrono>
#include <ctime>
#include <sys/stat.h>
#include <unordered_map>

namespace ggml_viz {

struct ImGuiApp::AppData {
    GLFWwindow* window = nullptr;
    std::unique_ptr<TraceReader> trace_reader;
    std::string current_filename;
    std::string error_message;

    bool trace_loaded = false;
    int selected_event = -1;

    bool live_mode = false;
    bool live_mode_no_hook = false;
    std::vector<Event> live_events;

    std::chrono::steady_clock::time_point last_live_update;
    std::atomic<bool> live_data_available{false};

    std::string live_file_path;
    std::time_t last_file_mod_time = 0;
    size_t last_file_size = 0;
    size_t last_file_event_count = 0;
    std::chrono::steady_clock::time_point last_file_check_time;
    std::unique_ptr<TraceReader> live_trace_reader;

    static constexpr size_t FILE_PATH_BUFFER_SIZE = 512;
    char file_path_buffer[FILE_PATH_BUFFER_SIZE] = {0};

    static constexpr size_t MAX_LIVE_EVENTS = 50'000;

    void limit_live_events_buffer() {
        if (live_events.size() > MAX_LIVE_EVENTS) {
            live_events.erase(
                live_events.begin(),
                live_events.end() - MAX_LIVE_EVENTS
            );
        }
    }
};

ImGuiApp::ImGuiApp() : data_(std::make_unique<AppData>()) {
}

ImGuiApp::~ImGuiApp() {
    shutdown();
}

int ImGuiApp::run() {
    if (!initialize()) {
        return -1;
    }

    if (!data_->window) {
        std::cerr << "[ImGuiApp] ERROR: Failed to create window\n";
        return -1;
    }

    while (!glfwWindowShouldClose(data_->window)) {
        glfwPollEvents();
        render_frame();
    }

    shutdown();
    return 0;
}

bool ImGuiApp::initialize() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    data_->window = glfwCreateWindow(1280, 720, "GGML Visualizer", nullptr, nullptr);
    if (!data_->window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(data_->window);
    glfwSwapInterval(1);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

#ifdef IMGUI_HAS_DOCK
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
#endif

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(data_->window, true);
    ImGui_ImplOpenGL3_Init("#version 150");

    return true;
}

void ImGuiApp::enable_live_mode(bool no_hook, const std::string& trace_file) {
    data_->live_mode = true;
    data_->live_mode_no_hook = no_hook;
    data_->live_events.clear();
    data_->last_live_update = std::chrono::steady_clock::now();
    data_->current_filename = "[Live Mode]";

    if (!no_hook) {
        try {
            auto& hook = GGMLHook::instance();
            hook.start();

            std::cout << "[ImGuiApp] Live mode enabled and GGML hook started" << std::endl;
            std::cout << "[ImGuiApp] Hook active: " << (hook.is_active() ? "YES" : "NO") << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "[ImGuiApp] Error starting GGML hook: " << e.what() << std::endl;
        }
    } else {
        std::cout << "[ImGuiApp] Live mode enabled with built-in hook disabled (--no-hook)" << std::endl;
    }

    if (!trace_file.empty()) {
        data_->live_file_path = trace_file;
        std::cout << "[ImGuiApp] Monitoring specified trace file: " << data_->live_file_path << std::endl;
    } else {
        const char* env_output = std::getenv("GGML_VIZ_OUTPUT");
        if (env_output) {
            data_->live_file_path = env_output;
            std::cout << "[ImGuiApp] Monitoring external trace file: " << data_->live_file_path << std::endl;
        } else {
            data_->live_file_path = "test.ggmlviz";
            std::cout << "[ImGuiApp] No GGML_VIZ_OUTPUT set, monitoring default: " << data_->live_file_path << std::endl;
            std::cout << "[ImGuiApp] NOTE: 0 events is expected until you run a GGML application" << std::endl;
            std::cout << "[ImGuiApp]       Run with LD_PRELOAD or DYLD_INSERT_LIBRARIES to capture events" << std::endl;
        }
    }
    

    data_->last_file_mod_time = 0;
    data_->last_file_size = 0;
}

void ImGuiApp::disable_live_mode() {
    data_->live_mode = false;
    data_->live_events.clear();
    

    try {
        auto& hook = GGMLHook::instance();
        if (hook.is_active()) {
            hook.stop();
            std::cout << "[ImGuiApp] Live mode disabled and GGML hook stopped" << std::endl;
        } else {
            std::cout << "[ImGuiApp] Live mode disabled (hook was not active)" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "[ImGuiApp] Error stopping GGML hook: " << e.what() << std::endl;
    }
    

    data_->live_file_path.clear();
    data_->last_file_mod_time = 0;
    data_->last_file_size = 0;
    data_->live_trace_reader.reset();
}

bool ImGuiApp::is_live_mode() const {
    return data_->live_mode;
}

void ImGuiApp::update_live_data() {
    if (!data_->live_mode) return;

    try {
        auto& hook = GGMLHook::instance();
        if (hook.is_active()) {
            auto new_events = hook.consume_available_events();
            if (!new_events.empty()) {
                data_->live_events.insert(data_->live_events.end(), new_events.begin(), new_events.end());
                data_->last_live_update = std::chrono::steady_clock::now();
                data_->live_data_available = true;
                data_->limit_live_events_buffer();
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[ImGuiApp] Error updating live data from hook: " << e.what() << std::endl;
    }

    auto now = std::chrono::steady_clock::now();
    auto time_since_last_check = std::chrono::duration_cast<std::chrono::milliseconds>(now - data_->last_file_check_time);

    if (!data_->live_file_path.empty() && time_since_last_check.count() > 100) {
        data_->last_file_check_time = now;

        try {
            struct stat file_stat;
            if (stat(data_->live_file_path.c_str(), &file_stat) == 0) {
                if (file_stat.st_mtime > data_->last_file_mod_time ||
                    static_cast<size_t>(file_stat.st_size) > data_->last_file_size) {

                    std::cout << "[ImGuiApp] File changed - reloading: " << data_->live_file_path
                              << " (size: " << file_stat.st_size << " bytes)" << std::endl;
                    auto new_trace_reader = std::make_unique<TraceReader>(data_->live_file_path);
                    if (new_trace_reader->is_valid()) {
                        const auto& events = new_trace_reader->events();
                        size_t start_idx = 0;

                        if (data_->live_trace_reader && events.size() >= data_->last_file_event_count) {
                            start_idx = data_->last_file_event_count;
                        } else {
                            start_idx = 0;
                            std::cout << "[ImGuiApp] File appears to be recreated/truncated, loading all events" << std::endl;
                        }

                        if (events.size() > start_idx) {
                            size_t new_event_count = events.size() - start_idx;
                            data_->live_events.insert(data_->live_events.end(),
                                                     events.begin() + start_idx, events.end());
                            data_->last_live_update = std::chrono::steady_clock::now();
                            data_->live_data_available = true;

                            std::cout << "[ImGuiApp] Loaded " << new_event_count
                                      << " new events from external file (total events in file: " << events.size() << ")" << std::endl;

                            data_->last_file_event_count = events.size();
                        } else {
                            std::cout << "[ImGuiApp] No new events to load (file has " << events.size()
                                      << " events, last processed: " << data_->last_file_event_count << ")" << std::endl;
                        }

                        data_->last_file_mod_time = file_stat.st_mtime;
                        data_->last_file_size = file_stat.st_size;
                        data_->live_trace_reader = std::move(new_trace_reader);
                        data_->limit_live_events_buffer();
                    }
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[ImGuiApp] Error monitoring external file: " << e.what() << std::endl;
        }
    }
}

void ImGuiApp::shutdown() {
    if (data_->window) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        glfwDestroyWindow(data_->window);
        glfwTerminate();
        data_->window = nullptr;
    }
}

void ImGuiApp::render_frame() {

    update_live_data();
    

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    

#ifdef IMGUI_HAS_DOCK
    ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
#endif
    

    render_main_menu_bar();
    

    render_hook_status_notification();
    

    render_stats_overlay();
    

    if (show_file_browser_) {
        render_file_browser();
    }
    
    if ((data_->trace_loaded && data_->trace_reader) || data_->live_mode) {
        if (show_timeline_) {
            render_timeline_view();
        }
        if (show_graph_) {
            render_graph_view();
        }
        if (show_tensor_inspector_) {
            render_tensor_inspector();
        }
        if (show_memory_view_) {
            render_memory_view();
        }
    }
    

    if (show_demo_window_) {
        ImGui::ShowDemoWindow(&show_demo_window_);
    }
    

    if (!data_->error_message.empty()) {
        ImGui::OpenPopup("Error");
        if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("%s", data_->error_message.c_str());
            if (ImGui::Button("OK")) {
                data_->error_message.clear();
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
    }
    

    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(data_->window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.45f, 0.55f, 0.60f, 1.00f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    glfwSwapBuffers(data_->window);
}

void ImGuiApp::render_main_menu_bar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Trace...")) {
                show_file_browser_ = true;
            }
            if (ImGui::MenuItem("Close Trace", nullptr, false, data_->trace_loaded)) {
                data_->trace_reader.reset();
                data_->trace_loaded = false;
                data_->current_filename.clear();
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Exit")) {
                glfwSetWindowShouldClose(data_->window, GLFW_TRUE);
            }
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("View")) {
            if (ImGui::MenuItem("Timeline", nullptr, &show_timeline_, data_->trace_loaded || data_->live_mode)) {}
            if (ImGui::MenuItem("Graph", nullptr, &show_graph_, data_->trace_loaded || data_->live_mode)) {}
            if (ImGui::MenuItem("Tensor Inspector", nullptr, &show_tensor_inspector_, data_->trace_loaded || data_->live_mode)) {}
            if (ImGui::MenuItem("Memory View", nullptr, &show_memory_view_, data_->trace_loaded || data_->live_mode)) {}
            ImGui::MenuItem("Demo Window", nullptr, &show_demo_window_);
            ImGui::EndMenu();
        }
        
        if (ImGui::BeginMenu("Help")) {
            if (ImGui::MenuItem("About")) {

            }
            ImGui::EndMenu();
        }
        

        if (data_->trace_loaded || data_->live_mode) {
            ImGui::SameLine(ImGui::GetWindowWidth() - 350);
            if (data_->live_mode) {
                try {
                    auto& hook = GGMLHook::instance();
                    const char* status = hook.is_active() ? "good" : "bad";
                    ImGui::Text("%s Live: %zu events", status, data_->live_events.size());
                } catch (...) {
                    ImGui::Text("Live: %zu events", data_->live_events.size());
                }
            } else {
                ImGui::Text("Loaded: %s (%zu events)",
                           data_->current_filename.c_str(),
                           data_->trace_reader->event_count());
            }
        }
        
        ImGui::EndMainMenuBar();
    }
}

void ImGuiApp::render_file_browser() {
    if (ImGui::Begin("Open Trace File", &show_file_browser_)) {
        ImGui::Text("Enter path to .ggmlviz file:");
        ImGui::InputText("##filepath", data_->file_path_buffer, sizeof(data_->file_path_buffer));
        
        ImGui::Separator();
        
        if (ImGui::Button("Open")) {
            if (strlen(data_->file_path_buffer) > 0) {
                if (load_trace_file(data_->file_path_buffer)) {
                    show_file_browser_ = false;
                }
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel")) {
            show_file_browser_ = false;
        }
        

        ImGui::Text("Note: Enter full path to trace file");
        ImGui::Text("Example: /path/to/trace.ggmlviz");
    }
    ImGui::End();
}

bool ImGuiApp::load_trace_file(const std::string& filename) {
    try {

        if (data_->live_mode) {
            disable_live_mode();
        }

        FILE* test_file = fopen(filename.c_str(), "rb");
        if (!test_file) {

            if (filename.empty()) {
                data_->error_message = "Error: No file path specified.";
            } else if (filename.find_last_of('.') == std::string::npos || 
                      filename.substr(filename.find_last_of('.')) != ".ggmlviz") {
                data_->error_message = "Error: Invalid file type.\n\nExpected a .ggmlviz trace file.\nSelected: " + filename;
            } else {
                data_->error_message = "Error: File not found or access denied.\n\nFile: " + filename + 
                                      "\n\nPlease check:\n• File exists\n• File permissions\n• Path is correct";
            }
            return false;
        }
        

        fseek(test_file, 0, SEEK_END);
        long file_size = ftell(test_file);
        fclose(test_file);
        
        if (file_size == 0) {
            data_->error_message = "Error: Empty trace file.\n\nFile: " + filename + 
                                  "\n\nThe trace file contains no data. Please ensure the file was generated correctly.";
            return false;
        }
        
        if (file_size < 12) {  // Minimum size for header
            data_->error_message = "Error: Invalid trace file.\n\nFile: " + filename + 
                                  "\n\nFile is too small (" + std::to_string(file_size) + " bytes) to contain valid trace data.";
            return false;
        }
        

        auto reader = std::make_unique<TraceReader>(filename);
        if (!reader->is_valid()) {

            FILE* check_file = fopen(filename.c_str(), "rb");
            if (check_file) {
                char magic[8] = {0};
                if (fread(magic, 1, 8, check_file) == 8) {
                    if (strncmp(magic, "GGMLVIZ1", 8) != 0) {
                        data_->error_message = "Error: Invalid trace file format.\n\nFile: " + filename + 
                                              "\n\nThis does not appear to be a valid GGML trace file.\n" +
                                              "Expected magic header 'GGMLVIZ1', found: '" + std::string(magic, 8) + "'";
                    } else {
                        data_->error_message = "Error: Corrupted trace file.\n\nFile: " + filename + 
                                              "\n\nThe file header is valid but the trace data appears to be corrupted.\n" +
                                              "The file may have been truncated or damaged.";
                    }
                } else {
                    data_->error_message = "Error: Cannot read trace file header.\n\nFile: " + filename + 
                                          "\n\nFile exists but cannot be read properly. Check file permissions.";
                }
                fclose(check_file);
            } else {
                data_->error_message = "Error: File access lost during loading.\n\nFile: " + filename;
            }
            return false;
        }
        

        if (reader->event_count() == 0) {
            data_->error_message = "Warning: Empty trace data.\n\nFile: " + filename + 
                                  "\n\nThe trace file loaded successfully but contains no events.\n" +
                                  "This might indicate:\n• No GGML operations were traced\n• Tracing was not enabled\n• The model ran but no operations occurred";

        }
        

        data_->trace_reader = std::move(reader);
        data_->trace_loaded = true;
        data_->current_filename = filename.substr(filename.find_last_of("/\\") + 1); // Just filename for display
        data_->selected_event = -1;
        
        return true;
        
    } catch (const std::bad_alloc& e) {
        data_->error_message = "Error: Out of memory.\n\nFile: " + filename + 
                              "\n\nNot enough memory to load this trace file.\n" +
                              "Try closing other applications or loading a smaller trace file.";
        return false;
    } catch (const std::exception& e) {
        data_->error_message = "Error: Unexpected error loading trace.\n\nFile: " + filename + 
                              "\n\nDetails: " + std::string(e.what()) + 
                              "\n\nThis may indicate a bug in the application or a severely corrupted file.";
        return false;
    } catch (...) {
        data_->error_message = "Error: Unknown error occurred.\n\nFile: " + filename + 
                              "\n\nAn unexpected error occurred while loading the trace file.";
        return false;
    }
}

void ImGuiApp::render_timeline_view() {

    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.13f, 0.54f, 0.82f, 1.0f)); // Blue color
    
    if (ImGui::Begin("Timeline View")) {

        const std::vector<Event>* events_ptr = nullptr;
        size_t event_count = 0;
        std::string mode_info;
        
        if (data_->live_mode) {
            events_ptr = &data_->live_events;
            event_count = data_->live_events.size();
            mode_info = "[LIVE MODE]";
            

            auto& hook = GGMLHook::instance();
            if (hook.is_active()) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "✅ LIVE MODE ACTIVE");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "❌ HOOK INACTIVE");
            }
            ImGui::Text("Live Events: %zu", event_count);
            
            if (event_count > 0) {
                auto time_since_last = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - data_->last_live_update).count();
                ImGui::Text("Last Update: %lld ms ago", time_since_last);
            }
            
        } else if (data_->trace_reader) {
            events_ptr = &data_->trace_reader->events();
            event_count = data_->trace_reader->events().size();
            mode_info = "[LOADED TRACE]";
        } else {
            ImGui::Text("No trace loaded and live mode not active");
            ImGui::End();
            return;
        }
        
        const auto& events = *events_ptr;
        

        ImGui::Text("%s", mode_info.c_str());
        ImGui::Text("Total Events: %zu", events.size());
        
        if (data_->live_mode) {

            if (events.size() >= 2) {
                uint64_t duration_ns = events.back().timestamp_ns - events.front().timestamp_ns;
                ImGui::Text("Duration: %.2f ms", duration_ns / 1e6);
            } else {
                ImGui::Text("Duration: N/A");
            }
            ImGui::Text("Operations: Live counting...");
        } else if (data_->trace_reader) {
            auto op_timings = data_->trace_reader->get_op_timings();
            ImGui::Text("Total Duration: %.2f ms", data_->trace_reader->get_total_duration_ns() / 1e6);
            ImGui::Text("Operations: %zu", op_timings.size());
        }
        
        ImGui::Separator();
        

        static char search_buffer[256] = "";
        ImGui::InputTextWithHint("##search", "🔍 Filter operations...", search_buffer, sizeof(search_buffer));
        ImGui::SameLine();
        if (ImGui::Button("Clear")) {
            search_buffer[0] = '\0';
        }
        

        if (ImGui::BeginTabBar("TimelineViews")) {

            if (ImGui::BeginTabItem("Visual Timeline")) {

                const TraceReader* trace_reader_for_widget = nullptr;
                if (data_->live_mode && data_->live_trace_reader) {
                    trace_reader_for_widget = data_->live_trace_reader.get();
                } else {
                    trace_reader_for_widget = data_->trace_reader.get();
                }
                
                if (trace_reader_for_widget) {
                    timeline_widget_.render("##timeline", trace_reader_for_widget, timeline_config_);
                    

                    int selected = timeline_widget_.get_selected_event();
                    if (selected != data_->selected_event) {
                        data_->selected_event = selected;
                    }
                } else {

                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                    ImVec2 canvas_size = ImGui::GetContentRegionAvail();
                    
                    if (canvas_size.x > 100 && canvas_size.y > 50) {

                        ImU32 track_color = IM_COL32(80, 80, 80, 100);
                        ImU32 tie_color = IM_COL32(60, 60, 60, 150);
                        

                        float track_y1 = canvas_pos.y + canvas_size.y * 0.4f;
                        float track_y2 = canvas_pos.y + canvas_size.y * 0.6f;
                        draw_list->AddLine(ImVec2(canvas_pos.x, track_y1), ImVec2(canvas_pos.x + canvas_size.x, track_y1), track_color, 3.0f);
                        draw_list->AddLine(ImVec2(canvas_pos.x, track_y2), ImVec2(canvas_pos.x + canvas_size.x, track_y2), track_color, 3.0f);
                        

                        for (float x = canvas_pos.x; x < canvas_pos.x + canvas_size.x; x += 30) {
                            draw_list->AddLine(ImVec2(x, track_y1 - 10), ImVec2(x, track_y2 + 10), tie_color, 2.0f);
                        }
                        

                        ImVec2 text_pos = ImVec2(canvas_pos.x + canvas_size.x * 0.5f - 100, canvas_pos.y + canvas_size.y * 0.5f - 20);
                        draw_list->AddText(text_pos, IM_COL32(150, 150, 150, 255), "Waiting for first operation...");
                    } else {
                        ImGui::Text("Waiting for first operation...");
                    }
                    

                    ImGui::Dummy(ImVec2(0, std::max(100.0f, canvas_size.y)));
                }
                
                ImGui::EndTabItem();
            }
            

            if (ImGui::BeginTabItem("Events")) {
                if (ImGui::BeginChild("EventList")) {
                    ImGuiListClipper clipper;
                    clipper.Begin(events.size());
                    
                    while (clipper.Step()) {
                        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; i++) {
                            const auto& event = events[i];
                            
                            bool is_selected = (data_->selected_event == i);
                            

                            std::string label = std::to_string(i) + ": " + event_type_name(event.type);
                            if (event.label) {
                                label += " (" + std::string(event.label) + ")";
                            }
                            
                            if (ImGui::Selectable(label.c_str(), is_selected)) {
                                data_->selected_event = i;
                                timeline_widget_.set_selected_event(i);
                            }
                            
                            if (ImGui::IsItemHovered()) {
                                ImGui::BeginTooltip();
                                ImGui::Text("Event: %s", event_name.c_str());
                                ImGui::Text("Timestamp: %.3f ms", event.timestamp_ns / 1e6);
                                ImGui::Text("Thread: %d", event.thread_id);
                                if (event.label) {
                                    ImGui::Text("Label: %s", event.label);
                                }
                                ImGui::EndTooltip();
                            }
                        }
                    }
                    clipper.End();
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            

            if (ImGui::BeginTabItem("Op Timings")) {
                if (ImGui::BeginChild("OpTimings")) {
                    if (data_->live_mode) {
                        ImGui::Text("Live mode: Operation timings calculated on-the-fly");
                        ImGui::Text("Events collected: %zu", events.size());

                    } else if (data_->trace_reader) {
                        auto op_timings = data_->trace_reader->get_op_timings();
                        uint64_t total_duration = data_->trace_reader->get_total_duration_ns();
                        
                        ImGui::Columns(3, "OpTimingsColumns");
                        ImGui::Text("Operation");
                        ImGui::NextColumn();
                        ImGui::Text("Duration");
                        ImGui::NextColumn();
                        ImGui::Text("%% of Total");
                        ImGui::NextColumn();
                        ImGui::Separator();
                        
                        for (size_t i = 0; i < op_timings.size(); i++) {
                            const auto& timing = op_timings[i];
                            
                            ImGui::Text("%s", timing.name.c_str());
                            ImGui::NextColumn();
                            ImGui::Text("%.3f ms", timing.duration_ns / 1e6);
                            ImGui::NextColumn();
                            if (total_duration > 0) {
                                ImGui::Text("%.1f%%", (timing.duration_ns * 100.0) / total_duration);
                            } else {
                                ImGui::Text("N/A");
                            }
                            ImGui::NextColumn();
                        }
                        ImGui::Columns(1);
                    }
                }
                ImGui::EndChild();
                ImGui::EndTabItem();
            }
            
            ImGui::EndTabBar();
        }
    }
    ImGui::End();
    ImGui::PopStyleColor(); // Pop blue timeline color
}

void ImGuiApp::render_graph_view() {

    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.30f, 0.69f, 0.31f, 1.0f)); // Green color
    
    if (ImGui::Begin("Graph View")) {
        if (!data_->trace_reader && !data_->live_mode) {
            ImGui::Text("No trace loaded and live mode not active");
            ImGui::End();
            return;
        }
        
        if (data_->live_mode) {
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "📊 LIVE GRAPH VIEW");
            ImGui::Text("Live events: %zu", data_->live_events.size());
            

            size_t graph_begin_count = 0;
            size_t graph_end_count = 0;
            for (const auto& event : data_->live_events) {
                if (event.type == EventType::GRAPH_COMPUTE_BEGIN) graph_begin_count++;
                if (event.type == EventType::GRAPH_COMPUTE_END) graph_end_count++;
            }
            ImGui::Text("Graph Begin Events: %zu", graph_begin_count);
            ImGui::Text("Graph End Events: %zu", graph_end_count);
            
            ImGui::Separator();
            

            if (data_->live_trace_reader) {
                graph_widget_.render("##compute_graph", data_->live_trace_reader.get(), graph_config_);
                



            } else {
                ImGui::Text("Loading graph data...");
            }
            
        } else if (data_->trace_reader) {
            auto graph_events = data_->trace_reader->get_graph_events();
            ImGui::Text("Graph Events: %zu", graph_events.size());
            
            ImGui::Separator();
            

            graph_widget_.render("##compute_graph", data_->trace_reader.get(), graph_config_);
            
            // Sync selection between graph widget and other views
            int selected_node = graph_widget_.get_selected_node();
            if (selected_node >= 0) {


                ImGui::Text("Selected Node: %d", selected_node);
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleColor(); // Pop green graph color
}

void ImGuiApp::render_tensor_inspector() {

    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.91f, 0.12f, 0.39f, 1.0f)); // Pink color
    
    if (ImGui::Begin("Tensor Inspector")) {
        if (!data_->trace_reader && !data_->live_mode) {
            ImGui::Text("No trace loaded and live mode not active");
            ImGui::End();
            return;
        }
        
        if (data_->live_mode) {
            ImGui::TextColored(ImVec4(0.91f, 0.12f, 0.39f, 1.0f), "🔬 LIVE TENSOR INSPECTOR");
            ImGui::Text("Live events: %zu", data_->live_events.size());
            
            if (data_->selected_event >= 0 && 
                data_->selected_event < static_cast<int>(data_->live_events.size())) {
                const auto& event = data_->live_events[data_->selected_event];
                
                ImGui::Separator();
                ImGui::Text("Selected Live Event Details:");
                ImGui::Text("Type: %d", static_cast<int>(event.type));
                ImGui::Text("Timestamp: %llu ns", event.timestamp_ns);
                ImGui::Text("Thread ID: %d", event.thread_id);
                
                if (event.label) {
                    ImGui::Text("Label: %s", event.label);
                }
            } else {
                ImGui::Text("Select an event from the timeline to inspect");
            }
            
        } else if (data_->trace_reader) {
            if (data_->selected_event >= 0 && 
                data_->selected_event < static_cast<int>(data_->trace_reader->events().size())) {
                const auto& event = data_->trace_reader->events()[data_->selected_event];
                
                ImGui::Text("Selected Event Details:");
                ImGui::Text("Type: %d", static_cast<int>(event.type));
                ImGui::Text("Timestamp: %llu ns", event.timestamp_ns);
                ImGui::Text("Thread ID: %d", event.thread_id);
                
                if (event.label) {
                    ImGui::Text("Label: %s", event.label);
                }
                

            } else {
                ImGui::Text("Select an event from the timeline to inspect");
            }
        }
    }
    ImGui::End();
    ImGui::PopStyleColor(); // Pop pink tensor inspector color
}

void ImGuiApp::render_memory_view() {

    ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(1.0f, 0.60f, 0.0f, 1.0f)); // Orange color
    
    if (ImGui::Begin("Memory View")) {
        if (!data_->trace_reader && !data_->live_mode) {
            ImGui::Text("No trace loaded and live mode not active");
            ImGui::End();
            return;
        }
        
        ImGui::Text("Memory visualization coming soon...");
    }
    ImGui::End();
    ImGui::PopStyleColor(); // Pop orange memory view color
}

void ImGuiApp::render_hook_status_notification() {

    bool should_show_notification = false;
    std::string notification_text;
    
    if (data_->live_mode) {
        if (data_->live_mode_no_hook) {

            if (!data_->live_trace_reader || data_->live_trace_reader->events().empty()) {
                should_show_notification = true;
                notification_text = "📄  FILE-BASED LIVE MODE - WAITING FOR DATA\n\n"
                                  "Monitoring trace file: " + data_->live_file_path + "\n\n"
                                  "To generate data, run:\n\n"
                                  "env GGML_VIZ_OUTPUT=" + data_->live_file_path + " \\\n"
                                  "    DYLD_INSERT_LIBRARIES=./build/src/libggml_viz_hook.dylib \\\n"
                                  "    ./third_party/llama.cpp/build/bin/llama-cli \\\n"
                                  "    -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \\\n"
                                  "    -p \"Hello world\" -n 10";
            }
        } else {

            try {
                auto& hook = GGMLHook::instance();
                if (!hook.is_active()) {
                    should_show_notification = true;
                    notification_text = "⚠️  RING BUFFER LIVE MODE - HOOK INACTIVE\n\n"
                                      "The GGML hook is not capturing events. To fix this:\n\n"
                                      "1. Set environment variables:\n"
                                      "   export GGML_VIZ_OUTPUT=trace.ggmlviz\n\n"
                                      "2. Run your GGML application with:\n"
                                      "   env DYLD_INSERT_LIBRARIES=./build/src/libggml_viz_hook.dylib \\\n"
                                      "       ./third_party/llama.cpp/build/bin/llama-cli \\\n"
                                      "       -m ./models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \\\n"
                                      "       -p \"Hello world\" -n 10\n\n"
                                      "3. Or for Linux:\n"
                                      "   env LD_PRELOAD=./build/src/libggml_viz_hook.so your_app";
                } else if (hook.event_count() == 0) {
                    should_show_notification = true;
                    notification_text = "ℹ️  RING BUFFER HOOK ACTIVE - WAITING FOR OPERATIONS\n\n"
                                      "The GGML hook is active and ready to capture events.\n"
                                      "Run GGML operations in your application to see data here.";
                }
            } catch (const std::exception& e) {
                should_show_notification = true;
                notification_text = "❌  HOOK ERROR\n\nError accessing GGML hook: " + std::string(e.what());
            }
        }
    } else if (!data_->trace_loaded) {
        should_show_notification = true;
        notification_text = "📁  NO TRACE LOADED\n\n"
                          "Load a trace file using File → Open Trace...\n"
                          "or enable Live Mode to capture real-time data.";
    }
    
    if (should_show_notification) {
        ImGui::SetNextWindowPos(ImGui::GetMainViewport()->GetCenter(), ImGuiCond_Always, ImVec2(0.5f, 0.5f));
        ImGui::SetNextWindowBgAlpha(0.9f);
        
        if (ImGui::BeginPopupModal("Hook Status", nullptr, 
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove)) {
            ImGui::Text("%s", notification_text.c_str());
            
            ImGui::Separator();
            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
            }
            ImGui::EndPopup();
        }
        

        if (!ImGui::IsPopupOpen("Hook Status")) {
            ImGui::OpenPopup("Hook Status");
        }
    }
}

void ImGuiApp::render_stats_overlay() {

    const float DISTANCE = 10.0f;
    ImGuiIO& io = ImGui::GetIO();
    ImVec2 window_pos = ImVec2(io.DisplaySize.x - DISTANCE, DISTANCE);
    ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f);
    
    ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
    ImGui::SetNextWindowBgAlpha(0.35f);
    
    if (ImGui::Begin("Stats Overlay", nullptr, 
        ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | 
        ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | 
        ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav)) {
        

        ImGui::Text("FPS: %.1f", io.Framerate);
        

        if (data_->live_mode) {
            try {
                auto& hook = GGMLHook::instance();
                ImGui::Text("Events: %zu", hook.event_count());
                ImGui::Text("Dropped: %zu", hook.get_dropped_events());
                ImGui::Text("Hook: %s", hook.is_active() ? "✅" : "❌");
            } catch (const std::exception&) {
                ImGui::Text("Hook: ❌");
            }
        } else if (data_->trace_reader) {
            ImGui::Text("Events: %zu", data_->trace_reader->events().size());
            ImGui::Text("Duration: %.1fms", data_->trace_reader->get_total_duration_ns() / 1e6);
        }
        
    }
    ImGui::End();
}

} // namespace ggml_viz