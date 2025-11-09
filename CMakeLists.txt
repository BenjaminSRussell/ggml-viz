cmake_minimum_required(VERSION 3.15)
project(ggml-viz VERSION 0.0.12 LANGUAGES C CXX)

if(APPLE)
    enable_language(OBJC)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)

add_subdirectory(third_party)

# OpenMP warnings are expected: macOS uses Accelerate, Linux uses available BLAS
add_subdirectory(third_party/ggml)

if(WIN32)
    include(FetchContent)
    FetchContent_Declare(
        minhook
        GIT_REPOSITORY https://github.com/TsudaKageyu/minhook.git
        GIT_TAG        v1.3.3
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(minhook)

    add_library(MinHook STATIC
        ${minhook_SOURCE_DIR}/src/buffer.c
        ${minhook_SOURCE_DIR}/src/hook.c
        ${minhook_SOURCE_DIR}/src/trampoline.c
        ${minhook_SOURCE_DIR}/src/hde/hde32.c
        ${minhook_SOURCE_DIR}/src/hde/hde64.c
    )
    target_include_directories(MinHook PUBLIC ${minhook_SOURCE_DIR}/include)
    target_compile_definitions(MinHook PRIVATE MH_STANDALONE)
    set_target_properties(MinHook PROPERTIES
        OUTPUT_NAME "MinHook"
        POSITION_INDEPENDENT_CODE ON
    )
endif()

add_subdirectory(src)
add_subdirectory(examples)

enable_testing()
add_subdirectory(tests)

