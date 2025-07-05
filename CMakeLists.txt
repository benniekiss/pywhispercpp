cmake_minimum_required(VERSION 3.15)
project(pywhispercpp LANGUAGES CXX)

add_subdirectory(pybind11)
add_subdirectory(whisper.cpp)

pybind11_add_module(_pywhispercpp src/main.cpp)

target_link_libraries(_pywhispercpp PRIVATE whisper)

# Set rpath for shared libs
if(APPLE)
    set_target_properties(_pywhispercpp PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(_pywhispercpp PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
