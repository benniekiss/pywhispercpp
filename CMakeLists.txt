cmake_minimum_required(VERSION 3.15)
project(pywhispercpp LANGUAGES CXX)

add_subdirectory(whisper.cpp)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_pywhispercpp src/main.cpp)

target_link_libraries(_pywhispercpp PRIVATE whisper)

# Set rpath for shared libs
if(APPLE)
    set_target_properties(_pywhispercpp PROPERTIES INSTALL_RPATH "@loader_path")
elseif(UNIX)
    set_target_properties(_pywhispercpp PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

install(TARGETS ggml
  LIBRARY DESTINATION pywhispercpp
  RUNTIME DESTINATION pywhispercpp
  ARCHIVE DESTINATION pywhispercpp)

install(TARGETS ggml-base
  LIBRARY DESTINATION pywhispercpp
  RUNTIME DESTINATION pywhispercpp
  ARCHIVE DESTINATION pywhispercpp)

install(TARGETS ggml-cpu
  LIBRARY DESTINATION pywhispercpp
  RUNTIME DESTINATION pywhispercpp
  ARCHIVE DESTINATION pywhispercpp)

install(TARGETS ggml-blas
  LIBRARY DESTINATION pywhispercpp
  RUNTIME DESTINATION pywhispercpp
  ARCHIVE DESTINATION pywhispercpp)

install(TARGETS ggml-metal
  LIBRARY DESTINATION pywhispercpp
  RUNTIME DESTINATION pywhispercpp
  ARCHIVE DESTINATION pywhispercpp)

install(TARGETS whisper
  LIBRARY DESTINATION pywhispercpp
  RUNTIME DESTINATION pywhispercpp
  ARCHIVE DESTINATION pywhispercpp)

install(TARGETS _pywhispercpp
  LIBRARY DESTINATION pywhispercpp         # Unix
  RUNTIME DESTINATION pywhispercpp         # Windows
  ARCHIVE DESTINATION pywhispercpp)        # Static lib, if needed
