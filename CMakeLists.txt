cmake_minimum_required(VERSION 3.15)
project(pywhispercpp LANGUAGES CXX)

add_subdirectory(whisper.cpp)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_pywhispercpp src/main.cpp)

target_link_libraries(_pywhispercpp PRIVATE whisper)

function(pywhispercpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    install(
        TARGETS ${target}
        LIBRARY DESTINATION pywhispercpp/lib
        RUNTIME DESTINATION pywhispercpp/lib
        ARCHIVE DESTINATION pywhispercpp/lib
        FRAMEWORK DESTINATION pywhispercpp/lib
        RESOURCE DESTINATION pywhispercpp/lib
    )

    set_target_properties(${target} PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    if(UNIX)
        if(APPLE)
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "@loader_path"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        else()
            set_target_properties(${target} PROPERTIES
                INSTALL_RPATH "$ORIGIN"
                BUILD_WITH_INSTALL_RPATH TRUE
            )
        endif()
    endif()
endfunction()

pywhispercpp_python_install_target(ggml)
pywhispercpp_python_install_target(ggml-base)

pywhispercpp_python_install_target(ggml-amx)
pywhispercpp_python_install_target(ggml-blas)
pywhispercpp_python_install_target(ggml-can)
pywhispercpp_python_install_target(ggml-cpu)
pywhispercpp_python_install_target(ggml-cuda)
pywhispercpp_python_install_target(ggml-hip)
pywhispercpp_python_install_target(ggml-kompute)
pywhispercpp_python_install_target(ggml-metal)
pywhispercpp_python_install_target(ggml-musa)
pywhispercpp_python_install_target(ggml-rpc)
pywhispercpp_python_install_target(ggml-sycl)
pywhispercpp_python_install_target(ggml-vulkan)

pywhispercpp_python_install_target(whisper)
pywhispercpp_python_install_target(_pywhispercpp)
