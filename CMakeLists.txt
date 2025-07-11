cmake_minimum_required(VERSION 3.15)
project(pywhispercpp LANGUAGES C CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_pywhispercpp src/main.cpp)

add_subdirectory(whisper.cpp)
target_link_libraries(_pywhispercpp PRIVATE whisper)

function(pywhispercpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    install(
        TARGETS ${target}
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION lib
        ARCHIVE DESTINATION lib
        FRAMEWORK DESTINATION lib
        RESOURCE DESTINATION lib
    )

    if(APPLE)
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "@loader_path;@loader_path/../../libggml/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    elseif(UNIX)
        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "$ORIGIN;$ORIGIN/../../libggml/lib"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endfunction()

pywhispercpp_python_install_target(whisper)
pywhispercpp_python_install_target(_pywhispercpp)
