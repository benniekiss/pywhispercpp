cmake_minimum_required(VERSION 3.15...3.27)
project(whispercpp LANGUAGES C CXX)

if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.8 COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(whispercpp src/main.cpp)

add_subdirectory(vendor/whisper.cpp)
target_link_libraries(whispercpp PRIVATE whisper)

function(whispercpp_python_install_target target)
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

whispercpp_python_install_target(whisper)
whispercpp_python_install_target(whispercpp)
