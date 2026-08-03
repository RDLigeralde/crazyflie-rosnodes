cmake_minimum_required(VERSION 3.5)
project(crazyflie_cpp VERSION 1.0)

# Enable C++14 and warnings
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# Options
option(BUILD_PYTHON_BINDINGS "Build Python bindings using pybind11" OFF)
option(BUILD_CPP_EXAMPLES "Build C++ example executables" ON) # Default ON to build examples

# Include the low-level link library dependency
add_subdirectory(crazyflie-link-cpp EXCLUDE_FROM_ALL)

# Define the main crazyflie_cpp library target
# NOTE: Based on user-provided file list, only these two source files exist.
# This is likely INCOMPLETE and will cause linker errors later.
add_library(crazyflie_cpp
  src/Crazyflie.cpp
  src/crtp.cpp
)

# Setup include directories for the library
target_include_directories(crazyflie_cpp
  PUBLIC
    # For installation
    $<INSTALL_INTERFACE:include/crazyflie_cpp>
    # For build tree users (like the examples)
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/crazyflie_cpp>
    # Also need headers from the link library dependency
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/crazyflie-link-cpp/include>
)

# Link the library against its internal dependency
target_link_libraries(crazyflie_cpp
  PRIVATE
    crazyflieLinkCpp # Link against the target built from the crazyflie-link-cpp subdirectory
)

# --- Build C++ Examples ---
if(BUILD_CPP_EXAMPLES)
    message(STATUS "Building C++ Examples")

    # Find required packages for examples
    find_package(Threads REQUIRED)

    # Define the executable target for the control example
    # Assumes the code is saved as 'examples/control.cpp'
    add_executable(cf_control examples/control.cpp)

    # Link the executable against the main library and other dependencies
    # crazyflieLinkCpp and its dependencies (like libusb) should be linked transitively
    # because cf_control links against crazyflie_cpp which links against crazyflieLinkCpp.
    target_link_libraries(cf_control PRIVATE
        crazyflie_cpp      # Link against the library target defined above
        Threads::Threads   # Link against C++ standard threads support
        # No need to explicitly link crazyflieLinkCpp here if crazyflie_cpp links it properly
    )

    # Optional: Install the example executable if 'make install' is run
    # install(TARGETS cf_control DESTINATION bin)

else()
    message(STATUS "Skipping C++ Examples (BUILD_CPP_EXAMPLES is OFF)")
endif()
# --- End of Example Building Logic ---


# --- Installation Rules ---
# Generates lib/cmake/crazyflie_cpp/crazyflie_cpp-targets.cmake (the file
# cmake/crazyflie_cpp-config.cmake already includes()s, but that never
# existed with everything below commented out — that's the whole reason
# downstream find_package(crazyflie_cpp) failed) plus a copy of
# crazyflie_cpp-config.cmake itself, both under lib/cmake/crazyflie_cpp so
# a plain -DCMAKE_PREFIX_PATH=<install prefix> is enough for consumers to
# find it, matching the crazyflie_cpp::crazyflie_cpp target name
# src/crazyradio_driver_cpp/CMakeLists.txt already links against.
include(GNUInstallDirs)
install(TARGETS crazyflie_cpp
  EXPORT crazyflie_cpp-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/crazyflie_cpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
install(EXPORT crazyflie_cpp-targets
  FILE crazyflie_cpp-targets.cmake
  NAMESPACE crazyflie_cpp::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crazyflie_cpp
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/cmake/crazyflie_cpp-config.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/crazyflie_cpp
)