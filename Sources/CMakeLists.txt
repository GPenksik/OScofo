# OScofo Library CMake Configuration
# Creates a shared library (DLL) for OScofo components

cmake_minimum_required(VERSION 3.15)
set(CMAKE_CONFIGURATION_TYPES Debug CACHE STRING "" FORCE)

# Set up OScofo as a shared library
add_library(OScofoLib SHARED
    OScofo/OScofo.cpp
    OScofo/mdp.cpp
    OScofo/mir.cpp
    OScofo/score.cpp
    OScofo/lua.cpp
)

# Create export header for DLL (Automatic generation of __declspec(dllimport/dllexport)
include(GenerateExportHeader)
generate_export_header(OScofoLib
    BASE_NAME OSCOFO
    EXPORT_MACRO_NAME OSCOFO_API
    EXPORT_FILE_NAME OScofo/oscofo_export.h
    STATIC_DEFINE OSCOFO_STATIC_DEFINE
)

# Set target properties
set_target_properties(OScofoLib PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "OScofoLib"
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    # Set output directory for standalone build
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Output"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Output"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Output"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/Output"
    ARCHIVE_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/Output"
    ARCHIVE_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Output"
)

# Include directories
target_include_directories(OScofoLib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/OScofo>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../Libraries/tree-sitter/lib/include>
    $<INSTALL_INTERFACE:include>
)

# Add Include directory for performance_timer.h and other headers
target_include_directories(OScofoLib PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Include
)

# Find and link required dependencies
find_package(Boost REQUIRED COMPONENTS headers)
target_link_libraries(OScofoLib PUBLIC
    Boost::headers
)

# Define macro to enable DLL export headers
target_compile_definitions(OScofoLib PRIVATE
    OSCOFO_BUILDING_DLL
)

# Platform-specific configurations
if(WIN32)
    target_compile_definitions(OScofoLib PRIVATE
        _USE_MATH_DEFINES
        WIN32_LEAN_AND_MEAN
        NOMINMAX
    )
endif()

# Find FFTW3 using modern CMake (vcpkg provides FFTW3::fftw3 target)
find_package(FFTW3 REQUIRED CONFIG)
if(FFTW3_FOUND)
    target_link_libraries(OScofoLib PRIVATE FFTW3::fftw3)
    message(STATUS "Found FFTW3 via CONFIG mode")
    set(FFTW3_ENABLED TRUE)
endif()

if(NOT FFTW3_ENABLED)
    message(WARNING "FFTW3 not found - MIR functionality may not work")
endif()

target_compile_definitions(OScofoLib PRIVATE 
    COMPILE_OSCOFO_WITH_LOGGER=0
)

if(COMPILE_OSCOFO_WITH_LOGGER)
    # Add HDF5 support - combine vcpkg HDF5 library with local header-only wrappers
    find_package(HDF5 REQUIRED)

    # Set up local header-only HDF5 wrapper directories
    set(HDF5_HEADER_DIRS 
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../Include/hdf5_logger"
        "${CMAKE_CURRENT_SOURCE_DIR}/../../../Include/highfive"
    )

    # Check if header-only HDF5 wrappers exist
    set(HDF5_HEADERS_FOUND TRUE)
    foreach(HDF5_DIR ${HDF5_HEADER_DIRS})
        if(NOT EXISTS ${HDF5_DIR})
            set(HDF5_HEADERS_FOUND FALSE)
            break()
        endif()
    endforeach()

    if(HDF5_FOUND AND HDF5_HEADERS_FOUND)
        # Use vcpkg HDF5 library + local header-only wrappers
        target_link_libraries(OScofoLib PRIVATE ${HDF5_LIBRARIES})
        target_include_directories(OScofoLib PRIVATE 
            ${HDF5_INCLUDE_DIRS}
            ${HDF5_HEADER_DIRS}
        )
        target_compile_definitions(OScofoLib PRIVATE 
            # COMPILE_OSCOFO_WITH_LOGGER=1
            H5_BUILT_AS_DYNAMIC_LIB
        )
        message(STATUS "Found HDF5 library: ${HDF5_LIBRARIES}")
        message(STATUS "Using header-only HDF5 wrappers from Include folder")
    elseif(HDF5_FOUND)
        # Only vcpkg HDF5 library available
        target_link_libraries(OScofoLib PRIVATE ${HDF5_LIBRARIES})
        target_include_directories(OScofoLib PRIVATE ${HDF5_INCLUDE_DIRS})
        target_compile_definitions(OScofoLib PRIVATE 
            # COMPILE_OSCOFO_WITH_LOGGER=1
            H5_BUILT_AS_DYNAMIC_LIB
        )
        message(STATUS "Found HDF5 library: ${HDF5_LIBRARIES}")
        message(WARNING "Header-only HDF5 wrappers not found in Include folder")
    else()
        target_compile_definitions(OScofoLib PRIVATE COMPILE_OSCOFO_WITH_LOGGER=0)
        message(STATUS "HDF5 not found - logging features disabled")
    endif()
endif()

# Include tree-sitter if needed
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Libraries/tree-sitter")
    target_include_directories(OScofoLib PRIVATE
        ../Libraries/tree-sitter/lib/include
    )
    # Add tree-sitter source files if they exist
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Libraries/tree-sitter/lib/src/lib.c")
        target_sources(OScofoLib PRIVATE
            ../Libraries/tree-sitter/lib/src/lib.c
        )
    endif()
    
    # Add parser if it exists
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../Resources/tree-sitter-scofo/src/parser.c")
        target_sources(OScofoLib PRIVATE
            ../Resources/tree-sitter-scofo/src/parser.c
        )
    endif()
endif()

# Install configuration
install(TARGETS OScofoLib
    EXPORT OScofoLibTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(FILES
    OScofo.hpp
    OScofo/mdp.hpp
    OScofo/mir.hpp
    OScofo/score.hpp
    OScofo/states.hpp
    OScofo/log.hpp
    ${CMAKE_CURRENT_BINARY_DIR}/OScofo/oscofo_export.h
    DESTINATION include/OScofo
)

# install(DIRECTORY ../Libraries/tree-sitter/lib/include/tree_sitter/ DESTINATION include/tree-sitter)

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../Libraries/tree-sitter/lib/include/tree_sitter/"
        DESTINATION "include/tree_sitter"
        FILES_MATCHING PATTERN "*.h")

install(EXPORT OScofoLibTargets
    FILE OScofoTargets.cmake
    NAMESPACE OScofo::
    DESTINATION lib/cmake/OScofo
)

# Create config file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    OScofoConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/OScofoConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/OScofoConfig.cmake"
    INSTALL_DESTINATION lib/cmake/OScofo
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/OScofoConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/OScofoConfigVersion.cmake"
    DESTINATION lib/cmake/OScofo
)

# Copy Audio folder to build output directory when building standalone
add_custom_command(TARGET OScofoLib POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../Audio
    $<TARGET_FILE_DIR:OScofoLib>/Audio
    COMMENT "Copying Audio folder to build output directory"
)