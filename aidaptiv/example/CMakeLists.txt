cmake_minimum_required(VERSION 3.21)
project(AIDAPTIV_DEMO CXX)

set(TARGET aidaptiv-demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LIB_ROOT ${CMAKE_SOURCE_DIR}/../)
set(LIB_HEADER_DIR ${LIB_ROOT}/include)
set(LIB_SYMBOL_DIR ${LIB_ROOT}/lib)
set(LIB_BIN_DIR ${LIB_ROOT}/bin)

function(config_library lib_name)
    add_library(${lib_name} SHARED IMPORTED)
    set_target_properties(${lib_name} PROPERTIES
        IMPORTED_LOCATION             ${LIB_BIN_DIR}/${lib_name}.dll
        IMPORTED_IMPLIB               ${LIB_SYMBOL_DIR}/${lib_name}.lib
        INTERFACE_INCLUDE_DIRECTORIES ${LIB_HEADER_DIR}
    )
endfunction()

config_library(aidaptiv)
config_library(llama)
config_library(llama-common)
config_library(ggml)
config_library(ggml-base)
config_library(ggml-cpu)
config_library(ggml-vulkan)

add_executable(${TARGET} demo.cpp)
target_link_libraries(${TARGET} PRIVATE aidaptiv)
target_link_libraries(${TARGET} PRIVATE llama)
target_link_libraries(${TARGET} PRIVATE llama-common)
target_link_libraries(${TARGET} PRIVATE ggml)
target_link_libraries(${TARGET} PRIVATE ggml-base)
target_link_libraries(${TARGET} PRIVATE ggml-cpu)
target_link_libraries(${TARGET} PRIVATE ggml-vulkan)
target_compile_definitions(${TARGET} PUBLIC AIDAPTIV_SHARED)
target_compile_definitions(${TARGET} PUBLIC LLAMA_SHARED)
target_compile_definitions(${TARGET} PUBLIC GGML_SHARED)

add_custom_command(TARGET ${TARGET} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_RUNTIME_DLLS:${TARGET}>
        $<TARGET_FILE_DIR:${TARGET}>
    COMMAND_EXPAND_LISTS
)
