cmake_minimum_required(VERSION 3.21)

set(TARGET aidaptiv-demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${TARGET} demo.cpp)
target_link_libraries(${TARGET} PRIVATE llama)
target_compile_definitions(${TARGET} PUBLIC AIDAPTIV_SHARED)
