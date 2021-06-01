cmake_minimum_required(VERSION 3.5)
project (Robustness)


#set (CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(eigen)
#link_directories(networkit/build)
#include_directories(networkit/include)
find_package(OpenMP)

add_subdirectory(networkit)


add_executable(robustness main cpp/greedy.cpp cpp/laplacian.cpp cpp/robustnessGreedy.cpp cpp/robustnessUSTGreedy.cpp)

target_compile_options(robustness PUBLIC -g -O1 -fno-omit-frame-pointer)
target_compile_options(networkit PUBLIC -Wno-deprecated-declarations -g -O1 -fno-omit-frame-pointer)

target_include_directories(robustness PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(robustness INTERFACE )
target_link_libraries(robustness PUBLIC networkit)
target_link_libraries(robustness PUBLIC OpenMP::OpenMP_CXX)


set_target_properties(networkit
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)

set_target_properties(robustness
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED YES
        CXX_EXTENSIONS NO
)