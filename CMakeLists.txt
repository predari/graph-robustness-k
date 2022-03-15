cmake_minimum_required(VERSION 3.5)
project (Robustness)



# slepc 
# provide: -DSLEPC_DIR=/home/maria/Documents/postdoc/soft/slepc to CMake
target_include_directories(robustness BEFORE PUBLIC ${SLEPC_DIR}/include)
set(SLEPC_LIB ${SLEPC_DIR}/lib)
#find_library(SLEPC_LIB slepc)





#set (CMAKE_CXX_STANDARD 11)
#set (CMAKE_CXX_STANDARD_REQUIRED True)


include_directories(eigen)
# link_directories(networkit/build)
# include_directories(networkit/include)
find_package(OpenMP)

add_subdirectory(networkit)

add_executable(robustness main cpp/greedy.cpp cpp/laplacian.cpp cpp/dynamicLaplacianSolver.cpp cpp/robustnessGreedy.cpp cpp/robustnessUSTGreedy.cpp)

target_compile_options(robustness PUBLIC -g -O3 -fno-omit-frame-pointer)
target_compile_options(networkit PUBLIC -Wno-deprecated-declarations -g -O3 -fno-omit-frame-pointer)

target_include_directories(robustness PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(robustness INTERFACE )


target_link_libraries(robustness PUBLIC networkit)
# TODO: provate or public
target_link_libraries(robustness PRIVATE ${SLEPC_LIB})
target_link_libraries(robustness PUBLIC OpenMP::OpenMP_CXX asan)
# For address sanitizing link to "asan"
# TODO: provate or public
target_link_libraries(robustness PRIVATE asan)

SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-rpath -Wl,${STINGER_LIB}")


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
