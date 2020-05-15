cmake_minimum_required(VERSION 3.5)
project (Robustness)

# Libraries
include_directories(eigen)

#set(PROJECT_LINK_LIBS networkit/libnetworkit.so)
#link_directories(networkit/build)
include_directories(networkit/include)

find_package(OpenMP)

add_subdirectory(networkit)

add_compile_options(-Wno-deprecated)

add_executable(robustness main)
#target_link_libraries(robustness ${PROJECT_LINK_LIBS} )
target_link_libraries(robustness networkit)
target_link_libraries(robustness OpenMP::OpenMP_CXX)