cmake_minimum_required(VERSION 3.8)
add_subdirectory(src)

if(BUILD_THS)
    add_subdirectory(th_op)
endif()
