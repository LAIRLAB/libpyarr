cmake_minimum_required(VERSION 2.4.6)

include(common.cmake)
include(boost_util.cmake)

add_library(boost_common
            boost_common.cc
            mymex.cc 
            autogen_converters.cpp
            misc.cpp)

set(CMAKE_BUILD_TYPE Debug)

# Add -DDEBUG if you want range checking on your pyarrs
set(CMAKE_CXX_FLAGS "-O3") # -DDEBUG")

set_target_properties(boost_common PROPERTIES SUFFIX ".so")
target_link_libraries(boost_common ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
