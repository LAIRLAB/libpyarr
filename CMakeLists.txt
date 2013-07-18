cmake_minimum_required(VERSION 2.4.6)

#set root of repository variable
set(REPO_ROOT_RELATIVE "../")
get_filename_component(REPO_ROOT ${REPO_ROOT_RELATIVE} ABSOLUTE)
message("Root of repository: " ${REPO_ROOT})

include(${REPO_ROOT}/common.cmake)
include(${REPO_ROOT}/boost_util.cmake)

add_library(boost_common
            boost_common.cc
            ob/OB_CPP_release/mymex.cc 
            misc.cpp)

set_target_properties(boost_common PROPERTIES SUFFIX ".so")
target_link_libraries(boost_common ${Boost_LIBRARIES} ${PYTHON_LIBRARIES}
                                   misc_util
                                   ml_base
                                   v_regressor
                                   pclassifier
                                   regressor
                                   data_2d
                                   IL
                                   ILU
                                   ale)
add_library(img_util
  img_util.cc)

set_target_properties(img_util PROPERTIES SUFFIX .so)
target_link_libraries(img_util ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} boost_common)


  

  