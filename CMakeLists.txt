cmake_minimum_required(VERSION 2.4.6)

#set root of repository variable
set(REPO_ROOT_RELATIVE "..")
get_filename_component(REPO_ROOT ${REPO_ROOT_RELATIVE} ABSOLUTE)
message("Root of repository: " ${REPO_ROOT})

include(${REPO_ROOT}/common.cmake)
include(${REPO_ROOT}/boost_util.cmake)

add_library(boost_common
            boost_common.cc
            OB_CPP_release/mymex.cc 
            misc.cpp)
add_library(obwrap
            obwrap.cc
            OB_CPP_release/mymex.cc
            OB_CPP_release/parserUtil.cc 
            OB_CPP_release/MaxGetSpatialPyramid.cc 
            OB_CPP_release/MaxGetSpatialPyramid_single.cc 
            OB_CPP_release/LSVMModel.cc 
            OB_CPP_release/mymex.cc 
            OB_CPP_release/myfeatures.cc 
            OB_CPP_release/dt.cc 
            OB_CPP_release/fconv.cc 
            OB_CPP_release/resize.cc 
            OB_CPP_release/detect.cpp)

set_target_properties(obwrap PROPERTIES SUFFIX ".so")
target_link_libraries(obwrap 
                      ${Boost_LIBRARIES}
                      ${PYTHON_LIBRARIES}
                      ml_base
                      v_regressor
                      pclassifier
                      regressor
                      jpeg                         
                      misc_util
                      features_2d
                      IL                           
                      ILU
                      ale)

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
                                   ale
                                   features_2d)
