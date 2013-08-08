

if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_BUILD_TYPE Debug)

set(HOME_DIR ENV{HOME})

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(LINK_PYTHON_THREADS TRUE)

#verbosity
set(CMAKE_VERBOSE_MAKEFILE TRUE)

#set output paths
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_FIND_LIBRARY_SUFFIXES ".dylib;.so;.a")

set(BUILD_SHARED_LIBS ON)
set(CMAKE_COMPILER_IS_GNUCXX ON)

include_directories(${REPO_ROOT}/misc_util/include
${REPO_ROOT}/common
${REPO_ROOT}/hog
${REPO_ROOT}/3rd_party/misc/eigen3/include
${REPO_ROOT}/scene_analysis/scene_analysis_2d/include/
${REPO_ROOT}/3rd_party/misc/yaml_cpp_027/build/yaml-cpp-0.2.7/include/
${REPO_ROOT}/3rd_party/misc/eigen3/build/eigen-eigen-6e7488e20373/
${REPO_ROOT}/clustering/cluster/include
${REPO_ROOT}/3rd_party/ml/ale/modified
${REPO_ROOT}/3rd_party/ml/kmeanspp/include
${REPO_ROOT}/feature_quantizer/include
${REPO_ROOT}/vision_2d/data_2d/include
${REPO_ROOT}/3rd_party/ml/pf/include
${REPO_ROOT}/3rd_party/ml/kdtree_ru/include
${REPO_ROOT}/vision_2d/grouping_2d/include
${REPO_ROOT}/vision_2d/features_2d/include
${REPO_ROOT}/region_hierarchy/region_2d/include
${REPO_ROOT}/region_hierarchy/region_generic/include
${REPO_ROOT}/struct_predict/infer_machine/include
${REPO_ROOT}/ml/pclassifier/include
${REPO_ROOT}/ml/ml_base/include
${REPO_ROOT}/ml/regressor/include
${REPO_ROOT}/ml/v_regressor/include
${REPO_ROOT}/ml/co_reg/include
${REPO_ROOT}/ml
${REPO_ROOT}/clustering/learn_metric/include
${REPO_ROOT}/clustering/graph_segment/include
${REPO_ROOT}/him_2d/
${REPO_ROOT}/homp/include/
/usr/include/python2.7/
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/include/python2.7/
/Developer/NVIDIA/CUDA-5.0/include/
/opt/local/include/eigen3
/opt/local/include
./include )
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -g -DUNIX")

if ($ENV{CMAKE_USE_FLOATS} STREQUAL "YES")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_FLOATS")
endif()

link_directories(${REPO_ROOT}/misc_util/lib
${REPO_ROOT}/common/lib
${REPO_ROOT}/hog/lib
${REPO_ROOT}/scene_analysis/scene_analysis_2d/lib/
${REPO_ROOT}/3rd_party/misc/yaml_cpp_027/build/yaml-cpp-0.2.7/
${REPO_ROOT}/3rd_party/ml/ale/lib
${REPO_ROOT}/3rd_party/ml/kmeanspp/lib
${REPO_ROOT}/feature_quantizer/lib
${REPO_ROOT}/vision_2d/data_2d/lib
${REPO_ROOT}/vision_2d/features_2d/lib
${REPO_ROOT}/vision_2d/grouping_2d/lib
${REPO_ROOT}/region_hierarchy/region_2d/lib
${REPO_ROOT}/region_hierarchy/region_generic/lib
${REPO_ROOT}/struct_predict/infer_machine/lib
${REPO_ROOT}/ml/pclassifier/lib
${REPO_ROOT}/ml/ml_base/lib
${REPO_ROOT}/ml/regressor/lib
${REPO_ROOT}/ml/v_regressor/lib
${REPO_ROOT}/ml/co_reg/lib
${REPO_ROOT}/ml/lib
${REPO_ROOT}/datasets/dataset_2d/lib
${REPO_ROOT}/3rd_party/ml/kmeanspp/lib
${REPO_ROOT}/clustering/cluster/lib
${REPO_ROOT}/misc_util/lib
${REPO_ROOT}/clustering/learn_metric/lib
${REPO_ROOT}/clustering/graph_segment/lib
${REPO_ROOT}/homp/lib/
${REPO_ROOT}/him_2d/lib/
/Developer/NVIDIA/CUDA-5.0/lib
/opt/local/lib/apple-gcc42/gcc/i686-apple-darwin12/4.2.1/
/usr/llvm-gcc-4.2/lib/gcc/i686-apple-darwin11/4.2.1/
/opt/local/lib
)
