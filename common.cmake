

if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(CMAKE_BUILD_TYPE Debug)

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

set(LIBPYARR $ENV{LIBPYARR_ROOT})
include_directories(${LIBPYARR}/
  ${LIBPYARR}/include)

if (NOT APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3 -g -DUNIX")
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g -DUNIX")
endif()

if ($ENV{CMAKE_USE_FLOATS} STREQUAL "YES")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_FLOATS")
endif()


include(${LIBPYARR}/boost_util.cmake)

link_directories(${LIBPYARR}/lib)
