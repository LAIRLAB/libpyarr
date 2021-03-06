#preferred location of boost
set(BOOST_LIBRARYDIR /usr/local/lib)
set(BOOST_INCLUDEDIR /usr/local/include/boost)

find_package(Boost COMPONENTS system thread python REQUIRED)

message("boost version: ${Boost_MINOR_VERSION}")
if (${Boost_VERSION} LESS 104700)
   message("using OLD boost: ${Boost_VERSION}")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DOLD_BOOST")
else()
   message("--> using NEW boost: ${Boost_VERSION}")
endif()

message("Boost includes: ${Boost_INCLUDE_DIRS}")
message("Boost lib dirs: ${Boost_LIBRARY_DIRS}")
message("Boost libs: ${Boost_LIBRARIES}")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/)

#temp hack
#set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)
find_package(PythonLibs 2.7 REQUIRED)
message("Python libs: " ${PYTHON_LIBRARIES})
message("Python include dir: " ${PYTHON_INCLUDE_DIRS})

if (NOT APPLE)
   #set(PYTHON_MIN_VERSION "2.7")
   set(OLD_PYLIBS ${PYTHON_LIBRARIES})
   set(OLD_PYINCDIRS ${PYTHON_INCLUDE_DIRS})
   execute_process(COMMAND "python" "$ENV{LIBPYARR_ROOT}/find_epd.py" "${PYTHON_MIN_VERSION}" OUTPUT_VARIABLE PYTHON_LIBRARIES ERROR_VARIABLE PYTHON_INCLUDE_DIRS)
   if("${PYTHON_LIBRARIES}" STREQUAL "")
     message("ERROR No python libraries found!!!")
     message("Python min version requested: ${PYTHON_MIN_VERSION}")
     message("Reverting to old pylibs: ${OLD_PYLIBS}")
     set(PYTHON_LIBRARIES ${OLD_PYLIBS})
   elseif("${PYTHON_INCLUDE_DIRS}" STREQUAL "") 
     message("ERROR No python includes found!!!")
     message("Python min version requested: ${PYTHON_MIN_VERSION}")
     message("Reverting to old py inc dirs: ${OLD_PYINCDIRS}")
     set(PYTHON_INCLUDE_DIRS ${OLD_PYINCDIRS})
   else()
     message("Python libs: " ${PYTHON_LIBRARIES})
     message("Python include dir: " ${PYTHON_INCLUDE_DIRS})
   endif()
   include_directories(/usr/include/python2.7)
endif()

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/numpy/core/include/)
link_directories(${Boost_LIBRARY_DIRS})

set(Boost_USE_MULTITHREADED ON) 

if(LINK_PYTHON_THREADS)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DLINK_PYTHON_THREADS")
endif()


