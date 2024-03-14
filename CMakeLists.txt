cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(ByteTransformer LANGUAGES CXX CUDA)

option(BUILD_THS "Build in TorchScript class mode" OFF)

add_definitions("-DCUTLASS_ATTENTION")

set(PYTHON_PATH "python3" CACHE STRING "Python path")

# find cuda
find_package(CUDA)
set(CUDA_HOME ${CUDA_TOOLKIT_ROOT_DIR} CACHE STRING "CUDA home path")
list(APPEND CMAKE_MODULE_PATH ${CUDA_HOME}/lib)
list(APPEND CMAKE_MODULE_PATH ${CUDA_HOME}/lib64)
find_package(CUDA REQUIRED)
set(CUDA_PATH ${CUDA_HOME})

if (${CUDA_VERSION} GREATER_EQUAL 11.0)
  message(STATUS "Add DCUDA11_MODE")
  add_definitions("-DCUDA11_MODE")
endif()


set(CUDAARCHS "80" CACHE STRING "CUDA Architectures")
set(CMAKE_CUDA_ARCHITECTURES ${CUDAARCHS})
message(STATUS "CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")

# setting compiler flags
if (DataType STREQUAL FP16)
	set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DFP16")
	set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DFP16")
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DFP16")
    message("-- Set the data type as FP16 ")
else()
    message("-- Set the data type as FP32 ")
endif()

set(CUDA_ARCH_FLAGS)
foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
  list(APPEND CUDA_ARCH_FLAGS "-gencode=arch=compute_${ARCH},code=\\\"sm_${ARCH},compute_${ARCH}\\\"")
endforeach()

string(JOIN " " JOINED_CUDA_ARCH_FLAGS "${CUDA_ARCH_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true ${JOINED_CUDA_ARCH_FLAGS}")
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDA_PATH}/include
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/include
  ${PROJECT_SOURCE_DIR}/3rdparty/cutlass/tools/util/include
  ${PROJECT_SOURCE_DIR}/cutlass_contrib/include
)

set(COMMON_LIB_DIRS
  ${CUDA_PATH}/lib
  ${CUDA_PATH}/lib64
)

if(BUILD_THS)
  set(TORCH_CUDA_ARCH_LIST)
  foreach(ARCH ${CMAKE_CUDA_ARCHITECTURES})
    if(ARCH STREQUAL "90")
      list(APPEND TORCH_CUDA_ARCH_LIST "9.0")
    elseif(ARCH STREQUAL "80")
      list(APPEND TORCH_CUDA_ARCH_LIST "8.0")
    elseif(ARCH STREQUAL "75")
      list(APPEND TORCH_CUDA_ARCH_LIST "7.5")
    elseif(ARCH STREQUAL "70")
      list(APPEND TORCH_CUDA_ARCH_LIST "7.0")
    else()
      message(WARNING "Unsupported CUDA arch [${ARCH}] for TORCH_CUDA_ARCH")
    endif()
  endforeach()

  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_DIR)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Torch config Error.")
  endif()
  list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
  find_package(Torch REQUIRED)

  execute_process(COMMAND ${PYTHON_PATH} "-c" "from __future__ import print_function; from distutils import sysconfig;
print(sysconfig.get_python_inc());
print(sysconfig.get_config_var('EXT_SUFFIX'));"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE _PYTHON_VALUES)
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "Python config Error.")
  endif()
  string(REGEX REPLACE ";" "\\\\;" _PYTHON_VALUES ${_PYTHON_VALUES})
  string(REGEX REPLACE "\n" ";" _PYTHON_VALUES ${_PYTHON_VALUES})
  list(GET _PYTHON_VALUES 0 PY_INCLUDE_DIR)
  list(GET _PYTHON_VALUES 1 PY_SUFFIX)
  list(APPEND COMMON_HEADER_DIRS ${PY_INCLUDE_DIR})

  execute_process(COMMAND ${PYTHON_PATH} "-c"
                  "from torch.utils import cpp_extension; import re; import torch;                                    \
                   version = tuple(int(i) for i in re.match('(\\d+)\\.(\\d+)\\.(\\d+)', torch.__version__).groups()); \
                   args = ([],True,False,False) if version >= (1, 8, 0) else ([],True,False);                         \
                   print(' '.join(cpp_extension._prepare_ldflags(*args)),end='');"
                  RESULT_VARIABLE _PYTHON_SUCCESS
                  OUTPUT_VARIABLE TORCH_LINK)
  message("-- TORCH_LINK ${TORCH_LINK}")
  if (NOT _PYTHON_SUCCESS MATCHES 0)
      message(FATAL_ERROR "PyTorch link config Error.")
  endif()
  message("CMAKE_CUDA_FLAGS after torch: ${CMAKE_CUDA_FLAGS}")
endif()

include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_subdirectory(bytetransformer)
add_subdirectory(unit_test)

if(BUILD_THS)
  set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
