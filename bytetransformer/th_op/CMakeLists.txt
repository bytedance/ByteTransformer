cmake_minimum_required(VERSION 3.13)

set(th_ths_files
	ths_op_f.cc
	ths_op.cc
)

add_definitions(-DTORCH_CUDA=1)

# hack for bugs in torch
if(TARGET torch_cpu)
	set_target_properties(torch_cpu PROPERTIES
						INTERFACE_COMPILE_OPTIONS "")
endif()
if(TARGET torch_cuda)
    set_target_properties(torch_cuda PROPERTIES
                            INTERFACE_COMPILE_OPTIONS "")
    set(NEW_TORCH_CUDA_LINK_VAR)
    get_target_property(OLD_TORCH_CUDA_LINK_VAR torch_cuda INTERFACE_LINK_LIBRARIES)
    foreach (TMPVAR ${OLD_TORCH_CUDA_LINK_VAR})
        string(REPLACE "/usr/local/cuda" "${CUDA_TOOLKIT_ROOT_DIR}" TMPVAR ${TMPVAR})
        list(APPEND NEW_TORCH_CUDA_LINK_VAR ${TMPVAR})
    endforeach(TMPVAR)
    set_target_properties(torch_cuda PROPERTIES
                            INTERFACE_LINK_LIBRARIES "${NEW_TORCH_CUDA_LINK_VAR}")
endif()

set(LIB_NAME_2 "ths_bytetransformer")
add_library(${LIB_NAME_2} SHARED ${th_ths_files})
set_target_properties(${LIB_NAME_2} PROPERTIES
                    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(${LIB_NAME_2} "${TORCH_LIBRARIES}" bytetransformer -lcublas -lcudart -lcublasLt)
