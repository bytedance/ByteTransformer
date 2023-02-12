cmake_minimum_required(VERSION 3.8)

set(lib_files
  gemm.cu
  gemm_bias_act.cu
  remove_padding.cu
  layernorm.cu
  attention_fused.cu
  attention_fused_long.cu
  attention_nofused.cu
  attention_nofused_utils.cu
  bert_transformer.cu
  cutlass_attention.cu
)

set_source_files_properties(cutlass_attention.cu PROPERTIES COMPILE_FLAGS --use_fast_math)

add_library(bytetransformer SHARED ${lib_files})
set_target_properties(bytetransformer PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_link_libraries(bytetransformer PUBLIC -lcublas -lcudart ${CMAKE_THREAD_LIBS_INIT})
set_property(TARGET bytetransformer PROPERTY POSITION_INDEPENDENT_CODE ON)
