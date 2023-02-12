cmake_minimum_required(VERSION 3.8)

set(bert_transformer_files
	bert_transformer_test.cc)
add_executable(bert_transformer_test ${bert_transformer_files})
target_link_libraries(bert_transformer_test PUBLIC -lcublas -lcudart bytetransformer ${CMAKE_THREAD_LIBS_INIT})

add_custom_target(copy_ut_scripts ALL COMMENT "Copying unit_test scripts")
add_custom_command(TARGET copy_ut_scripts
  POST_BUILD
  COMMAND cp ${CMAKE_CURRENT_SOURCE_DIR}/python_scripts/*.py ${PROJECT_BINARY_DIR}
)
