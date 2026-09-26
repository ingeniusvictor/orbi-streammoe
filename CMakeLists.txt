cmake_minimum_required(VERSION 3.24)

project(orbi_streammoe
  VERSION 0.1.0
  DESCRIPTION "Portable streamed-MoE inference runtime research"
  LANGUAGES C CXX
)

option(ORBI_STREAMMOE_BUILD_TESTS "Build ORBI StreamMoE tests" ON)
option(ORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP "Enable pinned mlc-ai/tokenizers-cpp backend" OFF)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

if(ORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP)
  set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(
    orbi_tokenizers_cpp
    GIT_REPOSITORY https://github.com/mlc-ai/tokenizers-cpp.git
    GIT_TAG c586c52f93f7b060753bd2388eb96a105cb7374d
    GIT_SHALLOW FALSE
    GIT_SUBMODULES msgpack sentencepiece
    GIT_SUBMODULES_RECURSE TRUE
  )
  FetchContent_MakeAvailable(orbi_tokenizers_cpp)
endif()

set(VULKAN_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  VulkanHeaders
  GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
  GIT_TAG v1.4.363
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(VulkanHeaders)

set(VOLK_PULL_IN_VULKAN ON CACHE BOOL "" FORCE)
set(VOLK_INSTALL OFF CACHE BOOL "" FORCE)
set(VOLK_HEADERS_ONLY OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  volk
  GIT_REPOSITORY https://github.com/zeux/volk.git
  GIT_TAG vulkan-sdk-1.4.363
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(volk)

add_library(orbi_streammoe_core
  src/core/version.cpp
  src/container/qpack.cpp
  src/container/safetensors.cpp
  src/container/qpack_dense.cpp
  src/container/mlx_affine_checkpoint.cpp
  src/cpu/reference_ops.cpp
  src/model/qwen_router.cpp
  src/model/routed_moe.cpp
  src/model/qwen_dense_binding.cpp
  src/model/checkpoint_sparse_moe.cpp
  src/model/checkpoint_moe_sublayer.cpp
  src/model/gated_deltanet_cpu.cpp
  src/model/checkpoint_gated_deltanet.cpp
  src/model/checkpoint_deltanet_sublayer.cpp
  src/model/checkpoint_linear_decoder_layer.cpp
  src/model/gqa_cpu.cpp
  src/model/checkpoint_gqa.cpp
  src/model/checkpoint_gqa_sublayer.cpp
  src/model/checkpoint_full_attention_decoder_layer.cpp
  src/model/checkpoint_decoder_layer.cpp
  src/model/checkpoint_decoder_stack.cpp
  src/model/qwen_global_binding.cpp
  src/model/qwen_global_streaming.cpp
  src/model/checkpoint_model_shell.cpp
  src/model/sharded_checkpoint_contract.cpp
  src/model/shard_header_contract.cpp
  src/session/greedy_token_session.cpp
  src/session/text_tokenizer_boundary.cpp
  src/session/greedy_text_session.cpp
  src/tokenizer/qwen_tokenizer_assets.cpp
  src/tokenizer/tokenizers_cpp_qwen_tokenizer.cpp
  src/model/shared_expert.cpp
  src/cache/expert_cache.cpp
  src/storage/qpack_expert_storage.cpp
  src/backend/vulkan_runtime.cpp
  src/backend/vulkan_compute_context.cpp
  src/backend/vulkan_compute_roundtrip.cpp
  src/backend/vulkan_rmsnorm.cpp
  src/backend/vulkan_q4_dequant.cpp
  src/backend/vulkan_q4_gemv.cpp
  src/backend/vulkan_qpack_expert.cpp
  src/backend/vulkan_qpack_expert_mlp.cpp
  src/backend/vulkan_swiglu.cpp
  src/backend/vulkan_float_buffer.cpp
  src/backend/vulkan_q4_weights.cpp
  src/backend/vulkan_resident_expert.cpp
  src/backend/vulkan_resident_expert_cache.cpp
  src/backend/vulkan_resident_shared_expert.cpp
  src/backend/vulkan_weighted_accumulate.cpp
)

if(WIN32)
  target_sources(orbi_streammoe_core PRIVATE
    src/storage/windows_overlapped_expert_storage.cpp
  )
else()
  find_package(Threads REQUIRED)
  target_sources(orbi_streammoe_core PRIVATE
    src/storage/posix_pread_expert_storage.cpp
  )
  target_link_libraries(orbi_streammoe_core PRIVATE Threads::Threads)
endif()

target_include_directories(orbi_streammoe_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(orbi_streammoe_core PUBLIC cxx_std_20)
target_link_libraries(orbi_streammoe_core PRIVATE nlohmann_json::nlohmann_json volk::volk)

if(ORBI_STREAMMOE_ENABLE_TOKENIZERS_CPP)
  target_compile_definitions(orbi_streammoe_core PRIVATE ORBI_STREAMMOE_HAS_TOKENIZERS_CPP=1)
  target_link_libraries(orbi_streammoe_core PRIVATE tokenizers_cpp tokenizers_c)
endif()

if(MSVC)
  target_compile_options(orbi_streammoe_core PRIVATE /W4 /permissive-)
else()
  target_compile_options(orbi_streammoe_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(ORBI_STREAMMOE_BUILD_TESTS)
  enable_testing()

  add_executable(orbi_streammoe_smoke
    tests/smoke.cpp
  )
  target_link_libraries(orbi_streammoe_smoke PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_smoke PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qpack_tests
    tests/qpack_test.cpp
  )
  target_link_libraries(orbi_streammoe_qpack_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_qpack_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_safetensors_dense_tests
    tests/safetensors_dense_test.cpp
  )
  target_link_libraries(orbi_streammoe_safetensors_dense_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_safetensors_dense_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_mlx_affine_checkpoint_tests
    tests/mlx_affine_checkpoint_test.cpp
  )
  target_link_libraries(orbi_streammoe_mlx_affine_checkpoint_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_mlx_affine_checkpoint_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qwen_dense_binding_tests
    tests/qwen_dense_binding_test.cpp
  )
  target_link_libraries(orbi_streammoe_qwen_dense_binding_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_qwen_dense_binding_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_sparse_moe_tests
    tests/checkpoint_sparse_moe_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_sparse_moe_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_sparse_moe_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_moe_sublayer_tests
    tests/checkpoint_moe_sublayer_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_moe_sublayer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_moe_sublayer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_gated_deltanet_cpu_tests
    tests/gated_deltanet_cpu_test.cpp
  )
  target_link_libraries(orbi_streammoe_gated_deltanet_cpu_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_gated_deltanet_cpu_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_gated_deltanet_tests
    tests/checkpoint_gated_deltanet_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_gated_deltanet_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_gated_deltanet_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_deltanet_sublayer_tests
    tests/checkpoint_deltanet_sublayer_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_deltanet_sublayer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_deltanet_sublayer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_linear_decoder_layer_tests
    tests/checkpoint_linear_decoder_layer_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_linear_decoder_layer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_linear_decoder_layer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_gqa_cpu_tests
    tests/gqa_cpu_test.cpp
  )
  target_link_libraries(orbi_streammoe_gqa_cpu_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_gqa_cpu_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_gqa_tests
    tests/checkpoint_gqa_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_gqa_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_gqa_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_gqa_sublayer_tests
    tests/checkpoint_gqa_sublayer_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_gqa_sublayer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_gqa_sublayer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_full_attention_decoder_layer_tests
    tests/checkpoint_full_attention_decoder_layer_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_full_attention_decoder_layer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_full_attention_decoder_layer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_decoder_layer_tests
    tests/checkpoint_decoder_layer_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_decoder_layer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_decoder_layer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_decoder_stack_tests
    tests/checkpoint_decoder_stack_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_decoder_stack_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_decoder_stack_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qwen_global_binding_tests
    tests/qwen_global_binding_test.cpp
  )
  target_link_libraries(orbi_streammoe_qwen_global_binding_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_qwen_global_binding_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qwen_global_streaming_tests
    tests/qwen_global_streaming_test.cpp
  )
  target_link_libraries(orbi_streammoe_qwen_global_streaming_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_qwen_global_streaming_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_checkpoint_model_shell_tests
    tests/checkpoint_model_shell_test.cpp
  )
  target_link_libraries(orbi_streammoe_checkpoint_model_shell_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_checkpoint_model_shell_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_sharded_checkpoint_contract_tests
    tests/sharded_checkpoint_contract_test.cpp
  )
  target_link_libraries(orbi_streammoe_sharded_checkpoint_contract_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_sharded_checkpoint_contract_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_official_checkpoint_metadata_tests
    tests/official_checkpoint_metadata_test.cpp
  )
  target_link_libraries(orbi_streammoe_official_checkpoint_metadata_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_official_checkpoint_metadata_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_shard_header_contract_tests
    tests/shard_header_contract_test.cpp
  )
  target_link_libraries(orbi_streammoe_shard_header_contract_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_shard_header_contract_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_official_shard_range_tests
    tests/official_shard_range_test.cpp
  )
  target_link_libraries(orbi_streammoe_official_shard_range_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_official_shard_range_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_greedy_token_session_tests
    tests/greedy_token_session_test.cpp
  )
  target_link_libraries(orbi_streammoe_greedy_token_session_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_greedy_token_session_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_text_tokenizer_boundary_tests
    tests/text_tokenizer_boundary_test.cpp
  )
  target_link_libraries(orbi_streammoe_text_tokenizer_boundary_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_text_tokenizer_boundary_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_greedy_text_session_tests
    tests/greedy_text_session_test.cpp
  )
  target_link_libraries(orbi_streammoe_greedy_text_session_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_greedy_text_session_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qwen_tokenizer_assets_tests
    tests/qwen_tokenizer_assets_test.cpp
  )
  target_link_libraries(orbi_streammoe_qwen_tokenizer_assets_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_qwen_tokenizer_assets_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_tokenizers_cpp_qwen_tokenizer_tests
    tests/tokenizers_cpp_qwen_tokenizer_test.cpp
  )
  target_link_libraries(orbi_streammoe_tokenizers_cpp_qwen_tokenizer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_tokenizers_cpp_qwen_tokenizer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qwen_official_tokenizer_parity_tests
    tests/qwen_official_tokenizer_parity_test.cpp
  )
  target_link_libraries(orbi_streammoe_qwen_official_tokenizer_parity_tests PRIVATE orbi_streammoe_core nlohmann_json::nlohmann_json)
  target_compile_features(orbi_streammoe_qwen_official_tokenizer_parity_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_cpu_reference_tests
    tests/cpu_reference_test.cpp
  )
  target_link_libraries(orbi_streammoe_cpu_reference_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_cpu_reference_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_expert_cache_tests
    tests/expert_cache_test.cpp
  )
  target_link_libraries(orbi_streammoe_expert_cache_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_expert_cache_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qwen_router_tests
    tests/qwen_router_test.cpp
  )
  target_link_libraries(orbi_streammoe_qwen_router_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_qwen_router_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_weighted_routed_moe_tests
    tests/weighted_routed_moe_test.cpp
  )
  target_link_libraries(orbi_streammoe_weighted_routed_moe_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_weighted_routed_moe_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_shared_expert_tests
    tests/shared_expert_test.cpp
  )
  target_link_libraries(orbi_streammoe_shared_expert_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_shared_expert_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_routed_moe_accumulate_tests
    tests/vulkan_routed_moe_accumulate_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_routed_moe_accumulate_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_routed_moe_accumulate_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_shared_expert_tests
    tests/vulkan_shared_expert_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_shared_expert_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_shared_expert_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_runtime_tests
    tests/vulkan_runtime_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_runtime_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_runtime_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_compute_context_tests
    tests/vulkan_compute_context_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_compute_context_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_compute_context_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_compute_roundtrip_tests
    tests/vulkan_compute_roundtrip_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_compute_roundtrip_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_compute_roundtrip_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_rmsnorm_tests
    tests/vulkan_rmsnorm_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_rmsnorm_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_rmsnorm_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_q4_dequant_tests
    tests/vulkan_q4_dequant_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_q4_dequant_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_q4_dequant_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_q4_gemv_tests
    tests/vulkan_q4_gemv_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_q4_gemv_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_q4_gemv_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_qpack_expert_tests
    tests/vulkan_qpack_expert_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_qpack_expert_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_qpack_expert_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_qpack_metadata_dtype_tests
    tests/qpack_metadata_dtype_test.cpp
  )
  target_link_libraries(orbi_streammoe_qpack_metadata_dtype_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_qpack_metadata_dtype_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_qpack_expert_mlp_tests
    tests/vulkan_qpack_expert_mlp_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_qpack_expert_mlp_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_qpack_expert_mlp_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_swiglu_tests
    tests/vulkan_swiglu_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_swiglu_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_swiglu_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_float_buffer_tests
    tests/vulkan_float_buffer_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_float_buffer_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_float_buffer_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_persistent_ops_tests
    tests/vulkan_persistent_ops_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_persistent_ops_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_persistent_ops_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_preloaded_q4_weights_tests
    tests/vulkan_preloaded_q4_weights_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_preloaded_q4_weights_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_preloaded_q4_weights_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_resident_expert_tests
    tests/vulkan_resident_expert_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_resident_expert_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_resident_expert_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_resident_output_tests
    tests/vulkan_resident_output_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_resident_output_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_resident_output_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_weighted_accumulate_tests
    tests/vulkan_weighted_accumulate_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_weighted_accumulate_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_weighted_accumulate_tests PRIVATE cxx_std_20)

  add_executable(orbi_streammoe_vulkan_resident_expert_cache_tests
    tests/vulkan_resident_expert_cache_test.cpp
  )
  target_link_libraries(orbi_streammoe_vulkan_resident_expert_cache_tests PRIVATE orbi_streammoe_core)
  target_compile_features(orbi_streammoe_vulkan_resident_expert_cache_tests PRIVATE cxx_std_20)

  add_test(NAME orbi_streammoe_smoke COMMAND orbi_streammoe_smoke)
  add_test(NAME orbi_streammoe_qpack_tests COMMAND orbi_streammoe_qpack_tests)
  add_test(NAME orbi_streammoe_safetensors_dense_tests COMMAND orbi_streammoe_safetensors_dense_tests)
  add_test(NAME orbi_streammoe_mlx_affine_checkpoint_tests COMMAND orbi_streammoe_mlx_affine_checkpoint_tests)
  add_test(NAME orbi_streammoe_qwen_dense_binding_tests COMMAND orbi_streammoe_qwen_dense_binding_tests)
  add_test(NAME orbi_streammoe_checkpoint_sparse_moe_tests COMMAND orbi_streammoe_checkpoint_sparse_moe_tests)
  add_test(NAME orbi_streammoe_checkpoint_moe_sublayer_tests COMMAND orbi_streammoe_checkpoint_moe_sublayer_tests)
  add_test(NAME orbi_streammoe_gated_deltanet_cpu_tests COMMAND orbi_streammoe_gated_deltanet_cpu_tests)
  add_test(NAME orbi_streammoe_checkpoint_gated_deltanet_tests COMMAND orbi_streammoe_checkpoint_gated_deltanet_tests)
  add_test(NAME orbi_streammoe_checkpoint_deltanet_sublayer_tests COMMAND orbi_streammoe_checkpoint_deltanet_sublayer_tests)
  add_test(NAME orbi_streammoe_checkpoint_linear_decoder_layer_tests COMMAND orbi_streammoe_checkpoint_linear_decoder_layer_tests)
  add_test(NAME orbi_streammoe_gqa_cpu_tests COMMAND orbi_streammoe_gqa_cpu_tests)
  add_test(NAME orbi_streammoe_checkpoint_gqa_tests COMMAND orbi_streammoe_checkpoint_gqa_tests)
  add_test(NAME orbi_streammoe_checkpoint_gqa_sublayer_tests COMMAND orbi_streammoe_checkpoint_gqa_sublayer_tests)
  add_test(NAME orbi_streammoe_checkpoint_full_attention_decoder_layer_tests COMMAND orbi_streammoe_checkpoint_full_attention_decoder_layer_tests)
  add_test(NAME orbi_streammoe_checkpoint_decoder_layer_tests COMMAND orbi_streammoe_checkpoint_decoder_layer_tests)
  add_test(NAME orbi_streammoe_checkpoint_decoder_stack_tests COMMAND orbi_streammoe_checkpoint_decoder_stack_tests)
  add_test(NAME orbi_streammoe_qwen_global_binding_tests COMMAND orbi_streammoe_qwen_global_binding_tests)
  add_test(NAME orbi_streammoe_qwen_global_streaming_tests COMMAND orbi_streammoe_qwen_global_streaming_tests)
  add_test(NAME orbi_streammoe_checkpoint_model_shell_tests COMMAND orbi_streammoe_checkpoint_model_shell_tests)
  add_test(NAME orbi_streammoe_sharded_checkpoint_contract_tests COMMAND orbi_streammoe_sharded_checkpoint_contract_tests)
  add_test(NAME orbi_streammoe_shard_header_contract_tests COMMAND orbi_streammoe_shard_header_contract_tests)
  add_test(NAME orbi_streammoe_greedy_token_session_tests COMMAND orbi_streammoe_greedy_token_session_tests)
  add_test(NAME orbi_streammoe_text_tokenizer_boundary_tests COMMAND orbi_streammoe_text_tokenizer_boundary_tests)
  add_test(NAME orbi_streammoe_greedy_text_session_tests COMMAND orbi_streammoe_greedy_text_session_tests)
  add_test(NAME orbi_streammoe_qwen_tokenizer_assets_tests COMMAND orbi_streammoe_qwen_tokenizer_assets_tests)
  add_test(NAME orbi_streammoe_tokenizers_cpp_qwen_tokenizer_tests COMMAND orbi_streammoe_tokenizers_cpp_qwen_tokenizer_tests)
  add_test(NAME orbi_streammoe_qwen_official_tokenizer_parity_tests COMMAND orbi_streammoe_qwen_official_tokenizer_parity_tests)
  add_test(NAME orbi_streammoe_cpu_reference_tests COMMAND orbi_streammoe_cpu_reference_tests)
  add_test(NAME orbi_streammoe_expert_cache_tests COMMAND orbi_streammoe_expert_cache_tests)
  add_test(NAME orbi_streammoe_qwen_router_tests COMMAND orbi_streammoe_qwen_router_tests)
  add_test(NAME orbi_streammoe_weighted_routed_moe_tests COMMAND orbi_streammoe_weighted_routed_moe_tests)
  add_test(NAME orbi_streammoe_shared_expert_tests COMMAND orbi_streammoe_shared_expert_tests)
  add_test(NAME orbi_streammoe_vulkan_routed_moe_accumulate_tests COMMAND orbi_streammoe_vulkan_routed_moe_accumulate_tests)
  add_test(NAME orbi_streammoe_vulkan_shared_expert_tests COMMAND orbi_streammoe_vulkan_shared_expert_tests)
  add_test(NAME orbi_streammoe_vulkan_runtime_tests COMMAND orbi_streammoe_vulkan_runtime_tests)
  add_test(NAME orbi_streammoe_vulkan_compute_context_tests COMMAND orbi_streammoe_vulkan_compute_context_tests)
  add_test(NAME orbi_streammoe_vulkan_compute_roundtrip_tests COMMAND orbi_streammoe_vulkan_compute_roundtrip_tests)
  add_test(NAME orbi_streammoe_vulkan_rmsnorm_tests COMMAND orbi_streammoe_vulkan_rmsnorm_tests)
  add_test(NAME orbi_streammoe_vulkan_q4_dequant_tests COMMAND orbi_streammoe_vulkan_q4_dequant_tests)
  add_test(NAME orbi_streammoe_vulkan_q4_gemv_tests COMMAND orbi_streammoe_vulkan_q4_gemv_tests)
  add_test(NAME orbi_streammoe_vulkan_qpack_expert_tests COMMAND orbi_streammoe_vulkan_qpack_expert_tests)
  add_test(NAME orbi_streammoe_qpack_metadata_dtype_tests COMMAND orbi_streammoe_qpack_metadata_dtype_tests)
  add_test(NAME orbi_streammoe_vulkan_qpack_expert_mlp_tests COMMAND orbi_streammoe_vulkan_qpack_expert_mlp_tests)
  add_test(NAME orbi_streammoe_vulkan_swiglu_tests COMMAND orbi_streammoe_vulkan_swiglu_tests)
  add_test(NAME orbi_streammoe_vulkan_float_buffer_tests COMMAND orbi_streammoe_vulkan_float_buffer_tests)
  add_test(NAME orbi_streammoe_vulkan_persistent_ops_tests COMMAND orbi_streammoe_vulkan_persistent_ops_tests)
  add_test(NAME orbi_streammoe_vulkan_preloaded_q4_weights_tests COMMAND orbi_streammoe_vulkan_preloaded_q4_weights_tests)
  add_test(NAME orbi_streammoe_vulkan_resident_expert_tests COMMAND orbi_streammoe_vulkan_resident_expert_tests)
  add_test(NAME orbi_streammoe_vulkan_resident_output_tests COMMAND orbi_streammoe_vulkan_resident_output_tests)
  add_test(NAME orbi_streammoe_vulkan_weighted_accumulate_tests COMMAND orbi_streammoe_vulkan_weighted_accumulate_tests)
  add_test(NAME orbi_streammoe_vulkan_resident_expert_cache_tests COMMAND orbi_streammoe_vulkan_resident_expert_cache_tests)

  if(WIN32)
    add_executable(orbi_streammoe_windows_io_tests
      tests/windows_overlapped_storage_test.cpp
    )
    target_link_libraries(orbi_streammoe_windows_io_tests PRIVATE orbi_streammoe_core)
    target_compile_features(orbi_streammoe_windows_io_tests PRIVATE cxx_std_20)

    add_test(
      NAME orbi_streammoe_windows_io_tests
      COMMAND orbi_streammoe_windows_io_tests
    )
  else()
    add_executable(orbi_streammoe_posix_io_tests
      tests/posix_pread_storage_test.cpp
    )
    target_link_libraries(orbi_streammoe_posix_io_tests PRIVATE orbi_streammoe_core)
    target_compile_features(orbi_streammoe_posix_io_tests PRIVATE cxx_std_20)

    add_test(
      NAME orbi_streammoe_posix_io_tests
      COMMAND orbi_streammoe_posix_io_tests
    )
  endif()
endif()
