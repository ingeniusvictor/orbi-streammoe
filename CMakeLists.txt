cmake_minimum_required(VERSION 3.24)

project(orbi_streammoe
  VERSION 0.1.0
  DESCRIPTION "Portable streamed-MoE inference runtime research"
  LANGUAGES CXX
)

option(ORBI_STREAMMOE_BUILD_TESTS "Build ORBI StreamMoE tests" ON)

include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_library(orbi_streammoe_core
  src/core/version.cpp
  src/container/qpack.cpp
  src/cpu/reference_ops.cpp
  src/cache/expert_cache.cpp
  src/storage/qpack_expert_storage.cpp
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
target_link_libraries(orbi_streammoe_core PRIVATE nlohmann_json::nlohmann_json)

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

  add_test(NAME orbi_streammoe_smoke COMMAND orbi_streammoe_smoke)
  add_test(NAME orbi_streammoe_qpack_tests COMMAND orbi_streammoe_qpack_tests)
  add_test(NAME orbi_streammoe_cpu_reference_tests COMMAND orbi_streammoe_cpu_reference_tests)
  add_test(NAME orbi_streammoe_expert_cache_tests COMMAND orbi_streammoe_expert_cache_tests)

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
