cmake_minimum_required(VERSION 3.24)

project(orbi_streammoe
  VERSION 0.1.0
  DESCRIPTION "Portable streamed-MoE inference runtime research"
  LANGUAGES CXX
)

option(ORBI_STREAMMOE_BUILD_TESTS "Build ORBI StreamMoE tests" ON)

add_library(orbi_streammoe_core
  src/core/version.cpp
)

target_include_directories(orbi_streammoe_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(orbi_streammoe_core PUBLIC cxx_std_20)

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

  add_test(NAME orbi_streammoe_smoke COMMAND orbi_streammoe_smoke)
endif()
