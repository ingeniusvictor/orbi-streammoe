#pragma once

#include <string_view>

namespace orbi::streammoe {

inline constexpr std::string_view kProjectName = "ORBI StreamMoE";
inline constexpr std::string_view kVersion = "0.1.0-dev";

std::string_view runtime_version() noexcept;

}  // namespace orbi::streammoe
