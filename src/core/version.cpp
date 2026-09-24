#include "orbi/streammoe/version.hpp"

namespace orbi::streammoe {

std::string_view runtime_version() noexcept {
  return kVersion;
}

}  // namespace orbi::streammoe
