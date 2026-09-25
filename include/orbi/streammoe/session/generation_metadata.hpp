#pragma once

#include <cstddef>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"

namespace orbi::streammoe {

struct QwenGenerationMetadata {
  std::vector<std::size_t> eos_token_ids;
};

/// Parse generation metadata from qpack auxiliary JSON.
///
/// Mirrors the frozen Swiftlet CLI's compatibility behavior: EOS token IDs are
/// collected from generation_config.json when present and from config.json,
/// accepting either one integer or an array of integers. Duplicates are
/// removed deterministically.
[[nodiscard]] QwenGenerationMetadata load_qwen_generation_metadata(
    const QpackReader& qpack);

}  // namespace orbi::streammoe
