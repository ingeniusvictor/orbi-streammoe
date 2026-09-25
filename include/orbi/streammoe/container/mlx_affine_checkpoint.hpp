#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "orbi/streammoe/container/qpack.hpp"
#include "orbi/streammoe/container/qpack_dense.hpp"

namespace orbi::streammoe {

struct MlxAffineQuantSpec {
  std::uint32_t group_size{};
  std::uint32_t bits{};
  std::string mode{"affine"};

  [[nodiscard]] bool operator==(const MlxAffineQuantSpec&) const = default;
};

struct MlxQuantizationConfig {
  std::optional<MlxAffineQuantSpec> default_quant;
  std::vector<std::pair<std::string, MlxAffineQuantSpec>> overrides;
};

struct MlxAffineModule {
  std::string path;
  MlxAffineQuantSpec quant;
  std::vector<std::size_t> packed_shape;
  std::vector<std::size_t> logical_shape;
  std::vector<std::size_t> scale_shape;
  std::size_t row_count{};
  std::size_t logical_in_dim{};
  std::vector<std::uint32_t> packed;
  std::vector<float> scales;
  std::vector<float> biases;
};

class QpackMlxCheckpoint {
 public:
  explicit QpackMlxCheckpoint(const QpackReader& qpack);

  [[nodiscard]] const QpackDenseReader& dense() const noexcept {
    return dense_;
  }

  [[nodiscard]] const MlxQuantizationConfig& quantization() const noexcept {
    return quantization_;
  }

  [[nodiscard]] bool is_quantized(std::string_view path) const noexcept;

  [[nodiscard]] std::optional<MlxAffineQuantSpec> quant_spec_for(
      std::string_view path) const;

  [[nodiscard]] MlxAffineModule read_affine_module(
      std::string_view path) const;

 private:
  QpackDenseReader dense_;
  MlxQuantizationConfig quantization_;
};

[[nodiscard]] MlxQuantizationConfig parse_mlx_quantization_config(
    const std::filesystem::path& config_path);

}  // namespace orbi::streammoe
