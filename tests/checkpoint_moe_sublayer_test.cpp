#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "orbi/streammoe/model/checkpoint_moe_sublayer.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main() {
  try {
    const std::vector<float> residual{1.0F, -2.0F, 3.0F, 0.5F};
    const std::vector<float> branch{0.25F, 1.5F, -0.5F, -0.25F};
    const std::vector<float> expected{1.25F, -0.5F, 2.5F, 0.25F};

    const auto output = add_residual_cpu(residual, branch);
    require(output.size() == expected.size(), "residual output size mismatch");
    for (std::size_t i = 0; i < output.size(); ++i) {
      require(
          std::fabs(output[i] - expected[i]) <= 1e-7F,
          "residual output mismatch");
    }

    bool mismatch_rejected = false;
    try {
      (void)add_residual_cpu(
          residual,
          std::span<const float>(branch).first(branch.size() - 1U));
    } catch (const std::invalid_argument&) {
      mismatch_rejected = true;
    }
    require(mismatch_rejected, "residual shape mismatch must be rejected");

    std::cout
        << "OSM-28 checkpoint MoE sublayer composition: PASS\n"
        << "  residual_add=PASS\n"
        << "  shape_guard=PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-28 checkpoint MoE sublayer composition: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
