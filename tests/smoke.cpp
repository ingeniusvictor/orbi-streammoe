#include <cstdlib>
#include <iostream>

#include "orbi/streammoe/model/arch_config.hpp"
#include "orbi/streammoe/version.hpp"

int main() {
  using namespace orbi::streammoe;

  if (kQwen3Next80BA3B.layer_count != 48) return EXIT_FAILURE;
  if (kQwen3Next80BA3B.full_attention_layer_count() != 12) return EXIT_FAILURE;
  if (kQwen3Next80BA3B.linear_layer_count() != 36) return EXIT_FAILURE;
  if (kQwen3Next80BA3B.expert_count != 512) return EXIT_FAILURE;
  if (kQwen3Next80BA3B.expert_top_k != 10) return EXIT_FAILURE;
  if (kQwen3Next80BA3B.routed_fetches_per_token() != 480) return EXIT_FAILURE;

  std::cout << kProjectName << " " << runtime_version() << "\n";
  std::cout << "OSM-00 smoke: PASS\n";
  return EXIT_SUCCESS;
}
