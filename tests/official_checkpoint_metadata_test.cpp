#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include "orbi/streammoe/model/sharded_checkpoint_contract.hpp"

using namespace orbi::streammoe;

namespace {

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error(
          "usage: orbi_streammoe_official_checkpoint_metadata_tests <model-dir>");
    }

    const auto contract =
        inspect_qwen3_next_sharded_checkpoint(std::filesystem::path(argv[1]));

    require(contract.hidden_size == 2048U, "official hidden_size mismatch");
    require(contract.vocab_size == 151936U, "official vocab_size mismatch");
    require(contract.num_hidden_layers == 48U, "official layer count mismatch");
    require(
        contract.full_attention_interval == 4U,
        "official full-attention interval mismatch");
    require(contract.num_experts == 512U, "official expert count mismatch");
    require(contract.num_experts_per_tok == 10U, "official top-k mismatch");
    require(contract.shard_count == 41U, "official shard count mismatch");
    require(
        contract.total_size_bytes > 150000000000ULL,
        "official checkpoint total size unexpectedly small");
    require(
        contract.total_size_bytes < 180000000000ULL,
        "official checkpoint total size unexpectedly large");

    require(contract.is_linear_layer(0U), "official layer 0 must be DeltaNet");
    require(contract.is_linear_layer(1U), "official layer 1 must be DeltaNet");
    require(contract.is_linear_layer(2U), "official layer 2 must be DeltaNet");
    require(!contract.is_linear_layer(3U), "official layer 3 must be GQA");
    require(!contract.is_linear_layer(47U), "official final layer must be GQA");

    std::cout
        << "OSM-38B official checkpoint metadata: PASS\n"
        << "  hidden_size=" << contract.hidden_size << "\n"
        << "  vocab_size=" << contract.vocab_size << "\n"
        << "  layers=" << contract.num_hidden_layers << "\n"
        << "  full_attention_interval=" << contract.full_attention_interval << "\n"
        << "  experts=" << contract.num_experts << "\n"
        << "  experts_per_token=" << contract.num_experts_per_tok << "\n"
        << "  shards=" << contract.shard_count << "\n"
        << "  tensors=" << contract.tensor_count << "\n"
        << "  total_size_bytes=" << contract.total_size_bytes << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr
        << "OSM-38B official checkpoint metadata: FAIL: "
        << e.what() << "\n";
    return 1;
  }
}
