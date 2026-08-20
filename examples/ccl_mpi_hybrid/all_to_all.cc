/**
 * InfiniCCL Example: AllToAll (OpenMPI + CCL Hybrid)
 *
 * This example first uses AllToAll through an OpenMPI inter communicator to
 * distribute a native CCL unique ID. It then initializes a native CCL
 * communicator and validates an out-of-place GPU all-to-all exchange.
 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

// Public API
#include "infiniccl.h"

// Example-Specific Utilities
#include "utils.h"

// Internal Headers (Accessible via example-specific include paths, technically
// not public APIs)
#include "backend_manifest.h"
#include "device.h"
#include "runtime.h"
#include "traits.h"

using namespace infini::ccl;

namespace {

bool ParseLocalRank(const char *text, int *local_rank) {
  if (!text || !local_rank) {
    return false;
  }

  int parsed = -1;
  const char *end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end || parsed < 0) {
    return false;
  }

  *local_rank = parsed;
  return true;
}

float BlockValue(int source_rank, int destination_rank) {
  return static_cast<float>((source_rank + 1) * 1000 + destination_rank + 1);
}

void FillInput(std::vector<float> *input, size_t count_per_peer, int world_size,
               int rank) {
  for (int destination = 0; destination < world_size; ++destination) {
    const size_t offset = static_cast<size_t>(destination) * count_per_peer;
    std::fill_n(input->begin() + offset, count_per_peer,
                BlockValue(rank, destination));
  }
}

bool ValidateAllToAll(const std::vector<float> &result, size_t count_per_peer,
                      int world_size, int rank) {
  bool correct = true;
  for (int source = 0; source < world_size; ++source) {
    const size_t offset = static_cast<size_t>(source) * count_per_peer;
    const bool block_correct = Validator::ValidateResult(
        result.data() + offset, count_per_peer, BlockValue(source, rank), rank);
    correct = block_correct && correct;
  }
  return correct;
}

void PrintAllToAllMetrics(size_t count_per_peer, int world_size,
                          double elapsed_ms) {
  constexpr double kBytesPerMiB = 1024.0 * 1024.0;
  constexpr double kBytesPerGB = 1.0e9;
  const double peer_bytes = static_cast<double>(count_per_peer) * sizeof(float);
  const double total_bytes = peer_bytes * static_cast<double>(world_size);
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << "Data size per peer: " << count_per_peer << " floats ("
            << std::fixed << std::setprecision(2) << peer_bytes / kBytesPerMiB
            << " MiB)" << std::endl;
  std::cout << "Total data per rank: "
            << count_per_peer * static_cast<size_t>(world_size) << " floats ("
            << total_bytes / kBytesPerMiB << " MiB)" << std::endl;
  std::cout << "Time:           " << std::setprecision(3) << elapsed_ms << " ms"
            << std::endl;
  if (elapsed_ms > 0.0) {
    const double algorithm_bandwidth =
        total_bytes / kBytesPerGB / (elapsed_ms / 1000.0);
    const double bus_bandwidth = algorithm_bandwidth *
                                 static_cast<double>(world_size - 1) /
                                 static_cast<double>(world_size);
    std::cout << "Throughput:     " << std::setprecision(2) << bus_bandwidth
              << " GB/s (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:  " << algorithm_bandwidth << " GB/s"
              << std::endl;
  } else {
    std::cout << "Throughput:     N/A (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:  N/A" << std::endl;
  }

  std::cout.flags(original_flags);
  std::cout.precision(original_precision);
}

void PrintResult(bool correct, const std::vector<float> &result,
                 size_t count_per_peer, int world_size, double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== Hybrid CCL AllToAll Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  PrintAllToAllMetrics(count_per_peer, world_size, elapsed_ms);
  std::cout << "Sample receive blocks: ";
  for (int source = 0; source < std::min(world_size, 4); ++source) {
    const size_t offset = static_cast<size_t>(source) * count_per_peer;
    std::cout << "[src" << source << ": " << result[offset] << "] ";
  }
  std::cout << std::endl;
}

bool RunAllToAllExample(int argc, char **argv) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  constexpr int kWarmupIterations = 2;
  constexpr int kProfileIterations = 20;
  constexpr size_t kCountPerPeer = 1 << 20;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = -1;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));
  if (size <= 0) {
    std::cerr << "Invalid world size for hybrid AllToAll." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  int local_rank = -1;
  if (!ParseLocalRank(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"), &local_rank)) {
    std::cerr << "Missing or invalid `OMPI_COMM_WORLD_LOCAL_RANK`."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  CHECK_RT(Rt, Rt::SetDevice(local_rank));

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for hybrid AllToAll."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  const size_t world_size = static_cast<size_t>(size);
  const size_t id_bytes = sizeof(infinicclUniqueId);
  if (id_bytes > std::numeric_limits<size_t>::max() / world_size) {
    std::cerr << "AllToAll bootstrap buffer size overflows `size_t`."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const size_t all_id_bytes = id_bytes * world_size;

  // Bootstrap the native communicator with AllToAll itself. Rank 0 repeats
  // its unique ID in every destination block; each rank then takes the block
  // received from source rank 0. With only the OpenMPI inter communicator
  // initialized, the CCL provider delegates this call to `MPI_Alltoall`.
  infinicclUniqueId id{};
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  std::vector<uint8_t> h_id_send(all_id_bytes, 0);
  if (rank == 0) {
    for (int destination = 0; destination < size; ++destination) {
      std::memcpy(
          h_id_send.data() + static_cast<size_t>(destination) * id_bytes, &id,
          id_bytes);
    }
  }

  uint8_t *d_id_send = nullptr;
  uint8_t *d_id_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_id_send), all_id_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_id_recv), all_id_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_id_send, h_id_send.data(), all_id_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclAllToAll(d_id_send, d_id_recv, id_bytes, infinicclUInt8,
                                 comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(&id, d_id_recv, id_bytes, Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Free(d_id_send));
  CHECK_RT(Rt, Rt::Free(d_id_recv));

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  if (kCountPerPeer > std::numeric_limits<size_t>::max() / world_size ||
      kCountPerPeer * world_size >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Hybrid AllToAll buffer size overflows `size_t`." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const size_t total_elements = kCountPerPeer * world_size;
  const size_t total_bytes = total_elements * sizeof(float);
  std::vector<float> h_send(total_elements);
  std::vector<float> h_recv(total_elements, 0.0f);
  FillInput(&h_send, kCountPerPeer, size, rank);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, kCountPerPeer,
                                   infinicclFloat32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < kProfileIterations; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, kCountPerPeer,
                                   infinicclFloat32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(kProfileIterations);

  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const bool local_correct =
      ValidateAllToAll(h_recv, kCountPerPeer, size, rank);

  // Use native AllToAll once more to exchange every rank's validation flag.
  // This makes all ranks derive the same final process status without direct
  // MPI or vendor-library calls in the example.
  std::vector<int32_t> h_status_send(world_size, local_correct ? 1 : 0);
  std::vector<int32_t> h_status_recv(world_size, 0);
  int32_t *d_status_send = nullptr;
  int32_t *d_status_recv = nullptr;
  const size_t status_bytes = world_size * sizeof(int32_t);
  CHECK_RT(Rt,
           Rt::Malloc(reinterpret_cast<void **>(&d_status_send), status_bytes));
  CHECK_RT(Rt,
           Rt::Malloc(reinterpret_cast<void **>(&d_status_recv), status_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_status_send, h_status_send.data(), status_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclAllToAll(d_status_send, d_status_recv, 1,
                                 infinicclInt32, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_status_recv.data(), d_status_recv, status_bytes,
                          Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const bool correct =
      local_correct && std::all_of(h_status_recv.begin(), h_status_recv.end(),
                                   [](int32_t value) { return value == 1; });

  if (rank == 0) {
    PrintResult(correct, h_recv, kCountPerPeer, size, elapsed_ms);
  }

  CHECK_RT(Rt, Rt::Free(d_status_send));
  CHECK_RT(Rt, Rt::Free(d_status_recv));
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == 0) {
    if (correct) {
      std::cout << "[Main Process] Hybrid CCL AllToAll validation passed."
                << std::endl;
    } else {
      std::cerr << "[Main Process] Hybrid CCL AllToAll validation failed."
                << std::endl;
    }
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return correct;
}

}  // namespace

int main(int argc, char **argv) {
  return RunAllToAllExample(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
