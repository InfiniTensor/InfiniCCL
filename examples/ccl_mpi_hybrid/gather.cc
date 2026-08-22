/**
 * InfiniCCL Example: Gather (OpenMPI + CCL Hybrid)
 *
 * This example first exercises Gather through its OpenMPI fallback, then
 * initializes a native CCL communicator and profiles the grouped-P2P path.
 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
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

constexpr int kRoot = 0;

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

bool ValidateGather(const std::vector<float> &result, size_t num_elements,
                    int world_size) {
  bool correct = true;
  for (int source = 0; source < world_size; ++source) {
    const size_t offset = static_cast<size_t>(source) * num_elements;
    const bool block_correct =
        Validator::ValidateResult(result.data() + offset, num_elements,
                                  static_cast<float>(source + 1), kRoot);
    correct = block_correct && correct;
  }
  return correct;
}

void PrintGatherMetrics(size_t num_elements, int world_size,
                        double elapsed_ms) {
  constexpr double kBytesPerMiB = 1024.0 * 1024.0;
  constexpr double kBytesPerGB = 1.0e9;
  const double rank_bytes = static_cast<double>(num_elements) * sizeof(float);
  const double gathered_bytes = rank_bytes * static_cast<double>(world_size);
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << "Data size per rank: " << num_elements << " floats ("
            << std::fixed << std::setprecision(2) << rank_bytes / kBytesPerMiB
            << " MiB)" << std::endl;
  std::cout << "Total data at root: "
            << num_elements * static_cast<size_t>(world_size) << " floats ("
            << gathered_bytes / kBytesPerMiB << " MiB)" << std::endl;
  std::cout << "Time:           " << std::setprecision(3) << elapsed_ms << " ms"
            << std::endl;
  if (elapsed_ms > 0.0 && std::isfinite(elapsed_ms)) {
    const double algorithm_bandwidth =
        gathered_bytes / kBytesPerGB / (elapsed_ms / 1000.0);
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
                 size_t num_elements, int world_size, double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== Hybrid CCL Gather Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Root rank: " << kRoot << std::endl;
  std::cout << "Sample receive blocks: ";
  for (int source = 0; source < std::min(world_size, 4); ++source) {
    const size_t offset = static_cast<size_t>(source) * num_elements;
    std::cout << "[r" << source << ": " << result[offset] << "] ";
  }
  std::cout << std::endl;
  PrintGatherMetrics(num_elements, world_size, elapsed_ms);
}

bool RunGatherExample(int argc, char **argv) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  constexpr int kWarmupIterations = 2;
  constexpr int kProfileIterations = 20;
  constexpr size_t kNumElements = 1 << 20;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = -1;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));
  if (size <= 0) {
    std::cerr << "Invalid world size for hybrid Gather." << std::endl;
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
    std::cerr << "Failed to query the hostname for hybrid Gather." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Before a native communicator exists, the selected CCL Gather provider
  // delegates this rank token exchange to the OpenMPI inter communicator.
  const float bootstrap_value = static_cast<float>(rank + 1);
  std::vector<float> h_bootstrap_recv;
  if (rank == kRoot) {
    h_bootstrap_recv.resize(static_cast<size_t>(size), 0.0f);
  }
  float *d_bootstrap_send = nullptr;
  float *d_bootstrap_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_send),
                          sizeof(float)));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_recv),
                            static_cast<size_t>(size) * sizeof(float)));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_bootstrap_send, &bootstrap_value, sizeof(float),
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclGather(d_bootstrap_send, d_bootstrap_recv, 1,
                               infinicclFloat32, kRoot, comm, nullptr));

  int32_t bootstrap_status = 1;
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(h_bootstrap_recv.data(), d_bootstrap_recv,
                            static_cast<size_t>(size) * sizeof(float),
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    for (int source = 0; source < size; ++source) {
      if (h_bootstrap_recv[static_cast<size_t>(source)] !=
          static_cast<float>(source + 1)) {
        bootstrap_status = 0;
      }
    }
  }
  CHECK_INFINI(infinicclBroadcast(&bootstrap_status, &bootstrap_status, 1,
                                  infinicclInt32, kRoot, comm, nullptr));
  CHECK_RT(Rt, Rt::Free(d_bootstrap_send));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_bootstrap_recv));
  }

  if (bootstrap_status != 1) {
    if (rank == kRoot) {
      std::cerr << "OpenMPI Gather fallback validation failed." << std::endl;
    }
    CHECK_INFINI(infinicclCommDestroy(comm));
    CHECK_INFINI(infinicclFinalize());
    return false;
  }
  if (rank == kRoot) {
    std::cout << "OpenMPI Gather fallback validation passed." << std::endl;
  }

  infinicclUniqueId id{};
  if (rank == kRoot) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclUInt8, kRoot,
                                  comm, nullptr));
  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  const size_t world_size = static_cast<size_t>(size);
  if (kNumElements > std::numeric_limits<size_t>::max() / world_size ||
      kNumElements * world_size >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Hybrid Gather buffer size overflows `size_t`." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const size_t rank_bytes = kNumElements * sizeof(float);
  const size_t gathered_elements = kNumElements * world_size;
  const size_t gathered_bytes = gathered_elements * sizeof(float);
  std::vector<float> h_send(kNumElements, static_cast<float>(rank + 1));
  std::vector<float> h_recv;
  if (rank == kRoot) {
    h_recv.resize(gathered_elements, 0.0f);
  }

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), rank_bytes));
  if (rank == kRoot) {
    CHECK_RT(Rt,
             Rt::Malloc(reinterpret_cast<void **>(&d_recv), gathered_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), rank_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, kNumElements, infinicclFloat32,
                                 kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < kProfileIterations; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, kNumElements, infinicclFloat32,
                                 kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(kProfileIterations);

  int32_t validation_status = 1;
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, gathered_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    validation_status =
        ValidateGather(h_recv, kNumElements, size) ? int32_t{1} : int32_t{0};
  }
  CHECK_INFINI(infinicclBroadcast(&validation_status, &validation_status, 1,
                                  infinicclInt32, kRoot, comm, nullptr));
  const bool correct = validation_status == 1;

  if (rank == kRoot) {
    PrintResult(correct, h_recv, kNumElements, size, elapsed_ms);
  }

  CHECK_RT(Rt, Rt::Free(d_send));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_recv));
  }
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == kRoot) {
    if (correct) {
      std::cout << "[Main Process] Hybrid CCL Gather validation passed."
                << std::endl;
    } else {
      std::cerr << "[Main Process] Hybrid CCL Gather validation failed."
                << std::endl;
    }
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return correct;
}

}  // namespace

int main(int argc, char **argv) {
  return RunGatherExample(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
