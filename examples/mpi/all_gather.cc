/**
 * InfiniCCL Example: AllGather
 * * This example demonstrates the planned API for performing a
 * collective data gathering across multiple GPUs and nodes.
 */

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
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

void PrintAllGatherMetrics(size_t num_elements, int world_size,
                           double elapsed_ms) {
  constexpr double kBytesPerMiB = 1024.0 * 1024.0;
  constexpr double kBytesPerGB = 1.0e9;
  const double rank_bytes = static_cast<double>(num_elements) * sizeof(float);
  const double total_bytes = rank_bytes * static_cast<double>(world_size);
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << "Data size per rank: " << num_elements << " floats ("
            << std::fixed << std::setprecision(2) << rank_bytes / kBytesPerMiB
            << " MiB)" << std::endl;
  std::cout << "Total data per rank: "
            << num_elements * static_cast<size_t>(world_size) << " floats ("
            << total_bytes / kBytesPerMiB << " MiB)" << std::endl;
  std::cout << "Time:           " << std::setprecision(3) << elapsed_ms << " ms"
            << std::endl;
  if (elapsed_ms > 0.0 && std::isfinite(elapsed_ms)) {
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

}  // namespace

bool RunAllGatherExample(int argc, char **argv, int warmup_iter,
                         int profile_iter, const size_t kNumElements) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank, size;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  // Map local rank to GPU device.
  // Note: this is just for info printing. In practice, this part is not needed.
  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_str != nullptr) {
    local_rank = std::atoi(local_rank_str);
  }

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  // Setup Communicator
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Prepare Data
  std::vector<float> h_send(kNumElements);
  std::vector<float> h_recv(kNumElements * size, 0.0f);

  // Initialize: each rank provides its (rank + 1) as data.
  for (size_t i = 0; i < kNumElements; i++) {
    h_send[i] = static_cast<float>(rank + 1);
  }

  float *d_send, *d_recv;
  size_t send_bytes = kNumElements * sizeof(*d_send);
  size_t recv_bytes = send_bytes * size;
  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, send_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == 0) {
    std::cout << "\n=== Performing AllGather on GPU Memory ===" << std::endl;
    std::cout << "Data size: " << kNumElements << " floats ("
              << send_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Operation: AllGather" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up and D2H transfer the answer.
  CHECK_INFINI(infinicclAllGather(d_send, d_recv, kNumElements,
                                  infinicclFloat32, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllGather(d_send, d_recv, kNumElements,
                                    infinicclFloat32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; i++) {
    CHECK_INFINI(infinicclAllGather(d_send, d_recv, kNumElements,
                                    infinicclFloat32, comm, nullptr));
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation
  bool correct = true;

  for (int src_rank = 0; src_rank < size; ++src_rank) {
    float expected = static_cast<float>(src_rank + 1);
    size_t offset = static_cast<size_t>(src_rank) * kNumElements;

    const bool block_correct = Validator::ValidateResult(
        h_recv.data() + offset, kNumElements, expected, rank);
    correct = block_correct && correct;
  }

  if (rank == 0) {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";

    std::cout << "\n=== AllGather Results ===" << std::endl;
    std::cout << "Correct: "
              << (correct ? (GREEN + std::string("YES") + RESET)
                          : (RED + std::string("NO") + RESET));
    std::cout << std::endl;

    std::cout << "Sample blocks: ";
    for (int src_rank = 0; src_rank < std::min(size, 4); ++src_rank) {
      size_t offset = static_cast<size_t>(src_rank) * kNumElements;
      std::cout << "[r" << src_rank << ": " << h_recv[offset] << "] ";
    }
    std::cout << std::endl;
  }

  if (rank == 0) {
    PrintAllGatherMetrics(kNumElements, size, elapsed);
  }

  // Cleanup
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));

  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == 0) {
    std::cout << "InfiniCCL finalized." << std::endl;
  }

  return correct;
}

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  return RunAllGatherExample(argc, argv, warmup_iters, profile_iters,
                             num_elements)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
