/**
 * InfiniCCL Example: `Gather` (OpenMPI + CCL Hybrid)
 *
 * This example demonstrates the API for performing a collective
 * data gathering across multiple accelerators and nodes, where only
 * `root` receives the gathered result.
 */

#include <unistd.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
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

bool RunGatherExample(int argc, char **argv, int warmup_iter, int profile_iter,
                      const size_t kNumElements) {
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
  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_str != nullptr) {
    local_rank = std::atoi(local_rank_str);
  }

  CHECK_RT(Rt, Rt::SetDevice(local_rank));

  // Setup Communicator
  infinicclUniqueId id;
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclChar, 0, comm,
                                  nullptr));

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  // Root of the `Gather` operation.
  constexpr int kRoot = 0;

  const bool is_root = rank == kRoot;

  // Prepare Data
  if (size <= 0 || kNumElements == 0 ||
      kNumElements > std::numeric_limits<size_t>::max() / sizeof(float) /
                         static_cast<size_t>(size)) {
    std::cerr << "`Gather` buffer size is invalid." << std::endl;
    CHECK_INFINI(infinicclCommDestroy(comm));
    CHECK_INFINI(infinicclFinalize());
    return false;
  }
  std::vector<float> h_send(kNumElements);
  std::vector<float> h_recv(
      is_root ? kNumElements * static_cast<size_t>(size) : 0, 0.0f);

  // Initialize: each rank provides its `rank + 1` as data.
  for (size_t i = 0; i < kNumElements; i++) {
    h_send[i] = static_cast<float>(rank + 1);
  }

  float *d_send = nullptr;
  float *d_recv = nullptr;
  const size_t send_bytes = kNumElements * sizeof(*d_send);
  const size_t recv_bytes = send_bytes * static_cast<size_t>(size);
  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, send_bytes));
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, recv_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                            Rt::MemcpyHostToDevice));
  }

  if (rank == kRoot) {
    std::cout << "\n=== Performing Gather on GPU Memory ===" << std::endl;
    std::cout << "Data size: " << kNumElements << " floats ("
              << send_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Operation: Gather" << std::endl;
    std::cout << "Root Rank: " << kRoot << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up
  for (int i = 0; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, kNumElements, infinicclFloat32,
                                 kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, kNumElements, infinicclFloat32,
                                 kRoot, comm, nullptr));
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation (only meaningful on `root`).
  bool correct = true;
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    for (int source = 0; source < size; ++source) {
      const size_t offset = static_cast<size_t>(source) * kNumElements;
      correct =
          Validator::ValidateResult(h_recv.data() + offset, kNumElements,
                                    static_cast<float>(source + 1), rank) &&
          correct;
    }
    std::cout << "\n=== Gather Results ===" << std::endl;
    std::cout << "Correct: " << (correct ? "YES" : "NO") << std::endl;
    std::cout << "Sample blocks: ";
    for (int source = 0; source < std::min(size, 4); ++source) {
      const size_t offset = static_cast<size_t>(source) * kNumElements;
      std::cout << "[r" << source << ": " << h_recv[offset] << "] ";
    }
    std::cout << std::endl;

    // `Gather` counts total bytes at root, with a `(size - 1) / size` bus
    // factor.
    const double alg_bw =
        static_cast<double>(recv_bytes) / 1.0e9 / (elapsed / 1000.0);
    const double bus_bw = alg_bw * static_cast<double>(size - 1) / size;
    std::cout << std::fixed << std::setprecision(3)
              << "Time:           " << elapsed << " ms" << std::endl;
    std::cout << std::setprecision(2) << "Throughput:     " << bus_bw
              << " GB/s (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:  " << alg_bw << " GB/s" << std::endl;
  }

  // Cleanup
  CHECK_RT(Rt, Rt::Free(d_send));
  if (is_root) {
    CHECK_RT(Rt, Rt::Free(d_recv));
  }

  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == kRoot) {
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return correct;
}

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  return RunGatherExample(argc, argv, warmup_iters, profile_iters, num_elements)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
