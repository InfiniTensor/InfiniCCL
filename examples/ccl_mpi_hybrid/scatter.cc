/**
 * InfiniCCL Example: `Scatter` (OpenMPI + CCL Hybrid)
 *
 * This example demonstrates the API for performing a collective
 * data distribution across multiple GPUs and nodes, where `root`
 * sends a distinct block to every rank.
 */

#include <unistd.h>

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

bool RunScatterExample(int argc, char **argv, int warmup_iter, int profile_iter,
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

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  // Setup Communicator
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));
  infinicclUniqueId id;
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclChar, 0, comm,
                                  nullptr));
  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  // Root of the Scatter
  constexpr int kRoot = 0;

  // Prepare Data
  if (size <= 0 || kNumElements == 0 ||
      kNumElements > std::numeric_limits<size_t>::max() / sizeof(float) /
                         static_cast<size_t>(size)) {
    std::cerr << "`Scatter` buffer size is invalid." << std::endl;
    CHECK_INFINI(infinicclCommDestroy(comm));
    CHECK_INFINI(infinicclFinalize());
    return false;
  }
  const bool is_root = rank == kRoot;
  std::vector<float> h_send(
      is_root ? kNumElements * static_cast<size_t>(size) : 0, 0.0f);
  std::vector<float> h_recv(kNumElements, 0.0f);

  // Initialize: `root` fills the block destined for rank `r` with `(r + 1)`.
  if (rank == kRoot) {
    for (int r = 0; r < size; ++r) {
      for (size_t i = 0; i < kNumElements; ++i) {
        h_send[static_cast<size_t>(r) * kNumElements + i] =
            static_cast<float>(r + 1);
      }
    }
  }

  float *d_send = nullptr;
  float *d_recv = nullptr;
  size_t recv_bytes = kNumElements * sizeof(*d_recv);
  size_t send_bytes = recv_bytes * size;
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc((void **)&d_send, send_bytes));
  }
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, recv_bytes));
  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                            Rt::MemcpyHostToDevice));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == kRoot) {
    std::cout << "\n=== Performing Scatter on GPU Memory ===" << std::endl;
    std::cout << "Data size per rank: " << kNumElements << " floats ("
              << recv_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Operation: Scatter" << std::endl;
    std::cout << "Root Rank: " << kRoot << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up
  for (int i = 0; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclScatter(d_send, d_recv, kNumElements,
                                  infinicclFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(infinicclScatter(d_send, d_recv, kNumElements,
                                  infinicclFloat32, kRoot, comm, nullptr));
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation: every rank should receive its own `(rank + 1)` block.
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const bool correct = Validator::ValidateResult(h_recv.data(), kNumElements,
                                                 static_cast<float>(rank + 1),
                                                 rank, true, "Scatter");

  // Metrics Reporting (Only from rank 0 for cleaner output)
  if (rank == kRoot) {
    Metrics metrics{elapsed, send_bytes, size};
    metrics.Print();
  }

  // Cleanup
  if (is_root) {
    CHECK_RT(Rt, Rt::Free(d_send));
  }
  CHECK_RT(Rt, Rt::Free(d_recv));

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

  return RunScatterExample(argc, argv, warmup_iters, profile_iters,
                           num_elements)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
