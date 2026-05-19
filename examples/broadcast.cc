/**
 * InfiniCCL Example: Broadcast
 *
 * This example demonstrates the API for performing a collective
 * broadcast operation across multiple accelerators and nodes,
 * supporting both out-of-place and in-place memory topologies.
 */

#include <unistd.h>

#include <iostream>
#include <vector>

// Public API
#include "infiniccl.h"

// Example-Specific Utilities
#include "utils.h"

// Internal Headers
#include "backend_manifest.h"
#include "device.h"
#include "runtime.h"
#include "traits.h"

using namespace infini::ccl;

void RunBroadcastExample(int argc, char **argv, int warmup_iter,
                         int profile_iter, const size_t kNumElements) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_INFINI(infiniInit(&argc, &argv));

  int rank, size;
  CHECK_INFINI(infiniGetRank(&rank));
  CHECK_INFINI(infiniGetSize(&size));

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_str != nullptr) {
    local_rank = std::atoi(local_rank_str);
  }

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  // Setup Communicator
  infiniComm_t comm = nullptr;
  CHECK_INFINI(infiniCommInitAll(&comm, size, nullptr));

  // Root of the Broadcast
  constexpr int kRoot = 0;
  constexpr float kRootMagicValue = 42.0f;

  // Prepare Host Data
  std::vector<float> h_send(kNumElements, 0.0f);
  std::vector<float> h_recv(kNumElements, 0.0f);

  // Initialize: Only root fills its buffer with meaningful payload.
  if (rank == kRoot) {
    for (size_t i = 0; i < kNumElements; i++) {
      h_send[i] = kRootMagicValue;
    }
  }

  // Allocate Device Memory
  float *d_send, *d_recv;
  size_t total_bytes = kNumElements * sizeof(float);
  CHECK_RT(Rt, Rt::Malloc(&d_send, total_bytes));
  CHECK_RT(Rt, Rt::Malloc(&d_recv, total_bytes));

  // Copy data from host to device memory.
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // --------------------------------------------------------------------------
  // Scenario 1: Out-of-Place Broadcast
  // --------------------------------------------------------------------------
  if (rank == kRoot) {
    std::cout << "\n=== Scenario 1: Performing Out-of-Place Broadcast ==="
              << std::endl;
    std::cout << "Data size: " << kNumElements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Root Node: " << kRoot << std::endl;
  }

  // Warm-up and pull single verification frame.
  CHECK_INFINI(infiniBroadcast(d_send, d_recv, kNumElements, infiniFloat32,
                               kRoot, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(infiniBroadcast(d_send, d_recv, kNumElements, infiniFloat32,
                                 kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling Out-of-Place
  Timer oop_timer;
  for (int i = 0; i < profile_iter; i++) {
    CHECK_INFINI(infiniBroadcast(d_send, d_recv, kNumElements, infiniFloat32,
                                 kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  double oop_elapsed =
      oop_timer.elapsed_ms() / static_cast<double>(profile_iter);

  // Out-of-Place Result Validation
  Validator::ValidateResult(h_recv.data(), kNumElements, kRootMagicValue, rank,
                            true, "Broadcast");

  if (rank == kRoot) {
    Metrics metrics{oop_elapsed, total_bytes, size};
    metrics.Print();
  }

  // --------------------------------------------------------------------------
  // Scenario 2: In-Place Broadcast
  // --------------------------------------------------------------------------
  if (rank == kRoot) {
    std::cout << "\n=== Scenario 2: Performing In-Place Broadcast ==="
              << std::endl;
  }

  // Reset receiving host buffer to verify clean overwrite tracking.
  std::fill(h_recv.begin(), h_recv.end(), 0.0f);

  // In-place means non-root ranks receive into their primary operational
  // buffer. Root broadcasts directly out of its existing storage buffer.
  float *d_inplace_buf = (rank == kRoot) ? d_send : d_recv;

  // Warm-up
  CHECK_INFINI(infiniBroadcast(d_inplace_buf, d_inplace_buf, kNumElements,
                               infiniFloat32, kRoot, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_inplace_buf, total_bytes,
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(infiniBroadcast(d_inplace_buf, d_inplace_buf, kNumElements,
                                 infiniFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling In-Place
  Timer ip_timer;
  for (int i = 0; i < profile_iter; i++) {
    CHECK_INFINI(infiniBroadcast(d_inplace_buf, d_inplace_buf, kNumElements,
                                 infiniFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  double ip_elapsed = ip_timer.elapsed_ms() / static_cast<double>(profile_iter);

  // In-Place Result Validation
  Validator::ValidateResult(h_recv.data(), kNumElements, kRootMagicValue, rank,
                            true, "Broadcast");

  if (rank == kRoot) {
    Metrics metrics{ip_elapsed, total_bytes, size};
    metrics.Print();
  }

  // --------------------------------------------------------------------------
  // Cleanup
  // --------------------------------------------------------------------------
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));

  CHECK_INFINI(infiniCommDestroy(comm));
  CHECK_INFINI(infiniFinalize());

  if (rank == kRoot) {
    std::cout
        << "\nInfiniCCL Broadcast execution complete and finalized safely."
        << std::endl;
  }
}

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  RunBroadcastExample(argc, argv, warmup_iters, profile_iters, num_elements);

  return EXIT_SUCCESS;
}
