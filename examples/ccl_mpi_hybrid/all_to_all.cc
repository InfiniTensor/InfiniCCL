/**
 * InfiniCCL Example: AllToAll (OpenMPI + CCL Hybrid)
 *
 * This example uses AllToAll through an OpenMPI communicator to distribute a
 * native CCL unique ID, then performs AllToAll on GPU memory.
 */

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#include "backend_manifest.h"
#include "device.h"
#include "infiniccl.h"
#include "runtime.h"
#include "traits.h"
#include "utils.h"

using namespace infini::ccl;

float ElementValue(int source, int destination, size_t index,
                   size_t count_per_peer, int size) {
  const size_t block = static_cast<size_t>(source) * size + destination;
  return static_cast<float>(block * count_per_peer + index);
}

bool ValidateResult(const std::vector<float> &result, size_t count_per_peer,
                    int size, int rank) {
  for (int source = 0; source < size; ++source) {
    const size_t offset = static_cast<size_t>(source) * count_per_peer;
    for (size_t index = 0; index < count_per_peer; ++index) {
      const float expected =
          ElementValue(source, rank, index, count_per_peer, size);
      if (result[offset + index] != expected) {
        std::cerr << "[Rank " << rank << "] AllToAll mismatch at source "
                  << source << ", index " << index << ": expected " << expected
                  << ", got " << result[offset + index] << std::endl;
        return false;
      }
    }
  }
  return true;
}

void PrintMetrics(double elapsed_ms, size_t total_bytes, int size) {
  const double seconds = elapsed_ms / 1000.0;
  const double gigabytes =
      static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double algorithm_bandwidth = gigabytes / seconds;
  const double bus_bandwidth =
      algorithm_bandwidth * static_cast<double>(size - 1) / size;

  std::cout << "Time:           " << std::fixed << std::setprecision(3)
            << elapsed_ms << " ms" << std::endl;
  std::cout << "Throughput:     " << std::fixed << std::setprecision(2)
            << bus_bandwidth << " GB/s (Bus BW)" << std::endl;
  std::cout << "Alg Bandwidth:  " << std::fixed << std::setprecision(2)
            << algorithm_bandwidth << " GB/s" << std::endl;
}

bool RunAllToAllExample(int argc, char **argv, int warmup_iter,
                        int profile_iter, size_t count_per_peer) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank, size;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));
  if (size <= 0) {
    std::cerr << "Invalid world size for AllToAll." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_str != nullptr) {
    local_rank = std::atoi(local_rank_str);
  }
  CHECK_RT(Rt, Rt::SetDevice(local_rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Exercise the MPI fallback before initializing the native communicator.
  infinicclUniqueId id{};
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  const size_t id_bytes = sizeof(id);
  const size_t all_id_bytes = id_bytes * static_cast<size_t>(size);
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
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(&id, d_id_recv, id_bytes, Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::Free(d_id_send));
  CHECK_RT(Rt, Rt::Free(d_id_recv));

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << std::endl;

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  const size_t world_size = static_cast<size_t>(size);
  if (count_per_peer > std::numeric_limits<size_t>::max() / world_size ||
      count_per_peer * world_size >
          std::numeric_limits<size_t>::max() / sizeof(float) ||
      count_per_peer > (size_t{1} << 24) / world_size / world_size) {
    std::cerr << "AllToAll buffer or validation value exceeds its range."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const size_t total_elements = count_per_peer * world_size;
  const size_t total_bytes = total_elements * sizeof(float);
  std::vector<float> h_send(total_elements);
  std::vector<float> h_recv(total_elements, -1.0f);
  for (int destination = 0; destination < size; ++destination) {
    const size_t offset = static_cast<size_t>(destination) * count_per_peer;
    for (size_t index = 0; index < count_per_peer; ++index) {
      h_send[offset + index] =
          ElementValue(rank, destination, index, count_per_peer, size);
    }
  }

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == 0) {
    std::cout << "\n=== Performing AllToAll on GPU Memory ===" << std::endl;
    std::cout << "Data size per rank: " << total_elements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  for (int i = 0; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, count_per_peer,
                                   infinicclFloat32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, count_per_peer,
                                   infinicclFloat32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Validate a separate untimed exchange with a sentinel-filled output.
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclAllToAll(d_send, d_recv, count_per_peer,
                                 infinicclFloat32, comm, nullptr));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));
  const bool correct = ValidateResult(h_recv, count_per_peer, size, rank);

  if (rank == 0) {
    PrintMetrics(elapsed_ms, total_bytes, size);
  }
  std::cout << "[Rank " << rank << "] AllToAll validation "
            << (correct ? "passed." : "failed.") << std::endl;

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
  size_t count_per_peer = 1 << 18;

  return RunAllToAllExample(argc, argv, warmup_iters, profile_iters,
                            count_per_peer)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
