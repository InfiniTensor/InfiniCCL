/**
 * InfiniCCL Example: ReduceScatter (OpenMPI + CCL Hybrid)
 *
 * This example first uses ReduceScatter through an OpenMPI inter communicator
 * to distribute a native CCL unique ID. It then initializes a native CCL
 * communicator and validates out-of-place and canonical in-place GPU
 * ReduceScatter.
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

void FillInput(std::vector<float> *input, size_t recv_count, int world_size,
               int rank) {
  for (int dest_rank = 0; dest_rank < world_size; ++dest_rank) {
    const float value =
        static_cast<float>(rank + 1) * static_cast<float>(dest_rank + 1);
    const size_t offset = static_cast<size_t>(dest_rank) * recv_count;
    std::fill_n(input->begin() + offset, recv_count, value);
  }
}

float ExpectedValue(int rank, int world_size) {
  const float rank_sum = static_cast<float>(world_size) *
                         (static_cast<float>(world_size) + 1.0f) / 2.0f;
  return rank_sum * static_cast<float>(rank + 1);
}

void PrintReduceScatterMetrics(size_t recv_count, int world_size,
                               double elapsed_ms) {
  constexpr double kBytesPerMiB = 1024.0 * 1024.0;
  constexpr double kBytesPerGB = 1.0e9;
  const size_t total_count = recv_count * static_cast<size_t>(world_size);
  const double recv_bytes = static_cast<double>(recv_count) * sizeof(float);
  const double total_bytes = static_cast<double>(total_count) * sizeof(float);
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << "Received data per rank: " << recv_count << " floats ("
            << std::fixed << std::setprecision(2) << recv_bytes / kBytesPerMiB
            << " MiB)" << std::endl;
  std::cout << "Total reduced data:    " << total_count << " floats ("
            << total_bytes / kBytesPerMiB << " MiB)" << std::endl;
  std::cout << "Time:                  " << std::setprecision(3) << elapsed_ms
            << " ms" << std::endl;
  if (world_size > 0 && elapsed_ms > 0.0 && std::isfinite(elapsed_ms)) {
    const double algorithm_bandwidth =
        total_bytes / kBytesPerGB / (elapsed_ms / 1000.0);
    const double bus_bandwidth = algorithm_bandwidth *
                                 static_cast<double>(world_size - 1) /
                                 static_cast<double>(world_size);
    std::cout << "Throughput:            " << std::setprecision(2)
              << bus_bandwidth << " GB/s (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:         " << algorithm_bandwidth << " GB/s"
              << std::endl;
  } else {
    std::cout << "Throughput:            N/A (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:         N/A" << std::endl;
  }

  std::cout.flags(original_flags);
  std::cout.precision(original_precision);
}

void PrintResult(const char *scenario, bool correct, float actual,
                 float expected, size_t recv_count, int world_size,
                 double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== " << scenario
            << " Hybrid CCL ReduceScatter Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Expect:  " << expected << std::endl;
  std::cout << "Actual:  " << actual << std::endl;
  PrintReduceScatterMetrics(recv_count, world_size, elapsed_ms);
}

bool RunReduceScatterExample(int argc, char **argv) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  constexpr int kWarmupIterations = 2;
  constexpr int kProfileIterations = 20;
  constexpr size_t kRecvCount = 1 << 20;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = -1;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));
  if (size <= 0) {
    std::cerr << "Invalid world size for hybrid ReduceScatter." << std::endl;
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
    std::cerr << "Failed to query the hostname for hybrid ReduceScatter."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Bootstrap the native communicator with ReduceScatter itself. Rank 0
  // repeats the same unique ID in every destination block; all other ranks
  // contribute zeros. With only the OpenMPI inter communicator initialized,
  // the CCL provider delegates this call to MPI_Reduce_scatter_block.
  infinicclUniqueId id{};
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  const size_t id_bytes = sizeof(id);
  const size_t world_size = static_cast<size_t>(size);
  if (world_size > std::numeric_limits<size_t>::max() / id_bytes) {
    std::cerr << "ReduceScatter bootstrap buffer size overflows `size_t`."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const size_t id_send_bytes = id_bytes * world_size;
  std::vector<uint8_t> h_id_send(id_send_bytes, 0);
  if (rank == 0) {
    for (int dest_rank = 0; dest_rank < size; ++dest_rank) {
      std::memcpy(h_id_send.data() + static_cast<size_t>(dest_rank) * id_bytes,
                  &id, id_bytes);
    }
  }

  uint8_t *d_id_send = nullptr;
  uint8_t *d_id_recv = nullptr;
  CHECK_RT(Rt,
           Rt::Malloc(reinterpret_cast<void **>(&d_id_send), id_send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_id_recv), id_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_id_send, h_id_send.data(), id_send_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclReduceScatter(d_id_send, d_id_recv, id_bytes,
                                      infinicclUInt8, infinicclSum, comm,
                                      nullptr));
  CHECK_RT(Rt, Rt::Memcpy(&id, d_id_recv, id_bytes, Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Free(d_id_send));
  CHECK_RT(Rt, Rt::Free(d_id_recv));

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  if (kRecvCount > std::numeric_limits<size_t>::max() / world_size ||
      kRecvCount * world_size >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Hybrid ReduceScatter buffer size overflows `size_t`."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const size_t send_count = kRecvCount * world_size;
  const size_t send_bytes = send_count * sizeof(float);
  const size_t recv_bytes = kRecvCount * sizeof(float);
  const size_t local_offset = static_cast<size_t>(rank) * kRecvCount;
  std::vector<float> h_send(send_count);
  std::vector<float> h_recv(kRecvCount, 0.0f);
  FillInput(&h_send, kRecvCount, size, rank);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), recv_bytes));

  std::array<bool, 2> local_correct{true, true};
  std::array<double, 2> elapsed_ms{};
  std::array<float, 2> actual{};
  for (int scenario = 0; scenario < 2; ++scenario) {
    const bool in_place = scenario == 1;
    CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    float *active_recv = in_place ? d_send + local_offset : d_recv;
    auto restore_local_block = [&]() {
      if (in_place) {
        CHECK_RT(Rt, Rt::Memcpy(active_recv, h_send.data() + local_offset,
                                recv_bytes, Rt::MemcpyHostToDevice));
        CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
      }
    };

    for (int i = 0; i < kWarmupIterations; ++i) {
      restore_local_block();
      CHECK_INFINI(infinicclReduceScatter(d_send, active_recv, kRecvCount,
                                          infinicclFloat32, infinicclSum, comm,
                                          nullptr));
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    if (in_place) {
      for (int i = 0; i < kProfileIterations; ++i) {
        restore_local_block();
        Timer timer;
        CHECK_INFINI(infinicclReduceScatter(d_send, active_recv, kRecvCount,
                                            infinicclFloat32, infinicclSum,
                                            comm, nullptr));
        CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
        elapsed_ms[scenario] += timer.ElapsedMs();
      }
      elapsed_ms[scenario] /= static_cast<double>(kProfileIterations);
    } else {
      Timer timer;
      for (int i = 0; i < kProfileIterations; ++i) {
        CHECK_INFINI(infinicclReduceScatter(d_send, active_recv, kRecvCount,
                                            infinicclFloat32, infinicclSum,
                                            comm, nullptr));
      }
      CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
      elapsed_ms[scenario] =
          timer.ElapsedMs() / static_cast<double>(kProfileIterations);
    }

    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), active_recv, recv_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    const float expected = ExpectedValue(rank, size);
    local_correct[scenario] =
        Validator::ValidateResult(h_recv.data(), kRecvCount, expected, rank);
    actual[scenario] = h_recv.front();
  }

  // Reduce both validation flags into every destination block. Every rank then
  // receives the same cluster-wide minimum and exits consistently.
  if (world_size > std::numeric_limits<size_t>::max() / 2 ||
      world_size * 2 > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
    std::cerr << "ReduceScatter status buffer size overflows `size_t`."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::vector<int32_t> h_status_send(world_size * 2, 0);
  for (int dest_rank = 0; dest_rank < size; ++dest_rank) {
    const size_t offset = static_cast<size_t>(dest_rank) * 2;
    h_status_send[offset] = local_correct[0] ? 1 : 0;
    h_status_send[offset + 1] = local_correct[1] ? 1 : 0;
  }
  std::array<int32_t, 2> h_status_recv{};
  int32_t *d_status_send = nullptr;
  int32_t *d_status_recv = nullptr;
  const size_t status_send_bytes =
      h_status_send.size() * sizeof(h_status_send[0]);
  const size_t status_recv_bytes =
      h_status_recv.size() * sizeof(h_status_recv[0]);
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_status_send),
                          status_send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_status_recv),
                          status_recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_status_send, h_status_send.data(),
                          status_send_bytes, Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclReduceScatter(d_status_send, d_status_recv,
                                      h_status_recv.size(), infinicclInt32,
                                      infinicclMin, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_status_recv.data(), d_status_recv,
                          status_recv_bytes, Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  const std::array<bool, 2> global_correct{h_status_recv[0] == 1,
                                           h_status_recv[1] == 1};
  if (rank == 0) {
    const float expected = ExpectedValue(rank, size);
    PrintResult("Out-of-Place", global_correct[0], actual[0], expected,
                kRecvCount, size, elapsed_ms[0]);
    PrintResult("In-Place", global_correct[1], actual[1], expected, kRecvCount,
                size, elapsed_ms[1]);
  }

  CHECK_RT(Rt, Rt::Free(d_status_send));
  CHECK_RT(Rt, Rt::Free(d_status_recv));
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == 0) {
    if (global_correct[0] && global_correct[1]) {
      std::cout << "[Main Process] All hybrid ReduceScatter scenarios passed."
                << std::endl;
    } else {
      std::cerr << "[Main Process] Hybrid ReduceScatter validation failed."
                << std::endl;
    }
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return global_correct[0] && global_correct[1];
}

}  // namespace

int main(int argc, char **argv) {
  return RunReduceScatterExample(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
