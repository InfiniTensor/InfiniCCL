/**
 * InfiniCCL Example: AllGather (OpenMPI + CCL Hybrid)
 *
 * This example first uses AllGather through an OpenMPI inter communicator to
 * distribute a native CCL unique ID. It then initializes a native CCL
 * communicator and validates out-of-place and in-place GPU AllGather.
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

bool ValidateAllGather(const std::vector<float> &result, size_t num_elements,
                       int world_size, int rank) {
  bool correct = true;
  for (int src_rank = 0; src_rank < world_size; ++src_rank) {
    const size_t offset = static_cast<size_t>(src_rank) * num_elements;
    const bool block_correct =
        Validator::ValidateResult(result.data() + offset, num_elements,
                                  static_cast<float>(src_rank + 1), rank);
    correct = block_correct && correct;
  }
  return correct;
}

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

void PrintResult(const char *scenario, bool correct,
                 const std::vector<float> &result, size_t num_elements,
                 int world_size, double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== " << scenario
            << " Hybrid CCL AllGather Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Sample blocks: ";
  for (int src_rank = 0; src_rank < world_size && src_rank < 4; ++src_rank) {
    const size_t offset = static_cast<size_t>(src_rank) * num_elements;
    std::cout << "[r" << src_rank << ": " << result[offset] << "] ";
  }
  std::cout << std::endl;
  PrintAllGatherMetrics(num_elements, world_size, elapsed_ms);
}

bool RunAllGatherExample(int argc, char **argv) {
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
    std::cerr << "Invalid world size for hybrid AllGather." << std::endl;
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
    std::cerr << "Failed to query the hostname for hybrid AllGather."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Bootstrap the native communicator with AllGather itself. At this point
  // only the OpenMPI inter communicator exists, so the CCL provider delegates
  // this call to the existing OpenMPI staging implementation.
  infinicclUniqueId id{};
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  const size_t id_bytes = sizeof(id);
  if (static_cast<size_t>(size) >
      std::numeric_limits<size_t>::max() / id_bytes) {
    std::cerr << "AllGather bootstrap buffer size overflows `size_t`."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const size_t gathered_id_bytes = id_bytes * static_cast<size_t>(size);

  unsigned char *d_id_send = nullptr;
  unsigned char *d_id_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_id_send), id_bytes));
  CHECK_RT(
      Rt, Rt::Malloc(reinterpret_cast<void **>(&d_id_recv), gathered_id_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_id_send, &id, id_bytes, Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclAllGather(d_id_send, d_id_recv, id_bytes, infinicclChar,
                                  comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(&id, d_id_recv, id_bytes, Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Free(d_id_send));
  CHECK_RT(Rt, Rt::Free(d_id_recv));

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  if (kNumElements > std::numeric_limits<size_t>::max() / sizeof(float) ||
      static_cast<size_t>(size) >
          std::numeric_limits<size_t>::max() / kNumElements ||
      kNumElements * static_cast<size_t>(size) >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Hybrid AllGather buffer size overflows `size_t`."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  const size_t send_bytes = kNumElements * sizeof(float);
  const size_t recv_elements = kNumElements * static_cast<size_t>(size);
  const size_t recv_bytes = recv_elements * sizeof(float);
  std::vector<float> h_send(kNumElements, static_cast<float>(rank + 1));
  std::vector<float> h_recv(recv_elements, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));

  std::array<bool, 2> local_correct{true, true};
  std::array<double, 2> elapsed_ms{};
  for (int scenario = 0; scenario < 2; ++scenario) {
    const bool in_place = scenario == 1;
    std::fill(h_recv.begin(), h_recv.end(), 0.0f);
    const size_t local_offset = static_cast<size_t>(rank) * kNumElements;
    if (in_place) {
      std::fill_n(h_recv.begin() + local_offset, kNumElements,
                  static_cast<float>(rank + 1));
    }

    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    const void *active_send =
        in_place ? static_cast<const void *>(d_recv + local_offset)
                 : static_cast<const void *>(d_send);

    for (int i = 0; i < kWarmupIterations; ++i) {
      CHECK_INFINI(infinicclAllGather(active_send, d_recv, kNumElements,
                                      infinicclFloat32, comm, nullptr));
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    Timer timer;
    for (int i = 0; i < kProfileIterations; ++i) {
      CHECK_INFINI(infinicclAllGather(active_send, d_recv, kNumElements,
                                      infinicclFloat32, comm, nullptr));
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    elapsed_ms[scenario] =
        timer.ElapsedMs() / static_cast<double>(kProfileIterations);

    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    local_correct[scenario] =
        ValidateAllGather(h_recv, kNumElements, size, rank);
  }

  // Gather both local validation flags so rank 0 reports cluster-wide results
  // and every rank reaches communicator destruction only after validation.
  std::array<int32_t, 2> h_status_send{local_correct[0] ? 1 : 0,
                                       local_correct[1] ? 1 : 0};
  std::vector<int32_t> h_status_recv(static_cast<size_t>(size) * 2, 0);
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
  CHECK_INFINI(infinicclAllGather(d_status_send, d_status_recv,
                                  h_status_send.size(), infinicclInt32, comm,
                                  nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_status_recv.data(), d_status_recv,
                          status_recv_bytes, Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  std::array<bool, 2> global_correct{true, true};
  for (int src_rank = 0; src_rank < size; ++src_rank) {
    for (int scenario = 0; scenario < 2; ++scenario) {
      const size_t index = static_cast<size_t>(src_rank) * 2 + scenario;
      global_correct[scenario] =
          global_correct[scenario] && h_status_recv[index] == 1;
    }
  }

  if (rank == 0) {
    // Rank 0's gathered payload is representative after cluster-wide status
    // aggregation because every rank validates the complete result.
    PrintResult("Out-of-Place", global_correct[0], h_recv, kNumElements, size,
                elapsed_ms[0]);
    PrintResult("In-Place", global_correct[1], h_recv, kNumElements, size,
                elapsed_ms[1]);
  }

  CHECK_RT(Rt, Rt::Free(d_status_send));
  CHECK_RT(Rt, Rt::Free(d_status_recv));
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == 0) {
    if (global_correct[0] && global_correct[1]) {
      std::cout << "[Main Process] All hybrid AllGather scenarios passed."
                << std::endl;
    } else {
      std::cerr << "[Main Process] Hybrid AllGather validation failed."
                << std::endl;
    }
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return global_correct[0] && global_correct[1];
}

}  // namespace

int main(int argc, char **argv) {
  return RunAllGatherExample(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
