/**
 * InfiniCCL Example: ReduceScatter
 * * This example demonstrates the planned API for performing a
 * collective sum-reduction across multiple GPUs and nodes.
 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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

bool RunReduceScatterExample(int argc, char **argv, int warmup_iter,
                             int profile_iter, size_t recv_count) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = -1;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));
  if (size <= 0) {
    std::cerr << "Invalid world size for ReduceScatter." << std::endl;
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
    std::cerr << "Failed to query the hostname for ReduceScatter." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';

  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  // Setup Communicator
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  const size_t world_size = static_cast<size_t>(size);
  if (recv_count > std::numeric_limits<size_t>::max() / world_size) {
    std::cerr << "ReduceScatter element count overflows `size_t`." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const size_t send_count = recv_count * world_size;
  if (send_count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "ReduceScatter buffer size overflows `size_t`." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Prepare Data
  std::vector<float> h_send(send_count);
  std::vector<float> h_recv(recv_count, 0.0f);

  // Give each destination block a distinct reduced value so validation also
  // checks that the scattered block matches this rank.
  for (int dest_rank = 0; dest_rank < size; ++dest_rank) {
    const float value =
        static_cast<float>(rank + 1) * static_cast<float>(dest_rank + 1);
    const size_t offset = static_cast<size_t>(dest_rank) * recv_count;
    std::fill_n(h_send.begin() + offset, recv_count, value);
  }

  float *d_send = nullptr;
  float *d_recv = nullptr;
  size_t send_bytes = send_count * sizeof(*d_send);
  size_t recv_bytes = recv_count * sizeof(*d_recv);

  CHECK_RT(Rt, Rt::Malloc((void **)&d_send, send_bytes));
  CHECK_RT(Rt, Rt::Malloc((void **)&d_recv, recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                          Rt::MemcpyHostToDevice));

  if (rank == 0) {
    std::cout << "\n=== Performing ReduceScatter on GPU Memory ==="
              << std::endl;
    std::cout << "Recv data size per rank: " << recv_count << " floats ("
              << recv_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Send data size per rank: " << send_count << " floats ("
              << send_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Operation: Sum" << std::endl;
    std::cout << "Warm-up iterations: " << warmup_iter << std::endl;
    std::cout << "Profile iterations: " << profile_iter << std::endl;
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up and D2H transfer the answer.
  CHECK_INFINI(infinicclReduceScatter(d_send, d_recv, recv_count,
                                      infinicclFloat32, infinicclSum, comm,
                                      nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  for (int i = 1; i < warmup_iter; ++i) {
    CHECK_INFINI(infinicclReduceScatter(d_send, d_recv, recv_count,
                                        infinicclFloat32, infinicclSum, comm,
                                        nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling
  Timer timer;

  for (int i = 0; i < profile_iter; ++i) {
    CHECK_INFINI(infinicclReduceScatter(d_send, d_recv, recv_count,
                                        infinicclFloat32, infinicclSum, comm,
                                        nullptr));
  }

  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                          Rt::MemcpyDeviceToHost));

  double elapsed = timer.ElapsedMs() / static_cast<double>(profile_iter);

  // Result Validation:
  const float rank_sum =
      static_cast<float>(size) * (static_cast<float>(size) + 1.0f) / 2.0f;
  const float expected = rank_sum * static_cast<float>(rank + 1);

  const bool correct = Validator::ValidateResult(
      h_recv.data(), recv_count, expected, rank, true, "ReduceScatter");

  // Metrics Reporting (Only from rank 0 for cleaner output)
  if (rank == 0) {
    PrintReduceScatterMetrics(recv_count, size, elapsed);
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

}  // namespace

int main(int argc, char **argv) {
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t recv_count = 1 << 20;

  const bool correct = RunReduceScatterExample(argc, argv, warmup_iters,
                                               profile_iters, recv_count);
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
