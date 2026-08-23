/**
 * InfiniCCL Example: Reduce (OpenMPI + CCL Hybrid)
 *
 * This example uses an OpenMPI inter communicator to distribute a native CCL
 * unique ID, then validates a rooted GPU reduction on the native communicator.
 */

#include <unistd.h>

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

float ExpectedValue(int world_size) {
  return static_cast<float>(world_size) *
         (static_cast<float>(world_size) + 1.0f) / 2.0f;
}

void PrintReduceMetrics(size_t count, double elapsed_ms) {
  constexpr double kBytesPerMiB = 1024.0 * 1024.0;
  constexpr double kBytesPerGB = 1.0e9;
  const double payload_bytes = static_cast<double>(count) * sizeof(float);
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << "Data size:      " << count << " floats (" << std::fixed
            << std::setprecision(2) << payload_bytes / kBytesPerMiB << " MiB)"
            << std::endl;
  std::cout << "Time:           " << std::setprecision(3) << elapsed_ms << " ms"
            << std::endl;
  if (elapsed_ms > 0.0 && std::isfinite(elapsed_ms)) {
    const double algorithm_bandwidth =
        payload_bytes / kBytesPerGB / (elapsed_ms / 1000.0);
    // `nccl-tests` uses a bus-bandwidth correction factor of 1 for Reduce.
    const double bus_bandwidth = algorithm_bandwidth;
    std::cout << "Throughput:     " << std::setprecision(2) << bus_bandwidth
              << " GB/s (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:  " << std::setprecision(2)
              << algorithm_bandwidth << " GB/s" << std::endl;
  } else {
    std::cout << "Throughput:     N/A (Bus BW)" << std::endl;
    std::cout << "Alg Bandwidth:  N/A" << std::endl;
  }

  std::cout.flags(original_flags);
  std::cout.precision(original_precision);
}

void PrintResult(bool correct, float expected, float actual, size_t count,
                 double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== Hybrid CCL Reduce Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Root rank: " << kRoot << std::endl;
  std::cout << "Expect:  " << expected << std::endl;
  std::cout << "Actual:  " << actual << std::endl;
  PrintReduceMetrics(count, elapsed_ms);
}

bool RunReduceExample(int argc, char **argv) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  constexpr int kWarmupIterations = 2;
  constexpr int kProfileIterations = 20;
  constexpr size_t kCount = 1 << 20;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = -1;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));
  if (size <= 0 || kRoot >= size) {
    std::cerr << "Invalid world size for hybrid Reduce." << std::endl;
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
    std::cerr << "Failed to query the hostname for hybrid Reduce." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Exercise Reduce before the native communicator exists. The CCL backend
  // must delegate this call to the OpenMPI inter communicator.
  constexpr size_t kBootstrapCount = 1;
  const float bootstrap_send = static_cast<float>(rank + 1);
  float bootstrap_recv = 0.0f;
  float *d_bootstrap_send = nullptr;
  float *d_bootstrap_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_send),
                          kBootstrapCount * sizeof(float)));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_recv),
                            kBootstrapCount * sizeof(float)));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_bootstrap_send, &bootstrap_send, sizeof(float),
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclReduce(d_bootstrap_send, d_bootstrap_recv,
                               kBootstrapCount, infinicclFloat32, infinicclSum,
                               kRoot, comm, nullptr));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(&bootstrap_recv, d_bootstrap_recv, sizeof(float),
                            Rt::MemcpyDeviceToHost));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  int32_t bootstrap_status =
      rank != kRoot || bootstrap_recv == ExpectedValue(size) ? 1 : 0;
  CHECK_INFINI(infinicclBroadcast(&bootstrap_status, &bootstrap_status, 1,
                                  infinicclInt32, kRoot, comm, nullptr));
  CHECK_RT(Rt, Rt::Free(d_bootstrap_send));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_bootstrap_recv));
  }

  infinicclUniqueId id{};
  if (rank == kRoot) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclChar, kRoot,
                                  comm, nullptr));
  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  if (kCount > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Hybrid Reduce buffer size overflows `size_t`." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const bool is_root = rank == kRoot;
  const size_t total_bytes = kCount * sizeof(float);
  std::vector<float> h_send(kCount, static_cast<float>(rank + 1));
  std::vector<float> h_recv(is_root ? kCount : 0, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  if (is_root) {
    std::cout << "\n=== Performing Hybrid CCL Reduce on GPU Memory ==="
              << std::endl;
    std::cout << "Operation: Sum" << std::endl;
    std::cout << "Root rank: " << kRoot << std::endl;
    std::cout << "Warm-up iterations: " << kWarmupIterations << std::endl;
    std::cout << "Profile iterations: " << kProfileIterations << std::endl;
  }

  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, kCount, infinicclFloat32,
                                 infinicclSum, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < kProfileIterations; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, kCount, infinicclFloat32,
                                 infinicclSum, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(kProfileIterations);

  bool correct = bootstrap_status == 1;
  const float expected = ExpectedValue(size);
  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    correct = Validator::ValidateResult(h_recv.data(), kCount, expected, rank,
                                        false, "Reduce") &&
              correct;
    PrintResult(correct, expected, h_recv.front(), kCount, elapsed_ms);
  }

  int32_t validation_status = correct ? 1 : 0;
  CHECK_INFINI(infinicclBroadcast(&validation_status, &validation_status, 1,
                                  infinicclInt32, kRoot, comm, nullptr));
  correct = validation_status == 1;

  CHECK_RT(Rt, Rt::Free(d_send));
  if (is_root) {
    CHECK_RT(Rt, Rt::Free(d_recv));
  }
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (is_root) {
    if (correct) {
      std::cout << "[Main Process] Hybrid CCL Reduce validation passed."
                << std::endl;
    } else {
      std::cerr << "[Main Process] Hybrid CCL Reduce validation failed."
                << std::endl;
    }
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return correct;
}

}  // namespace

int main(int argc, char **argv) {
  return RunReduceExample(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
