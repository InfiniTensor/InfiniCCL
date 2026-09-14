/**
 * InfiniCCL Example: Scatter (OpenMPI + CCL Hybrid)
 *
 * This example first exercises Scatter through its OpenMPI fallback, then
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

void FillScatterInput(std::vector<float> *input, size_t num_elements,
                      int world_size) {
  for (int destination = 0; destination < world_size; ++destination) {
    const size_t offset = static_cast<size_t>(destination) * num_elements;
    std::fill_n(input->begin() + offset, num_elements,
                static_cast<float>(destination + 1));
  }
}

template <typename Rt>
bool CollectValidationReports(bool local_correct, float local_sample, int rank,
                              int world_size, infinicclComm_t comm,
                              std::vector<float> *root_reports) {
  const float report = local_correct ? local_sample : -1.0f;
  float *d_report = nullptr;
  float *d_reports = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_report), sizeof(float)));
  if (rank == kRoot) {
    root_reports->assign(static_cast<size_t>(world_size), 0.0f);
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_reports),
                            static_cast<size_t>(world_size) * sizeof(float)));
  }
  CHECK_RT(
      Rt, Rt::Memcpy(d_report, &report, sizeof(float), Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclGather(d_report, d_reports, 1, infinicclFloat32, kRoot,
                               comm, nullptr));

  int32_t global_status = 1;
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(root_reports->data(), d_reports,
                            static_cast<size_t>(world_size) * sizeof(float),
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    for (int source = 0; source < world_size; ++source) {
      if ((*root_reports)[static_cast<size_t>(source)] !=
          static_cast<float>(source + 1)) {
        global_status = 0;
      }
    }
  }
  CHECK_INFINI(infinicclBroadcast(&global_status, &global_status, 1,
                                  infinicclInt32, kRoot, comm, nullptr));

  CHECK_RT(Rt, Rt::Free(d_report));
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_reports));
  }
  return global_status == 1;
}

void PrintScatterMetrics(size_t num_elements, int world_size,
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
  std::cout << "Total data at root: "
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

void PrintResult(bool correct, const std::vector<float> &samples,
                 size_t num_elements, int world_size, double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== Hybrid CCL Scatter Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Root rank: " << kRoot << std::endl;
  std::cout << "Sample receive blocks: ";
  for (int rank = 0; rank < std::min(world_size, 4); ++rank) {
    std::cout << "[r" << rank << ": " << samples[static_cast<size_t>(rank)]
              << "] ";
  }
  std::cout << std::endl;
  PrintScatterMetrics(num_elements, world_size, elapsed_ms);
}

bool RunScatterExample(int argc, char **argv) {
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
    std::cerr << "Invalid world size for hybrid Scatter." << std::endl;
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
    std::cerr << "Failed to query the hostname for hybrid Scatter."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  // Before a native communicator exists, the selected CCL Scatter provider
  // delegates this rank token distribution to the OpenMPI inter communicator.
  std::vector<float> h_bootstrap_send;
  if (rank == kRoot) {
    h_bootstrap_send.resize(static_cast<size_t>(size));
    for (int destination = 0; destination < size; ++destination) {
      h_bootstrap_send[static_cast<size_t>(destination)] =
          static_cast<float>(destination + 1);
    }
  }
  float h_bootstrap_recv = 0.0f;
  float *d_bootstrap_send = nullptr;
  float *d_bootstrap_recv = nullptr;
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_send),
                            static_cast<size_t>(size) * sizeof(float)));
    CHECK_RT(Rt, Rt::Memcpy(d_bootstrap_send, h_bootstrap_send.data(),
                            static_cast<size_t>(size) * sizeof(float),
                            Rt::MemcpyHostToDevice));
  }
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_bootstrap_recv),
                          sizeof(float)));
  CHECK_INFINI(infinicclScatter(d_bootstrap_send, d_bootstrap_recv, 1,
                                infinicclFloat32, kRoot, comm, nullptr));
  CHECK_RT(Rt, Rt::Memcpy(&h_bootstrap_recv, d_bootstrap_recv, sizeof(float),
                          Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  const bool bootstrap_local_correct = h_bootstrap_recv == rank + 1.0f;
  std::vector<float> bootstrap_reports;
  const bool bootstrap_correct =
      CollectValidationReports<Rt>(bootstrap_local_correct, h_bootstrap_recv,
                                   rank, size, comm, &bootstrap_reports);
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_bootstrap_send));
  }
  CHECK_RT(Rt, Rt::Free(d_bootstrap_recv));

  if (!bootstrap_correct) {
    if (rank == kRoot) {
      std::cerr << "OpenMPI Scatter fallback validation failed." << std::endl;
    }
    CHECK_INFINI(infinicclCommDestroy(comm));
    CHECK_INFINI(infinicclFinalize());
    return false;
  }
  if (rank == kRoot) {
    std::cout << "OpenMPI Scatter fallback validation passed." << std::endl;
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
    std::cerr << "Hybrid Scatter buffer size overflows `size_t`." << std::endl;
    std::exit(EXIT_FAILURE);
  }
  const size_t rank_bytes = kNumElements * sizeof(float);
  const size_t total_elements = kNumElements * world_size;
  const size_t total_bytes = total_elements * sizeof(float);
  std::vector<float> h_send;
  if (rank == kRoot) {
    h_send.resize(total_elements, 0.0f);
    FillScatterInput(&h_send, kNumElements, size);
  }
  std::vector<float> h_recv(kNumElements, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
    CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
  }
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), rank_bytes));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  for (int i = 0; i < kWarmupIterations; ++i) {
    CHECK_INFINI(infinicclScatter(d_send, d_recv, kNumElements,
                                  infinicclFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < kProfileIterations; ++i) {
    CHECK_INFINI(infinicclScatter(d_send, d_recv, kNumElements,
                                  infinicclFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(kProfileIterations);

  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, rank_bytes,
                          Rt::MemcpyDeviceToHost));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const bool local_correct = Validator::ValidateResult(
      h_recv.data(), kNumElements, static_cast<float>(rank + 1), rank);
  std::vector<float> validation_reports;
  const bool correct = CollectValidationReports<Rt>(
      local_correct, h_recv.front(), rank, size, comm, &validation_reports);

  if (rank == kRoot) {
    PrintResult(correct, validation_reports, kNumElements, size, elapsed_ms);
  }

  if (rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_send));
  }
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (rank == kRoot) {
    if (correct) {
      std::cout << "[Main Process] Hybrid CCL Scatter validation passed."
                << std::endl;
    } else {
      std::cerr << "[Main Process] Hybrid CCL Scatter validation failed."
                << std::endl;
    }
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return correct;
}

}  // namespace

int main(int argc, char **argv) {
  return RunScatterExample(argc, argv) ? EXIT_SUCCESS : EXIT_FAILURE;
}
