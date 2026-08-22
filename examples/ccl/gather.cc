/**
 * InfiniCCL Example: Thread-per-GPU Single-Node Gather
 *
 * This example creates one native CCL rank per GPU and gathers one block from
 * every rank into rank 0 using grouped point-to-point operations.
 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

// Public API
#include "infiniccl.h"

// Example-Specific Utilities
#include "utils.h"

// Internal Headers (Accessible via example-specific include paths, technically
// not public APIs)
#include "backend_manifest.h"

using namespace infini::ccl;

namespace {

constexpr int kRoot = 0;

struct ScenarioState {
  std::atomic<bool> correct{true};
  std::atomic<int> completed{0};
};

struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iterations;
  int profile_iterations;
  ScenarioState *state;
};

template <typename T>
bool ParsePositiveNumber(const char *text, T *value) {
  if (!text || !value) {
    return false;
  }

  T parsed{};
  const char *end = text + std::strlen(text);
  const auto result = std::from_chars(text, end, parsed);
  if (result.ec != std::errc{} || result.ptr != end || parsed <= 0) {
    return false;
  }

  *value = parsed;
  return true;
}

bool ValidateGather(const std::vector<float> &result, size_t num_elements,
                    int world_size) {
  bool correct = true;
  for (int source = 0; source < world_size; ++source) {
    const size_t offset = static_cast<size_t>(source) * num_elements;
    const bool block_correct =
        Validator::ValidateResult(result.data() + offset, num_elements,
                                  static_cast<float>(source + 1), kRoot);
    correct = block_correct && correct;
  }
  return correct;
}

void PrintGatherMetrics(size_t num_elements, int world_size,
                        double elapsed_ms) {
  constexpr double kBytesPerMiB = 1024.0 * 1024.0;
  constexpr double kBytesPerGB = 1.0e9;
  const double rank_bytes = static_cast<double>(num_elements) * sizeof(float);
  const double gathered_bytes = rank_bytes * static_cast<double>(world_size);
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << "Data size per rank: " << num_elements << " floats ("
            << std::fixed << std::setprecision(2) << rank_bytes / kBytesPerMiB
            << " MiB)" << std::endl;
  std::cout << "Total data at root: "
            << num_elements * static_cast<size_t>(world_size) << " floats ("
            << gathered_bytes / kBytesPerMiB << " MiB)" << std::endl;
  std::cout << "Time:           " << std::setprecision(3) << elapsed_ms << " ms"
            << std::endl;
  if (elapsed_ms > 0.0 && std::isfinite(elapsed_ms)) {
    const double algorithm_bandwidth =
        gathered_bytes / kBytesPerGB / (elapsed_ms / 1000.0);
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

void WaitForAll(ScenarioState *state, int world_size) {
  state->completed.fetch_add(1, std::memory_order_acq_rel);
  while (state->completed.load(std::memory_order_acquire) < world_size) {
    std::this_thread::yield();
  }
}

void PrintResult(bool correct, const std::vector<float> &result,
                 size_t num_elements, int world_size, double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== CCL Gather Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Root rank: " << kRoot << std::endl;
  std::cout << "Sample receive blocks: ";
  for (int source = 0; source < std::min(world_size, 4); ++source) {
    const size_t offset = static_cast<size_t>(source) * num_elements;
    std::cout << "[r" << source << ": " << result[offset] << "] ";
  }
  std::cout << std::endl;
  PrintGatherMetrics(num_elements, world_size, elapsed_ms);
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for the Gather worker."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << args.rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << args.rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  const size_t rank_bytes = args.num_elements * sizeof(float);
  const size_t gathered_elements =
      args.num_elements * static_cast<size_t>(args.size);
  const size_t gathered_bytes = gathered_elements * sizeof(float);
  std::vector<float> h_send(args.num_elements,
                            static_cast<float>(args.rank + 1));
  std::vector<float> h_recv;
  if (args.rank == kRoot) {
    h_recv.resize(gathered_elements, 0.0f);
  }

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), rank_bytes));
  if (args.rank == kRoot) {
    CHECK_RT(Rt,
             Rt::Malloc(reinterpret_cast<void **>(&d_recv), gathered_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), rank_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  for (int i = 0; i < args.warmup_iterations; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, args.num_elements,
                                 infinicclFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < args.profile_iterations; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, args.num_elements,
                                 infinicclFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(args.profile_iterations);

  if (args.rank == kRoot) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, gathered_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    args.state->correct.store(
        ValidateGather(h_recv, args.num_elements, args.size),
        std::memory_order_release);
  }

  WaitForAll(args.state, args.size);
  if (args.rank == kRoot) {
    PrintResult(args.state->correct.load(std::memory_order_acquire), h_recv,
                args.num_elements, args.size, elapsed_ms);
  }

  CHECK_RT(Rt, Rt::Free(d_send));
  if (args.rank == kRoot) {
    CHECK_RT(Rt, Rt::Free(d_recv));
  }
  CHECK_INFINI(infinicclCommDestroy(comm));
}

void PrintUsage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "Options:\n"
            << "  -g <num_gpus>      Number of GPUs (default: 8)\n"
            << "  -w <warmup_iters>  Warmup iterations (default: 2)\n"
            << "  -p <profile_iters> Profile iterations (default: 20)\n"
            << "  -n <num_elements>  Elements contributed by each rank "
               "(default: 1048576)\n";
}

}  // namespace

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iterations = 2;
  int profile_iterations = 20;
  size_t num_elements = 1 << 20;

  int opt = 0;
  while ((opt = getopt(argc, argv, "g:w:p:n:h")) != -1) {
    bool parsed = false;
    switch (opt) {
      case 'g':
        parsed = ParsePositiveNumber(optarg, &num_gpus);
        break;
      case 'w':
        parsed = ParsePositiveNumber(optarg, &warmup_iterations);
        break;
      case 'p':
        parsed = ParsePositiveNumber(optarg, &profile_iterations);
        break;
      case 'n':
        parsed = ParsePositiveNumber(optarg, &num_elements);
        break;
      case 'h':
        PrintUsage(argv[0]);
        return EXIT_SUCCESS;
      default:
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!parsed) {
      std::cerr << "Invalid positive numeric option for Gather." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (optind != argc) {
    std::cerr << "Unexpected positional argument for Gather." << std::endl;
    return EXIT_FAILURE;
  }
  if (static_cast<size_t>(num_gpus) >
          std::numeric_limits<size_t>::max() / num_elements ||
      num_elements * static_cast<size_t>(num_gpus) >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Gather buffer size overflows `size_t`." << std::endl;
    return EXIT_FAILURE;
  }

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for Gather." << std::endl;
    return EXIT_FAILURE;
  }
  hostname.back() = '\0';
  std::cout << "[Main Process] Host: " << hostname.data()
            << " | Target GPUs: " << num_gpus << std::endl;
  std::cout << "[Main Process] Elements per rank: " << num_elements
            << " floats | Warmup: " << warmup_iterations
            << " | Profile: " << profile_iterations << std::endl;

  infinicclUniqueId shared_id{};
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  ScenarioState state;
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);
  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,          shared_id,
                    num_elements, warmup_iterations, profile_iterations,
                    &state};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  const bool correct = state.correct.load(std::memory_order_acquire);
  if (correct) {
    std::cout << "[Main Process] CCL Gather validation passed." << std::endl;
  } else {
    std::cerr << "[Main Process] CCL Gather validation failed." << std::endl;
  }
  std::cout
      << "[Main Process] All worker threads joined. InfiniCCL finalized safely."
      << std::endl;
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
