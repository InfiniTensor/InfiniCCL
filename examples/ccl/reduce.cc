/**
 * InfiniCCL Example: Thread-per-GPU Single-Node Reduce
 *
 * This example creates one native CCL rank per GPU and validates a rooted
 * reduction without an MPI launcher.
 */

#include <unistd.h>

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

struct RunState {
  std::atomic<bool> correct{true};
  std::atomic<int> completed{0};
};

struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t count;
  int warmup_iterations;
  int profile_iterations;
  RunState *state;
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

  std::cout << "\n=== CCL Reduce Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Root rank: " << kRoot << std::endl;
  std::cout << "Expect:  " << expected << std::endl;
  std::cout << "Actual:  " << actual << std::endl;
  PrintReduceMetrics(count, elapsed_ms);
}

void WaitForAll(RunState *state, int world_size) {
  state->completed.fetch_add(1, std::memory_order_acq_rel);
  while (state->completed.load(std::memory_order_acquire) < world_size) {
    std::this_thread::yield();
  }
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for the Reduce worker."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << args.rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << args.rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  const bool is_root = args.rank == kRoot;
  const size_t total_bytes = args.count * sizeof(float);
  std::vector<float> h_send(args.count, static_cast<float>(args.rank + 1));
  std::vector<float> h_recv(is_root ? args.count : 0, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  for (int i = 0; i < args.warmup_iterations; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, args.count, infinicclFloat32,
                                 infinicclSum, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < args.profile_iterations; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, args.count, infinicclFloat32,
                                 infinicclSum, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(args.profile_iterations);

  bool local_correct = true;
  const float expected = ExpectedValue(args.size);
  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    local_correct = Validator::ValidateResult(
        h_recv.data(), args.count, expected, args.rank, false, "Reduce");
    if (!local_correct) {
      args.state->correct.store(false, std::memory_order_relaxed);
    }
  }

  WaitForAll(args.state, args.size);
  if (is_root) {
    PrintResult(args.state->correct.load(std::memory_order_acquire), expected,
                h_recv.front(), args.count, elapsed_ms);
  }

  CHECK_RT(Rt, Rt::Free(d_send));
  if (is_root) {
    CHECK_RT(Rt, Rt::Free(d_recv));
  }
  CHECK_INFINI(infinicclCommDestroy(comm));
}

void PrintUsage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "Options:\n"
            << "  -g <num_gpus>      Number of GPUs (default: 8)\n"
            << "  -w <warmup_iters>  Warm-up iterations (default: 2)\n"
            << "  -p <profile_iters> Profile iterations (default: 20)\n"
            << "  -n <count>         Elements reduced per rank (default: "
               "1048576)\n";
}

}  // namespace

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iterations = 2;
  int profile_iterations = 20;
  size_t count = 1 << 20;

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
        parsed = ParsePositiveNumber(optarg, &count);
        break;
      case 'h':
        PrintUsage(argv[0]);
        return EXIT_SUCCESS;
      default:
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!parsed) {
      std::cerr << "Invalid positive numeric option for Reduce." << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (optind != argc) {
    std::cerr << "Unexpected positional argument for Reduce." << std::endl;
    return EXIT_FAILURE;
  }
  if (count > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Reduce buffer size overflows `size_t`." << std::endl;
    return EXIT_FAILURE;
  }

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for Reduce." << std::endl;
    return EXIT_FAILURE;
  }
  hostname.back() = '\0';
  std::cout << "[Main Process] Host: " << hostname.data()
            << " | Target GPUs: " << num_gpus << std::endl;
  std::cout << "[Main Process] Count: " << count
            << " floats | Warmup: " << warmup_iterations
            << " | Profile: " << profile_iterations << " | Root: " << kRoot
            << std::endl;

  infinicclUniqueId shared_id{};
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  RunState state;
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);
  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,  num_gpus,          shared_id,
                    count, warmup_iterations, profile_iterations,
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
    std::cout << "[Main Process] CCL Reduce validation passed." << std::endl;
  } else {
    std::cerr << "[Main Process] CCL Reduce validation failed." << std::endl;
  }
  std::cout
      << "[Main Process] All worker threads joined. InfiniCCL finalized safely."
      << std::endl;
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
