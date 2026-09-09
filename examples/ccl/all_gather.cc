/**
 * InfiniCCL Example: Thread-per-GPU Single-Node AllGather
 *
 * This example creates one native CCL rank per GPU, then validates
 * out-of-place and in-place AllGather without an MPI launcher.
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

struct ScenarioState {
  std::atomic<bool> correct{true};
  std::atomic<int> completed{0};
};

struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
  ScenarioState *out_of_place;
  ScenarioState *in_place;
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
            << " CCL AllGather Results ===" << std::endl;
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

void WaitForScenario(ScenarioState *state, int world_size) {
  state->completed.fetch_add(1, std::memory_order_release);
  while (state->completed.load(std::memory_order_acquire) < world_size) {
    std::this_thread::yield();
  }
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

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for the AllGather worker."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';

  std::cout << "[Rank " << args.rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << args.rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  const size_t send_bytes = args.num_elements * sizeof(float);
  const size_t recv_elements =
      args.num_elements * static_cast<size_t>(args.size);
  const size_t recv_bytes = recv_elements * sizeof(float);

  std::vector<float> h_send(args.num_elements,
                            static_cast<float>(args.rank + 1));
  std::vector<float> h_recv(recv_elements, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), recv_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));

  auto run_scenario = [&](bool in_place) {
    std::fill(h_recv.begin(), h_recv.end(), 0.0f);
    const size_t local_offset =
        static_cast<size_t>(args.rank) * args.num_elements;
    if (in_place) {
      std::fill_n(h_recv.begin() + local_offset, args.num_elements,
                  static_cast<float>(args.rank + 1));
    }

    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    const void *active_send =
        in_place ? static_cast<const void *>(d_recv + local_offset)
                 : static_cast<const void *>(d_send);

    for (int i = 0; i < args.warmup_iter; ++i) {
      CHECK_INFINI(infinicclAllGather(active_send, d_recv, args.num_elements,
                                      infinicclFloat32, comm, nullptr));
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    Timer timer;
    for (int i = 0; i < args.profile_iter; ++i) {
      CHECK_INFINI(infinicclAllGather(active_send, d_recv, args.num_elements,
                                      infinicclFloat32, comm, nullptr));
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    const double elapsed_ms =
        timer.ElapsedMs() / static_cast<double>(args.profile_iter);

    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    const bool correct =
        ValidateAllGather(h_recv, args.num_elements, args.size, args.rank);
    ScenarioState *state = in_place ? args.in_place : args.out_of_place;
    if (!correct) {
      state->correct.store(false, std::memory_order_relaxed);
    }
    WaitForScenario(state, args.size);

    if (args.rank == 0) {
      PrintResult(in_place ? "In-Place" : "Out-of-Place",
                  state->correct.load(std::memory_order_acquire), h_recv,
                  args.num_elements, args.size, elapsed_ms);
    }
  };

  run_scenario(false);
  run_scenario(true);

  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

void PrintUsage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "Options:\n"
            << "  -g <num_gpus>        Number of GPUs (default: 8)\n"
            << "  -w <warmup_iters>    Warmup iterations (default: 2)\n"
            << "  -p <profile_iters>   Profile iterations (default: 20)\n"
            << "  -n <num_elements>    Elements per rank (default: 1048576)\n";
}

}  // namespace

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  int opt = 0;
  while ((opt = getopt(argc, argv, "g:w:p:n:h")) != -1) {
    bool parsed = false;
    switch (opt) {
      case 'g':
        parsed = ParsePositiveNumber(optarg, &num_gpus);
        break;
      case 'w':
        parsed = ParsePositiveNumber(optarg, &warmup_iters);
        break;
      case 'p':
        parsed = ParsePositiveNumber(optarg, &profile_iters);
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
      std::cerr << "Invalid positive numeric option for AllGather."
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (optind != argc) {
    std::cerr << "Unexpected positional argument for AllGather." << std::endl;
    return EXIT_FAILURE;
  }
  if (num_elements > std::numeric_limits<size_t>::max() / sizeof(float) ||
      static_cast<size_t>(num_gpus) >
          std::numeric_limits<size_t>::max() / num_elements ||
      num_elements * static_cast<size_t>(num_gpus) >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "AllGather buffer size overflows `size_t`." << std::endl;
    return EXIT_FAILURE;
  }

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for AllGather." << std::endl;
    return EXIT_FAILURE;
  }
  hostname.back() = '\0';
  std::cout << "[Main Process] Host: " << hostname.data()
            << " | Target GPUs: " << num_gpus << std::endl;

  infinicclUniqueId shared_id{};
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  ScenarioState out_of_place;
  ScenarioState in_place;
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);
  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,      shared_id,     num_elements,
                    warmup_iters, profile_iters, &out_of_place, &in_place};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  const bool correct = out_of_place.correct.load(std::memory_order_acquire) &&
                       in_place.correct.load(std::memory_order_acquire);
  if (correct) {
    std::cout << "[Main Process] All AllGather scenarios passed." << std::endl;
  } else {
    std::cerr << "[Main Process] AllGather validation failed." << std::endl;
  }
  std::cout
      << "[Main Process] All worker threads joined. InfiniCCL finalized safely."
      << std::endl;
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
