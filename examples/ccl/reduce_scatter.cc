/**
 * InfiniCCL Example: Thread-per-GPU Single-Node ReduceScatter
 *
 * This example creates one native CCL rank per GPU, then validates
 * out-of-place and canonical in-place ReduceScatter without an MPI launcher.
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
  size_t recv_count;
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

void WaitForScenario(ScenarioState *state, int world_size) {
  state->completed.fetch_add(1, std::memory_order_release);
  while (state->completed.load(std::memory_order_acquire) < world_size) {
    std::this_thread::yield();
  }
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

void PrintResult(const char *scenario, bool correct,
                 const std::vector<float> &result, float expected,
                 size_t recv_count, int world_size, double elapsed_ms) {
  constexpr const char *kGreen = "\033[32m";
  constexpr const char *kRed = "\033[31m";
  constexpr const char *kReset = "\033[0m";

  std::cout << "\n=== " << scenario
            << " CCL ReduceScatter Results ===" << std::endl;
  std::cout << "Correct: "
            << (correct ? (kGreen + std::string("YES") + kReset)
                        : (kRed + std::string("NO") + kReset))
            << std::endl;
  std::cout << "Expect:  " << expected << std::endl;
  std::cout << "Actual:  " << result.front() << std::endl;
  PrintReduceScatterMetrics(recv_count, world_size, elapsed_ms);
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for the ReduceScatter worker."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  hostname.back() = '\0';
  std::cout << "[Rank " << args.rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << args.rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  const size_t send_count = args.recv_count * static_cast<size_t>(args.size);
  const size_t send_bytes = send_count * sizeof(float);
  const size_t recv_bytes = args.recv_count * sizeof(float);
  const size_t local_offset = static_cast<size_t>(args.rank) * args.recv_count;

  std::vector<float> h_send(send_count);
  std::vector<float> h_recv(args.recv_count, 0.0f);
  FillInput(&h_send, args.recv_count, args.size, args.rank);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), send_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), recv_bytes));

  auto run_scenario = [&](bool in_place) {
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

    for (int i = 0; i < args.warmup_iter; ++i) {
      restore_local_block();
      CHECK_INFINI(infinicclReduceScatter(d_send, active_recv, args.recv_count,
                                          infinicclFloat32, infinicclSum, comm,
                                          nullptr));
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    double elapsed_ms = 0.0;
    if (in_place) {
      for (int i = 0; i < args.profile_iter; ++i) {
        restore_local_block();
        Timer timer;
        CHECK_INFINI(infinicclReduceScatter(d_send, active_recv,
                                            args.recv_count, infinicclFloat32,
                                            infinicclSum, comm, nullptr));
        CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
        elapsed_ms += timer.ElapsedMs();
      }
      elapsed_ms /= static_cast<double>(args.profile_iter);
    } else {
      Timer timer;
      for (int i = 0; i < args.profile_iter; ++i) {
        CHECK_INFINI(infinicclReduceScatter(d_send, active_recv,
                                            args.recv_count, infinicclFloat32,
                                            infinicclSum, comm, nullptr));
      }
      CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
      elapsed_ms = timer.ElapsedMs() / static_cast<double>(args.profile_iter);
    }

    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), active_recv, recv_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    const float expected = ExpectedValue(args.rank, args.size);
    const bool correct = Validator::ValidateResult(
        h_recv.data(), args.recv_count, expected, args.rank);
    ScenarioState *state = in_place ? args.in_place : args.out_of_place;
    if (!correct) {
      state->correct.store(false, std::memory_order_relaxed);
    }
    WaitForScenario(state, args.size);

    if (args.rank == 0) {
      PrintResult(in_place ? "In-Place" : "Out-of-Place",
                  state->correct.load(std::memory_order_acquire), h_recv,
                  expected, args.recv_count, args.size, elapsed_ms);
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
            << "  -n <recv_count>      Elements received per rank (default: "
               "1048576)\n";
}

}  // namespace

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t recv_count = 1 << 20;

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
        parsed = ParsePositiveNumber(optarg, &recv_count);
        break;
      case 'h':
        PrintUsage(argv[0]);
        return EXIT_SUCCESS;
      default:
        PrintUsage(argv[0]);
        return EXIT_FAILURE;
    }

    if (!parsed) {
      std::cerr << "Invalid positive numeric option for ReduceScatter."
                << std::endl;
      return EXIT_FAILURE;
    }
  }

  if (optind != argc) {
    std::cerr << "Unexpected positional argument for ReduceScatter."
              << std::endl;
    return EXIT_FAILURE;
  }
  if (recv_count > std::numeric_limits<size_t>::max() / sizeof(float) ||
      static_cast<size_t>(num_gpus) >
          std::numeric_limits<size_t>::max() / recv_count ||
      recv_count * static_cast<size_t>(num_gpus) >
          std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "ReduceScatter buffer size overflows `size_t`." << std::endl;
    return EXIT_FAILURE;
  }

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    std::cerr << "Failed to query the hostname for ReduceScatter." << std::endl;
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
    ThreadArgs args{rank,         num_gpus,      shared_id,     recv_count,
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
    std::cout << "[Main Process] All ReduceScatter scenarios passed."
              << std::endl;
  } else {
    std::cerr << "[Main Process] ReduceScatter validation failed." << std::endl;
  }
  std::cout
      << "[Main Process] All worker threads joined. InfiniCCL finalized safely."
      << std::endl;
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
