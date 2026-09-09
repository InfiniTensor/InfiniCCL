/**
 * InfiniCCL Example: Thread-per-GPU Single-Node Send/Recv
 *
 * This example initializes one native CCL communicator rank per GPU thread
 * and transfers data from rank 0 to rank 1 with `infinicclSend()` and
 * `infinicclRecv()`. Other ranks participate in communicator initialization
 * and teardown only.
 *
 * Run this example with `--launcher none`; it does not use MPI.
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

#include "backend_manifest.h"
#include "infiniccl.h"
#include "utils.h"

namespace ccl = infini::ccl;

namespace {

constexpr int kSenderRank = 0;
constexpr int kReceiverRank = 1;
constexpr float kPayloadValue = 7.0f;

struct RunState {
  std::atomic<int> initialized{0};
  std::atomic<int> completed{0};
  bool correct = false;
  float actual = 0.0f;
  double elapsed_ms = 0.0;
};

struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId unique_id;
  size_t num_elements;
  int warmup_iters;
  int profile_iters;
  RunState *state;
};

template <typename T>
bool ParseInteger(const char *text, T *value) {
  if (!text || !value) {
    return false;
  }

  const char *end = text + std::strlen(text);
  auto result = std::from_chars(text, end, *value);
  return result.ec == std::errc{} && result.ptr == end;
}

void WaitForAll(std::atomic<int> *counter, int size) {
  counter->fetch_add(1, std::memory_order_acq_rel);
  while (counter->load(std::memory_order_acquire) != size) {
    std::this_thread::yield();
  }
}

void PrintSendRecvMetrics(size_t num_elements, double elapsed_ms) {
  constexpr double kBytesPerMiB = 1024.0 * 1024.0;
  constexpr double kBytesPerGB = 1.0e9;
  const double payload_bytes =
      static_cast<double>(num_elements) * sizeof(float);
  const auto original_flags = std::cout.flags();
  const auto original_precision = std::cout.precision();

  std::cout << "Data size:      " << num_elements << " floats (" << std::fixed
            << std::setprecision(2) << payload_bytes / kBytesPerMiB << " MiB)"
            << std::endl;
  std::cout << "Time:           " << std::setprecision(3) << elapsed_ms << " ms"
            << std::endl;
  if (elapsed_ms > 0.0 && std::isfinite(elapsed_ms)) {
    const double algorithm_bandwidth =
        payload_bytes / kBytesPerGB / (elapsed_ms / 1000.0);
    // A one-way point-to-point transfer moves one payload over the bus.
    const double bus_bandwidth = algorithm_bandwidth;
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

void RunWorker(ThreadArgs args) {
  constexpr ccl::Device::Type kDeviceType =
      ccl::ListGetBest<ccl::DevicePriority>(ccl::EnabledDevices{});
  using Rt = ccl::Runtime<kDeviceType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(
      infinicclCommInitRank(&comm, args.size, args.unique_id, args.rank));

  const bool is_sender = args.rank == kSenderRank;
  const bool is_receiver = args.rank == kReceiverRank;
  const bool transfers_data = is_sender || is_receiver;
  const size_t total_bytes = args.num_elements * sizeof(float);

  std::vector<float> host_buffer(transfers_data ? args.num_elements : 0,
                                 is_sender ? kPayloadValue : 0.0f);
  float *device_buffer = nullptr;

  if (transfers_data) {
    CHECK_RT(
        Rt, Rt::Malloc(reinterpret_cast<void **>(&device_buffer), total_bytes));
    CHECK_RT(Rt, Rt::Memcpy(device_buffer, host_buffer.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  }

  WaitForAll(&args.state->initialized, args.size);

  auto transfer = [&]() {
    if (is_sender) {
      return infinicclSend(device_buffer, args.num_elements, infinicclFloat32,
                           kReceiverRank, comm, nullptr);
    }
    if (is_receiver) {
      return infinicclRecv(device_buffer, args.num_elements, infinicclFloat32,
                           kSenderRank, comm, nullptr);
    }
    return infinicclSuccess;
  };

  for (int i = 0; i < args.warmup_iters; ++i) {
    CHECK_INFINI(transfer());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < args.profile_iters; ++i) {
    CHECK_INFINI(transfer());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  if (is_sender) {
    args.state->elapsed_ms =
        timer.ElapsedMs() / static_cast<double>(args.profile_iters);
  }

  if (is_receiver) {
    CHECK_RT(Rt, Rt::Memcpy(host_buffer.data(), device_buffer, total_bytes,
                            Rt::MemcpyDeviceToHost));
    args.state->correct = Validator::ValidateResult(
        host_buffer.data(), args.num_elements, kPayloadValue, args.rank, false,
        "CCL Send/Recv");
    args.state->actual = host_buffer.front();
  }

  WaitForAll(&args.state->completed, args.size);

  if (transfers_data) {
    CHECK_RT(Rt, Rt::Free(device_buffer));
  }
  CHECK_INFINI(infinicclCommDestroy(comm));
}

void PrintUsage(const char *program) {
  std::cout << "Usage: " << program << " [options]\n"
            << "Options:\n"
            << "  -g <num_gpus>        Number of GPUs (default: 8)\n"
            << "  -w <warmup_iters>    Warm-up iterations (default: 2)\n"
            << "  -p <profile_iters>   Profile iterations (default: 20)\n"
            << "  -n <num_elements>    Number of elements (default: 1048576)\n";
}

}  // namespace

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

  int option = 0;
  while ((option = getopt(argc, argv, "g:w:p:n:h")) != -1) {
    bool valid = true;
    switch (option) {
      case 'g':
        valid = ParseInteger(optarg, &num_gpus);
        break;
      case 'w':
        valid = ParseInteger(optarg, &warmup_iters);
        break;
      case 'p':
        valid = ParseInteger(optarg, &profile_iters);
        break;
      case 'n':
        valid = ParseInteger(optarg, &num_elements);
        break;
      case 'h':
        PrintUsage(argv[0]);
        return EXIT_SUCCESS;
      default:
        valid = false;
        break;
    }

    if (!valid) {
      std::cerr << "Invalid numeric argument." << std::endl;
      PrintUsage(argv[0]);
      return EXIT_FAILURE;
    }
  }

  if (optind != argc) {
    std::cerr << "Unexpected positional argument." << std::endl;
    PrintUsage(argv[0]);
    return EXIT_FAILURE;
  }

  if (num_gpus < 2 || warmup_iters < 0 || profile_iters <= 0 ||
      num_elements == 0 ||
      num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Send/Recv requires at least two GPUs, a non-negative warm-up "
                 "count, a positive profile count, and a positive element "
                 "count."
              << std::endl;
    return EXIT_FAILURE;
  }

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size() - 1) != 0) {
    std::cerr << "Failed to query the local hostname." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "[Main Process] Host: " << hostname.data()
            << " | Target GPUs: " << num_gpus << " | Sender: " << kSenderRank
            << " | Receiver: " << kReceiverRank << std::endl;

  infinicclUniqueId unique_id{};
  CHECK_INFINI(infinicclGetUniqueId(&unique_id));

  RunState state;
  std::vector<std::thread> workers;
  workers.reserve(num_gpus);

  for (int rank = 0; rank < num_gpus; ++rank) {
    workers.emplace_back(RunWorker,
                         ThreadArgs{rank, num_gpus, unique_id, num_elements,
                                    warmup_iters, profile_iters, &state});
  }

  for (auto &worker : workers) {
    worker.join();
  }

  const char *color = state.correct ? "\033[32m" : "\033[31m";
  std::cout << "\n=== CCL Send/Recv Results ===" << std::endl;
  std::cout << "Correct: " << color << (state.correct ? "YES" : "NO")
            << "\033[0m" << std::endl;
  std::cout << "Expect:  " << kPayloadValue << std::endl;
  std::cout << "Actual:  " << state.actual << std::endl;
  PrintSendRecvMetrics(num_elements, state.elapsed_ms);

  std::cout << "[Main Process] All worker threads joined." << std::endl;
  return state.correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
