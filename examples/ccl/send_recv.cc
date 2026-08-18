/**
 * InfiniCCL Example: Thread-per-GPU Single-Node Send/Recv
 *
 * This example transfers data from GPU 0 to GPU 1 through InfiniCCL's native
 * CCL backend without an MPI launcher.
 */

#include <unistd.h>

#include <atomic>
#include <charconv>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <thread>
#include <vector>

#include "backend_manifest.h"
#include "infiniccl.h"
#include "utils.h"

using namespace infini::ccl;

namespace {

constexpr int kRankCount = 2;
constexpr int kSender = 0;
constexpr int kReceiver = 1;
constexpr float kSendValue = 7.0f;

struct ThreadArgs {
  int rank;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
  std::atomic_bool* all_correct;
};

template <typename T>
bool ParseNumber(const char* text, T* value) {
  const std::string_view input{text};
  T parsed{};
  const auto [end, error] =
      std::from_chars(input.data(), input.data() + input.size(), parsed);

  if (error != std::errc{} || end != input.data() + input.size()) {
    return false;
  }

  *value = parsed;
  return true;
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, kRankCount, args.id, args.rank));

  std::vector<float> host_buffer(args.num_elements,
                                 args.rank == kSender ? kSendValue : 0.0f);
  float* device_buffer = nullptr;
  const size_t total_bytes = args.num_elements * sizeof(float);

  CHECK_RT(Rt,
           Rt::Malloc(reinterpret_cast<void**>(&device_buffer), total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(device_buffer, host_buffer.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  auto exchange = [&]() {
    if (args.rank == kSender) {
      return infinicclSend(device_buffer, args.num_elements, infinicclFloat32,
                           kReceiver, comm, nullptr);
    }
    return infinicclRecv(device_buffer, args.num_elements, infinicclFloat32,
                         kSender, comm, nullptr);
  };

  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(exchange());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(exchange());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  if (args.rank == kReceiver) {
    CHECK_RT(Rt, Rt::Memcpy(host_buffer.data(), device_buffer, total_bytes,
                            Rt::MemcpyDeviceToHost));
    const bool correct =
        Validator::ValidateResult(host_buffer.data(), args.num_elements,
                                  kSendValue, kSender, true, "Send/Recv");
    if (!correct) {
      args.all_correct->store(false, std::memory_order_relaxed);
    }
  } else {
    std::cout << "\n=== Single-Node Threaded Send/Recv Results ==="
              << std::endl;
    Metrics metrics{elapsed, total_bytes, kRankCount};
    metrics.Print();
  }

  CHECK_RT(Rt, Rt::Free(device_buffer));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

}  // namespace

int main(int argc, char** argv) {
  size_t num_elements = 1 << 20;
  int warmup_iterations = 2;
  int profile_iterations = 20;

  int opt;
  while ((opt = getopt(argc, argv, "n:w:p:h")) != -1) {
    switch (opt) {
      case 'n':
        if (!ParseNumber(optarg, &num_elements)) {
          std::cerr << "Invalid value for `-n`." << std::endl;
          return EXIT_FAILURE;
        }
        break;
      case 'w':
        if (!ParseNumber(optarg, &warmup_iterations)) {
          std::cerr << "Invalid value for `-w`." << std::endl;
          return EXIT_FAILURE;
        }
        break;
      case 'p':
        if (!ParseNumber(optarg, &profile_iterations)) {
          std::cerr << "Invalid value for `-p`." << std::endl;
          return EXIT_FAILURE;
        }
        break;
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "  -n <elements>     Elements to transfer (default: "
                  << (1 << 20) << ")\n"
                  << "  -w <iterations>   Warm-up iterations (default: 2)\n"
                  << "  -p <iterations>   Profile iterations (default: 20)\n";
        return EXIT_SUCCESS;
      default:
        return EXIT_FAILURE;
    }
  }

  if (num_elements == 0 || warmup_iterations < 0 ||
      profile_iterations <= 0) {
    std::cerr << "Elements and profile iterations must be positive; warm-up "
                 "iterations must be non-negative."
              << std::endl;
    return EXIT_FAILURE;
  }

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::atomic_bool all_correct{true};
  std::vector<std::thread> threads;
  threads.reserve(kRankCount);

  for (int rank = 0; rank < kRankCount; ++rank) {
    ThreadArgs args{rank,
                    shared_id,
                    num_elements,
                    warmup_iterations,
                    profile_iterations,
                    &all_correct};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  if (!all_correct.load(std::memory_order_relaxed)) {
    std::cerr << "Send/Recv validation failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "Send/Recv validation passed." << std::endl;
  return EXIT_SUCCESS;
}
