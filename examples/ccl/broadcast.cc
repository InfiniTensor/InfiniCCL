/**
 * InfiniCCL Example: Thread-per-GPU Single-Node Broadcast
 *
 * This example spawns one CPU thread per GPU and performs native CCL
 * broadcasts without an MPI launcher.
 */

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

#include "backend_manifest.h"
#include "infiniccl.h"
#include "utils.h"

using namespace infini::ccl;

struct ThreadArgs {
  int rank;
  int size;
  int root;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
  std::atomic_bool *all_correct;
};

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  constexpr float kRootValue = 42.0f;
  constexpr float kSentinelValue = -1.0f;
  const size_t total_bytes = args.num_elements * sizeof(float);

  std::vector<float> h_send(args.num_elements, kSentinelValue);
  std::vector<float> h_recv(args.num_elements, kSentinelValue);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));

  auto ResetBuffers = [&]() {
    std::fill(h_send.begin(), h_send.end(),
              args.rank == args.root ? kRootValue : kSentinelValue);
    std::fill(h_recv.begin(), h_recv.end(), kSentinelValue);
    CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  };

  auto RunScenario = [&](const std::string &name, float *verify_buff,
                         auto &&collective_call) {
    for (int i = 0; i < args.warmup_iter; ++i) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    Timer timer;
    for (int i = 0; i < args.profile_iter; ++i) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    const double elapsed =
        timer.ElapsedMs() / static_cast<double>(args.profile_iter);

    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), verify_buff, total_bytes,
                            Rt::MemcpyDeviceToHost));

    const bool correct = Validator::ValidateResult(
        h_recv.data(), args.num_elements, kRootValue, args.rank, true, name);
    if (!correct) {
      args.all_correct->store(false, std::memory_order_relaxed);
    }

    if (args.rank == 0) {
      Metrics metrics{elapsed, total_bytes, args.size};
      metrics.Print();
    }
  };

  ResetBuffers();
  RunScenario("Out-of-Place Broadcast", d_recv, [&]() {
    return infinicclBroadcast(d_send, d_recv, args.num_elements,
                              infinicclFloat32, args.root, comm, nullptr);
  });

  ResetBuffers();
  float *d_in_place = args.rank == args.root ? d_send : d_recv;
  RunScenario("In-Place Broadcast", d_in_place, [&]() {
    return infinicclBroadcast(d_in_place, d_in_place, args.num_elements,
                              infinicclFloat32, args.root, comm, nullptr);
  });

  ResetBuffers();
  d_in_place = args.rank == args.root ? d_send : d_recv;
  RunScenario("Legacy In-Place Bcast", d_in_place, [&]() {
    return infinicclBcast(d_in_place, args.num_elements, infinicclFloat32,
                          args.root, comm, nullptr);
  });

  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iters = 1;
  int profile_iters = 20;
  size_t num_elements = 1 << 25;

  int opt;
  while ((opt = getopt(argc, argv, "g:w:p:n:h")) != -1) {
    switch (opt) {
      case 'g':
        num_gpus = std::stoi(optarg);
        break;
      case 'w':
        warmup_iters = std::stoi(optarg);
        break;
      case 'p':
        profile_iters = std::stoi(optarg);
        break;
      case 'n':
        num_elements = static_cast<size_t>(std::stoull(optarg));
        break;
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -g <num_gpus>        Number of GPUs (default: 8)\n"
                  << "  -w <warmup_iters>    Warmup iterations (default: 1)\n"
                  << "  -p <profile_iters>   Profile iterations (default: 20)\n"
                  << "  -n <num_elements>    Number of elements (default: "
                  << (1 << 25) << ")\n";
        return EXIT_SUCCESS;
      default:
        std::cerr << "Invalid argument. Use `-h` for help." << std::endl;
        return EXIT_FAILURE;
    }
  }

  if (num_gpus <= 0 || warmup_iters < 0 || profile_iters <= 0 ||
      num_elements == 0 ||
      num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "Invalid execution parameter." << std::endl;
    return EXIT_FAILURE;
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  const int root = num_gpus > 1 ? num_gpus - 1 : 0;
  std::cout << "[Main Process] Host: " << hostname
            << " | Target GPUs: " << num_gpus << " | Root: " << root
            << std::endl;

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::atomic_bool all_correct{true};
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);

  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,     root,          shared_id,
                    num_elements, warmup_iters, profile_iters, &all_correct};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  if (!all_correct.load(std::memory_order_relaxed)) {
    std::cerr << "Broadcast validation failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "[Main Process] All broadcast scenarios passed." << std::endl;
  return EXIT_SUCCESS;
}
