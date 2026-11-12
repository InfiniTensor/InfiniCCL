/**
 * InfiniCCL Example: Thread-per-GPU Single-Node `Gather`
 *
 * This example creates one native CCL rank per GPU and performs a rooted
 * gather without any OpenMPI dependencies.
 *
 * Use `--launcher none` for this thread-per-GPU example.
 */

#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
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

constexpr int kRoot = 0;

// Structure to pass execution data to each GPU worker thread.
struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
  bool *correct;
};

// Worker function executed by each CPU thread.
void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  // Bind this CPU thread to its local GPU.
  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  // Prepare Data (only root receives the gathered result).
  const bool is_root = args.rank == kRoot;
  const size_t send_bytes = args.num_elements * sizeof(float);
  const size_t recv_bytes = send_bytes * static_cast<size_t>(args.size);
  std::vector<float> h_send(args.num_elements,
                            static_cast<float>(args.rank + 1));
  std::vector<float> h_recv(
      is_root ? args.num_elements * static_cast<size_t>(args.size) : 0, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), send_bytes));
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), recv_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), send_bytes,
                          Rt::MemcpyHostToDevice));
  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), recv_bytes,
                            Rt::MemcpyHostToDevice));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up Iterations
  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, args.num_elements,
                                 infinicclFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling Iterations
  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(infinicclGather(d_send, d_recv, args.num_elements,
                                 infinicclFloat32, kRoot, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, recv_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    // Result Validation
    bool correct = true;
    for (int source = 0; source < args.size; ++source) {
      const size_t offset = static_cast<size_t>(source) * args.num_elements;
      correct = Validator::ValidateResult(
                    h_recv.data() + offset, args.num_elements,
                    static_cast<float>(source + 1), args.rank) &&
                correct;
    }
    *args.correct = correct;

    // Metrics Reporting (Only Root)
    std::cout << "\n=== Single-Node Threaded Gather Results ===" << std::endl;
    std::cout << "Correct: " << (correct ? "YES" : "NO") << std::endl;
    std::cout << "Sample blocks: ";
    for (int source = 0; source < std::min(args.size, 4); ++source) {
      const size_t offset = static_cast<size_t>(source) * args.num_elements;
      std::cout << "[r" << source << ": " << h_recv[offset] << "] ";
    }
    std::cout << std::endl;

    Metrics metrics{elapsed_ms, recv_bytes, args.size};
    metrics.Print();
  }

  // Cleanup local rank resources.
  CHECK_RT(Rt, Rt::Free(d_send));
  if (is_root) {
    CHECK_RT(Rt, Rt::Free(d_recv));
  }
  CHECK_INFINI(infinicclCommDestroy(comm));
}

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t num_elements = 1 << 20;

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
                  << "  -w <warmup_iters>    Warmup iterations (default: 2)\n"
                  << "  -p <profile_iters>   Profile iterations (default: 20)\n"
                  << "  -n <num_elements>    Number of elements (default: "
                  << (1 << 20) << ")\n";
        return EXIT_SUCCESS;
      default:
        std::cerr << "Invalid argument. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }
  }

  if (num_gpus <= 0 || warmup_iters <= 0 || profile_iters <= 0 ||
      num_elements == 0 ||
      num_elements > std::numeric_limits<size_t>::max() / sizeof(float) /
                         static_cast<size_t>(num_gpus)) {
    std::cerr << "`Gather` arguments or buffer size are invalid." << std::endl;
    return EXIT_FAILURE;
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::cout << "[Main Process] Host: " << hostname
            << " | Target GPUs: " << num_gpus << std::endl;

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  bool correct = true;
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);
  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,      shared_id, num_elements,
                    warmup_iters, profile_iters, &correct};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  const bool result = correct;
  std::cout << "[Main Process] Gather validation "
            << (result ? "passed." : "failed.") << std::endl;
  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
