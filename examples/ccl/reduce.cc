/**
 * InfiniCCL Example: Thread-per-GPU Single-Node `Reduce`
 *
 * This example creates one native CCL rank per GPU and performs a rooted
 * reduction without any OpenMPI dependencies.
 *
 * Use `--launcher none` for this thread-per-GPU example.
 */

#include <unistd.h>

#include <atomic>
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
  std::atomic<bool> *correct;
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

  // Prepare Data (only root receives the reduced result).
  const bool is_root = args.rank == kRoot;
  const size_t total_bytes = args.num_elements * sizeof(float);
  std::vector<float> h_send(args.num_elements,
                            static_cast<float>(args.rank + 1));
  std::vector<float> h_recv(is_root ? args.num_elements : 0, 0.0f);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  }
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up Iterations
  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, args.num_elements,
                                 infinicclFloat32, infinicclSum, kRoot, comm,
                                 nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling Iterations
  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(infinicclReduce(d_send, d_recv, args.num_elements,
                                 infinicclFloat32, infinicclSum, kRoot, comm,
                                 nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  if (is_root) {
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    // Result Validation
    float expected = 0.0f;
    for (int r = 0; r < args.size; ++r) {
      expected += static_cast<float>(r + 1);
    }
    const bool correct = Validator::ValidateResult(
        h_recv.data(), args.num_elements, expected, args.rank, true, "Reduce");
    args.correct->store(correct, std::memory_order_relaxed);

    // Metrics Reporting (Only Root)
    std::cout << "\n=== Single-Node Threaded Reduce Results ===" << std::endl;
    std::cout << "Data size: " << args.num_elements << " floats ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    Metrics metrics{elapsed_ms, total_bytes, args.size};
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
      num_elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
    std::cerr << "`Reduce` arguments or buffer size are invalid." << std::endl;
    return EXIT_FAILURE;
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::cout << "[Main Process] Host: " << hostname
            << " | Target GPUs: " << num_gpus << std::endl;

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::atomic<bool> correct{true};
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

  const bool result = correct.load(std::memory_order_relaxed);
  std::cout << "[Main Process] Reduce validation "
            << (result ? "passed." : "failed.") << std::endl;
  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
