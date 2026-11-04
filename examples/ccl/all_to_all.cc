/**
 * InfiniCCL Example: Thread-per-GPU Single-Node AllToAll
 *
 * This example creates one native CCL rank per GPU and performs an
 * out-of-place all-to-all exchange without any OpenMPI dependencies.
 *
 * Use `--launcher none` since this example spawns its own GPU worker threads.
 */

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
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

// Structure to pass execution data to each GPU worker thread.
struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t count_per_peer;
  int warmup_iter;
  int profile_iter;
  std::atomic_bool *all_correct;
};

// Worker function executed by each CPU thread.
void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  // Bind this CPU thread to its designated local GPU device.
  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  // Prepare Host Data Structures
  const size_t total_elements =
      args.count_per_peer * static_cast<size_t>(args.size);
  const size_t total_bytes = total_elements * sizeof(uint32_t);
  std::vector<uint32_t> h_send(total_elements);
  std::vector<uint32_t> h_recv(total_elements, 0);

  // Layout: one block per destination, encoding the source and destination.
  for (int dst = 0; dst < args.size; ++dst) {
    uint32_t value = static_cast<uint32_t>(args.rank) * 1000 + dst;
    size_t off = static_cast<size_t>(dst) * args.count_per_peer;
    for (size_t i = 0; i < args.count_per_peer; ++i) {
      h_send[off + i] = value;
    }
  }

  // Allocate GPU Memory using InfiniCCL's Runtime abstraction layer.
  uint32_t *d_send = nullptr;
  uint32_t *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Warm-up Iterations
  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, args.count_per_peer,
                                   infinicclUInt32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // Profiling Iterations
  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, args.count_per_peer,
                                   infinicclUInt32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  // Copy output data back from device to host.
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));

  // Result Validation
  // recv `block[src]` comes from src's send `block[dst=rank]`.
  bool correct = true;
  for (int src = 0; src < args.size; ++src) {
    uint32_t expected = static_cast<uint32_t>(src) * 1000 + args.rank;
    size_t off = static_cast<size_t>(src) * args.count_per_peer;
    bool block_ok = Validator::ValidateResult(
        h_recv.data() + off, args.count_per_peer, expected, args.rank);
    correct = correct && block_ok;
  }
  if (!correct) {
    args.all_correct->store(false, std::memory_order_relaxed);
  }

  // Metrics Reporting (Only Rank 0)
  if (args.rank == 0) {
    std::cout << "\n=== Single-Node Threaded AllToAll Results ===" << std::endl;
    std::cout << "Data size per rank: " << total_elements << " uint32 values ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    std::cout << "Correct: " << (correct ? "YES" : "NO") << std::endl;
    std::cout << "Sample recv blocks: ";
    for (int src = 0; src < std::min(args.size, 4); ++src) {
      size_t off = static_cast<size_t>(src) * args.count_per_peer;
      std::cout << "[src" << src << ": " << h_recv[off] << "] ";
    }
    std::cout << std::endl;
    Metrics metrics{elapsed_ms, total_bytes, args.size};
    metrics.Print();
  }

  // Cleanup local rank resources.
  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
}

int main(int argc, char **argv) {
  int num_gpus = 8;
  int warmup_iters = 2;
  int profile_iters = 20;
  size_t count_per_peer = 1 << 20;

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
        count_per_peer = static_cast<size_t>(std::stoull(optarg));
        break;
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "Options:\n"
                  << "  -g <num_gpus>        Number of GPUs (default: 8)\n"
                  << "  -w <warmup_iters>    Warmup iterations (default: 2)\n"
                  << "  -p <profile_iters>   Profile iterations (default: 20)\n"
                  << "  -n <count_per_peer>  Elements sent to each peer "
                     "(default: 1048576)\n";
        return EXIT_SUCCESS;
      default:
        std::cerr << "Invalid argument. Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }
  }

  if (num_gpus <= 0 || warmup_iters <= 0 || profile_iters <= 0 ||
      count_per_peer == 0) {
    std::cerr << "All arguments must be positive." << std::endl;
    return EXIT_FAILURE;
  }
  const size_t size = static_cast<size_t>(num_gpus);
  if (count_per_peer > std::numeric_limits<size_t>::max() / size ||
      count_per_peer * size >
          std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
    std::cerr << "AllToAll buffer size exceeds its range." << std::endl;
    return EXIT_FAILURE;
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::cout << "[Main Process] Host: " << hostname
            << " | Target GPUs: " << num_gpus << std::endl;

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::atomic_bool all_correct{true};
  // Spawn CPU thread pool.
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);
  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,      shared_id,   count_per_peer,
                    warmup_iters, profile_iters, &all_correct};
    threads.emplace_back(WorkerThread, args);
  }

  // Await execution completion across all threads.
  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  if (!all_correct.load(std::memory_order_relaxed)) {
    std::cerr << "AllToAll validation failed." << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "AllToAll validation passed." << std::endl;
  return EXIT_SUCCESS;
}
