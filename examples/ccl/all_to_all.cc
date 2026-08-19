/**
 * InfiniCCL Example: Thread-per-GPU Single-Node AllToAll
 *
 * This example creates one native CCL rank per GPU and performs an
 * out-of-place all-to-all exchange without any OpenMPI dependencies.
 */

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

#include "backend_manifest.h"
#include "infiniccl.h"
#include "utils.h"

using namespace infini::ccl;

struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t count_per_peer;
  int warmup_iter;
  int profile_iter;
  std::atomic<bool> *correct;
};

uint32_t ElementValue(int source, int destination, size_t index,
                      size_t count_per_peer, int size) {
  const size_t block = static_cast<size_t>(source) * size + destination;
  return static_cast<uint32_t>(block * count_per_peer + index);
}

bool ValidateResult(const std::vector<uint32_t> &result, size_t count_per_peer,
                    int size, int rank) {
  for (int source = 0; source < size; ++source) {
    const size_t offset = static_cast<size_t>(source) * count_per_peer;
    for (size_t index = 0; index < count_per_peer; ++index) {
      const uint32_t expected =
          ElementValue(source, rank, index, count_per_peer, size);
      if (result[offset + index] != expected) {
        std::cerr << "[Rank " << rank << "] AllToAll mismatch at source "
                  << source << ", index " << index << ": expected " << expected
                  << ", got " << result[offset + index] << std::endl;
        return false;
      }
    }
  }
  return true;
}

void PrintMetrics(double elapsed_ms, size_t total_bytes, int size) {
  const double seconds = elapsed_ms / 1000.0;
  const double gigabytes =
      static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0);
  const double algorithm_bandwidth = gigabytes / seconds;
  const double bus_bandwidth =
      algorithm_bandwidth * static_cast<double>(size - 1) / size;

  std::cout << "Time:           " << std::fixed << std::setprecision(3)
            << elapsed_ms << " ms" << std::endl;
  std::cout << "Throughput:     " << std::fixed << std::setprecision(2)
            << bus_bandwidth << " GB/s (Bus BW)" << std::endl;
  std::cout << "Alg Bandwidth:  " << std::fixed << std::setprecision(2)
            << algorithm_bandwidth << " GB/s" << std::endl;
}

void WorkerThread(ThreadArgs args) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(args.rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, args.size, args.id, args.rank));

  const size_t total_elements =
      args.count_per_peer * static_cast<size_t>(args.size);
  const size_t total_bytes = total_elements * sizeof(uint32_t);
  std::vector<uint32_t> h_send(total_elements);
  std::vector<uint32_t> h_recv(total_elements,
                               std::numeric_limits<uint32_t>::max());

  for (int destination = 0; destination < args.size; ++destination) {
    const size_t offset =
        static_cast<size_t>(destination) * args.count_per_peer;
    for (size_t index = 0; index < args.count_per_peer; ++index) {
      h_send[offset + index] = ElementValue(args.rank, destination, index,
                                            args.count_per_peer, args.size);
    }
  }

  uint32_t *d_send = nullptr;
  uint32_t *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  for (int i = 0; i < args.warmup_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, args.count_per_peer,
                                   infinicclUInt32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < args.profile_iter; ++i) {
    CHECK_INFINI(infinicclAllToAll(d_send, d_recv, args.count_per_peer,
                                   infinicclUInt32, comm, nullptr));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  const double elapsed_ms =
      timer.ElapsedMs() / static_cast<double>(args.profile_iter);

  // Use an untimed exchange for validation so every unwritten element retains
  // the sentinel value without affecting the performance measurement.
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclAllToAll(d_send, d_recv, args.count_per_peer,
                                 infinicclUInt32, comm, nullptr));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                          Rt::MemcpyDeviceToHost));

  if (!ValidateResult(h_recv, args.count_per_peer, args.size, args.rank)) {
    args.correct->store(false, std::memory_order_relaxed);
  }

  if (args.rank == 0) {
    std::cout << "\n=== Single-Node Threaded AllToAll Results ===" << std::endl;
    std::cout << "Data size per rank: " << total_elements << " uint32 values ("
              << total_bytes / 1024 / 1024 << " MB)" << std::endl;
    PrintMetrics(elapsed_ms, total_bytes, args.size);
  }

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
          std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
      count_per_peer > std::numeric_limits<uint32_t>::max() / size / size) {
    std::cerr << "AllToAll buffer or validation value exceeds its range."
              << std::endl;
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
    ThreadArgs args{rank,         num_gpus,      shared_id, count_per_peer,
                    warmup_iters, profile_iters, &correct};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  const bool result = correct.load(std::memory_order_relaxed);
  std::cout << "[Main Process] AllToAll validation "
            << (result ? "passed." : "failed.") << std::endl;
  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
