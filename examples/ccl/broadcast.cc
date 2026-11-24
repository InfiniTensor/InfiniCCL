/**
 * InfiniCCL Example: Thread-per-GPU Single-Node `Broadcast`
 *
 * This example creates one native CCL rank per GPU and performs a rooted
 * broadcast without any OpenMPI dependencies, supporting out-of-place,
 * in-place, and legacy in-place topologies.
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
constexpr float kRootMagicValue = 42.0f;

// Structure to pass execution data to each GPU worker thread.
struct ThreadArgs {
  int rank;
  int size;
  infinicclUniqueId id;
  size_t num_elements;
  int warmup_iter;
  int profile_iter;
  int *correct;
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

  const bool is_root = args.rank == kRoot;
  const size_t total_bytes = args.num_elements * sizeof(float);

  // Prepare Host Data: only root fills its buffer with the broadcast payload.
  std::vector<float> h_send(is_root ? args.num_elements : 0, 0.0f);
  std::vector<float> h_recv(args.num_elements, 0.0f);
  if (is_root) {
    for (size_t i = 0; i < args.num_elements; ++i) {
      h_send[i] = kRootMagicValue;
    }
  }

  // Allocate Device Memory.
  float *d_send = nullptr;
  float *d_recv = nullptr;
  if (is_root) {
    CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
    CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
  }
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));
  CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  // The Abstract Execution & Validation Function
  auto ProfileAndValidateScenario = [&](const std::string &scenario_name,
                                        float *d_verify_source,
                                        bool *scenario_correct,
                                        auto &&collective_call) {
    if (is_root) {
      std::cout << "\n=== Performing " << scenario_name << " ===" << std::endl;
      if (scenario_name.find("Scenario 1") != std::string::npos) {
        std::cout << "Data size: " << args.num_elements << " floats ("
                  << total_bytes / 1024 / 1024 << " MB)" << std::endl;
        std::cout << "Root Node: " << kRoot << std::endl;
      }
    }

    // Warm-up Iteration & Initial Capture
    CHECK_INFINI(collective_call());
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_verify_source, total_bytes,
                            Rt::MemcpyDeviceToHost));

    for (int i = 1; i < args.warmup_iter; ++i) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    // Profiling
    Timer timer;
    for (int i = 0; i < args.profile_iter; i++) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    double elapsed = timer.ElapsedMs() / static_cast<double>(args.profile_iter);

    // Validation: every rank checks the broadcast payload it received.
    *scenario_correct = Validator::ValidateResult(
        h_recv.data(), args.num_elements, kRootMagicValue, args.rank, true,
        scenario_name);

    // Performance Reporting (Only Root)
    if (is_root) {
      Metrics metrics{elapsed, total_bytes, args.size};
      metrics.Print();
    }
  };

  bool scenario_correct = true;
  bool all_correct = true;

  // --------------------------------------------------------------------------
  // Scenario 1: Out-of-Place Broadcast
  // --------------------------------------------------------------------------
  ProfileAndValidateScenario(
      "Scenario 1: Out-of-Place Broadcast", d_recv, &scenario_correct, [&]() {
        return infinicclBroadcast(d_send, d_recv, args.num_elements,
                                  infinicclFloat32, kRoot, comm, nullptr);
      });
  all_correct = all_correct && scenario_correct;

  // --------------------------------------------------------------------------
  // Scenario 2: In-Place Broadcast
  // --------------------------------------------------------------------------
  std::fill(h_recv.begin(), h_recv.end(), 0.0f);
  float *d_inplace_buf = is_root ? d_send : d_recv;

  ProfileAndValidateScenario("Scenario 2: In-Place Broadcast", d_inplace_buf,
                             &scenario_correct, [&]() {
                               return infinicclBroadcast(
                                   d_inplace_buf, d_inplace_buf,
                                   args.num_elements, infinicclFloat32, kRoot,
                                   comm, nullptr);
                             });
  all_correct = all_correct && scenario_correct;

  // --------------------------------------------------------------------------
  // Scenario 3: Legacy In-Place Bcast
  // --------------------------------------------------------------------------
  std::fill(h_recv.begin(), h_recv.end(), 0.0f);

  ProfileAndValidateScenario("Scenario 3: Legacy In-Place Bcast", d_inplace_buf,
                             &scenario_correct, [&]() {
                               return infinicclBcast(
                                   d_inplace_buf, args.num_elements,
                                   infinicclFloat32, kRoot, comm, nullptr);
                             });
  all_correct = all_correct && scenario_correct;

  *args.correct = all_correct;

  // Cleanup local rank resources.
  if (is_root) {
    CHECK_RT(Rt, Rt::Free(d_send));
  }
  CHECK_RT(Rt, Rt::Free(d_recv));
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
    std::cerr << "`Broadcast` arguments or buffer size are invalid."
              << std::endl;
    return EXIT_FAILURE;
  }

  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  std::cout << "[Main Process] Host: " << hostname
            << " | Target GPUs: " << num_gpus << std::endl;

  infinicclUniqueId shared_id;
  CHECK_INFINI(infinicclGetUniqueId(&shared_id));

  std::vector<int> correct(num_gpus, 0);
  std::vector<std::thread> threads;
  threads.reserve(num_gpus);
  for (int rank = 0; rank < num_gpus; ++rank) {
    ThreadArgs args{rank,         num_gpus,      shared_id,     num_elements,
                    warmup_iters, profile_iters, &correct[rank]};
    threads.emplace_back(WorkerThread, args);
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  const bool result = std::all_of(correct.begin(), correct.end(),
                                  [](int value) { return value != 0; });
  std::cout << "[Main Process] Broadcast validation "
            << (result ? "passed." : "failed.") << std::endl;
  return result ? EXIT_SUCCESS : EXIT_FAILURE;
}
