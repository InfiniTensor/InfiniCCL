/**
 * InfiniCCL Example: Broadcast (OpenMPI + CCL Hybrid)
 *
 * This example uses OpenMPI to distribute a CCL unique ID across processes,
 * initializes one native CCL rank per GPU, and then performs CCL broadcasts.
 */

#include <unistd.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "backend_manifest.h"
#include "device.h"
#include "infiniccl.h"
#include "runtime.h"
#include "traits.h"
#include "utils.h"

using namespace infini::ccl;

bool RunBroadcastExample(int argc, char **argv, int warmup_iter,
                         int profile_iter, size_t num_elements) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = 0;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size()) != 0) {
    constexpr char kUnknownHostname[] = "unknown";
    std::copy_n(kUnknownHostname, sizeof(kUnknownHostname), hostname.begin());
    std::cerr << "[Rank " << rank
              << "] Failed to query the host name; using `unknown`."
              << std::endl;
  }
  hostname.back() = '\0';

  const char *local_rank_text = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_text) {
    local_rank = std::atoi(local_rank_text);
  }
  CHECK_RT(Rt, Rt::SetDevice(local_rank));

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  infinicclUniqueId id{};
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&id));
  }
  CHECK_INFINI(infinicclBroadcast(&id, &id, sizeof(id), infinicclChar, 0, comm,
                                  nullptr));

  CHECK_INFINI(infinicclCommInitRank(&comm, size, id, rank));

  const int root = size > 1 ? size - 1 : 0;
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << Device::StringFromType(kDevType) << " | Device "
            << local_rank << " | Broadcast Root: " << root << std::endl;

  constexpr float kRootValue = 42.0f;
  constexpr float kSentinelValue = -1.0f;
  const size_t total_bytes = num_elements * sizeof(float);

  std::vector<float> h_send(num_elements, kSentinelValue);
  std::vector<float> h_recv(num_elements, kSentinelValue);

  float *d_send = nullptr;
  float *d_recv = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_send), total_bytes));
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&d_recv), total_bytes));

  auto ResetBuffers = [&]() {
    std::fill(h_send.begin(), h_send.end(),
              rank == root ? kRootValue : kSentinelValue);
    std::fill(h_recv.begin(), h_recv.end(), kSentinelValue);
    CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::Memcpy(d_recv, h_recv.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  };

  auto RunScenario = [&](const std::string &name, float *verify_buff,
                         auto &&collective_call) {
    for (int i = 0; i < warmup_iter; ++i) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

    Timer timer;
    for (int i = 0; i < profile_iter; ++i) {
      CHECK_INFINI(collective_call());
    }
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    const double elapsed =
        timer.ElapsedMs() / static_cast<double>(profile_iter);

    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), verify_buff, total_bytes,
                            Rt::MemcpyDeviceToHost));

    const bool correct = Validator::ValidateResult(
        h_recv.data(), num_elements, kRootValue, rank, true, name);
    if (rank == 0) {
      Metrics metrics{elapsed, total_bytes, size};
      metrics.Print();
    }

    return correct;
  };

  bool all_correct = true;

  ResetBuffers();
  all_correct &=
      RunScenario("Out-of-Place Hybrid CCL Broadcast", d_recv, [&]() {
        return infinicclBroadcast(d_send, d_recv, num_elements,
                                  infinicclFloat32, root, comm, nullptr);
      });

  ResetBuffers();
  float *d_in_place = rank == root ? d_send : d_recv;
  all_correct &=
      RunScenario("In-Place Hybrid CCL Broadcast", d_in_place, [&]() {
        return infinicclBroadcast(d_in_place, d_in_place, num_elements,
                                  infinicclFloat32, root, comm, nullptr);
      });

  ResetBuffers();
  d_in_place = rank == root ? d_send : d_recv;
  all_correct &=
      RunScenario("Legacy In-Place Hybrid CCL Bcast", d_in_place, [&]() {
        return infinicclBcast(d_in_place, num_elements, infinicclFloat32, root,
                              comm, nullptr);
      });

  CHECK_RT(Rt, Rt::Free(d_send));
  CHECK_RT(Rt, Rt::Free(d_recv));
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  if (!all_correct) {
    std::cerr << "[Rank " << rank << "] Hybrid CCL broadcast validation failed."
              << std::endl;
  }

  return all_correct;
}

int main(int argc, char **argv) {
  constexpr int kWarmupIterations = 2;
  constexpr int kProfileIterations = 20;
  constexpr size_t kNumElements = 1 << 20;

  return RunBroadcastExample(argc, argv, kWarmupIterations, kProfileIterations,
                             kNumElements)
             ? EXIT_SUCCESS
             : EXIT_FAILURE;
}
