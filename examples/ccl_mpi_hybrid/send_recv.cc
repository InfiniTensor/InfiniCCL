/**
 * InfiniCCL Example: Send/Recv (OpenMPI + CCL Hybrid)
 *
 * OpenMPI launches the ranks and distributes the native CCL unique ID. After
 * rank-based CCL communicator initialization, rank 0 sends GPU data to rank 1
 * through the public `infinicclSend()` and `infinicclRecv()` APIs.
 */

#include <unistd.h>

#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include "backend_manifest.h"
#include "device.h"
#include "infiniccl.h"
#include "runtime.h"
#include "traits.h"
#include "utils.h"

namespace ccl = infini::ccl;

namespace {

constexpr int kSenderRank = 0;
constexpr int kReceiverRank = 1;
constexpr int kRequiredRanks = 2;
constexpr float kPayloadValue = 7.0f;

bool ParseLocalRank(const char *text, int *local_rank) {
  if (!text || !local_rank) {
    return false;
  }

  const char *end = text + std::strlen(text);
  auto result = std::from_chars(text, end, *local_rank);
  return result.ec == std::errc{} && result.ptr == end && *local_rank >= 0;
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

}  // namespace

int main(int argc, char **argv) {
  constexpr int kWarmupIters = 2;
  constexpr int kProfileIters = 20;
  constexpr size_t kNumElements = 1 << 20;

  constexpr ccl::Device::Type kDeviceType =
      ccl::ListGetBest<ccl::DevicePriority>(ccl::EnabledDevices{});
  using Rt = ccl::Runtime<kDeviceType>;

  CHECK_INFINI(infinicclInit(&argc, &argv));

  int rank = 0;
  int size = 0;
  CHECK_INFINI(infinicclGetRank(&rank));
  CHECK_INFINI(infinicclGetSize(&size));

  if (size < kRequiredRanks) {
    if (rank == kSenderRank) {
      std::cerr << "Hybrid Send/Recv requires at least two ranks." << std::endl;
    }
    CHECK_INFINI(infinicclFinalize());
    return EXIT_FAILURE;
  }

  int local_rank = -1;
  if (!ParseLocalRank(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"), &local_rank)) {
    std::cerr << "Rank " << rank
              << " received an invalid `OMPI_COMM_WORLD_LOCAL_RANK` value."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  CHECK_RT(Rt, Rt::SetDevice(local_rank));

  std::array<char, 256> hostname{};
  if (gethostname(hostname.data(), hostname.size() - 1) != 0) {
    std::cerr << "Rank " << rank << " failed to query the local hostname."
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::cout << "[Rank " << rank << "] Host: " << hostname.data()
            << " | GPU: " << ccl::Device::StringFromType(kDeviceType)
            << " | Device " << local_rank << std::endl;

  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitAll(&comm, size, nullptr));

  infinicclUniqueId unique_id{};
  if (rank == kSenderRank) {
    CHECK_INFINI(infinicclGetUniqueId(&unique_id));
  }
  CHECK_INFINI(infinicclBroadcast(&unique_id, &unique_id, sizeof(unique_id),
                                  infinicclChar, kSenderRank, comm, nullptr));
  CHECK_INFINI(infinicclCommInitRank(&comm, size, unique_id, rank));

  const bool is_sender = rank == kSenderRank;
  const bool is_receiver = rank == kReceiverRank;
  const bool transfers_data = is_sender || is_receiver;
  const size_t total_bytes = kNumElements * sizeof(float);

  std::vector<float> host_buffer(transfers_data ? kNumElements : 0,
                                 is_sender ? kPayloadValue : 0.0f);
  float *device_buffer = nullptr;

  if (transfers_data) {
    CHECK_RT(
        Rt, Rt::Malloc(reinterpret_cast<void **>(&device_buffer), total_bytes));
    CHECK_RT(Rt, Rt::Memcpy(device_buffer, host_buffer.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  }

  auto transfer = [&]() {
    if (is_sender) {
      return infinicclSend(device_buffer, kNumElements, infinicclFloat32,
                           kReceiverRank, comm, nullptr);
    }
    if (is_receiver) {
      return infinicclRecv(device_buffer, kNumElements, infinicclFloat32,
                           kSenderRank, comm, nullptr);
    }
    return infinicclSuccess;
  };

  for (int i = 0; i < kWarmupIters; ++i) {
    CHECK_INFINI(transfer());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));

  Timer timer;
  for (int i = 0; i < kProfileIters; ++i) {
    CHECK_INFINI(transfer());
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  double elapsed_ms = timer.ElapsedMs() / static_cast<double>(kProfileIters);

  bool correct = true;
  if (is_receiver) {
    CHECK_RT(Rt, Rt::Memcpy(host_buffer.data(), device_buffer, total_bytes,
                            Rt::MemcpyDeviceToHost));
    correct = Validator::ValidateResult(host_buffer.data(), kNumElements,
                                        kPayloadValue, rank, false,
                                        "Hybrid CCL Send/Recv");
    const char *color = correct ? "\033[32m" : "\033[31m";
    std::cout << "\n=== Hybrid CCL Send/Recv Results ===" << std::endl;
    std::cout << "Correct: " << color << (correct ? "YES" : "NO") << "\033[0m"
              << std::endl;
    std::cout << "Expect:  " << kPayloadValue << std::endl;
    std::cout << "Actual:  " << host_buffer.front() << std::endl;
    PrintSendRecvMetrics(kNumElements, elapsed_ms);
  }

  // The receiver publishes validation status from device memory. This works
  // with the current OpenMPI staging path and with native CCL `Broadcast`.
  std::int32_t completion_token = is_receiver && correct ? 1 : 0;
  std::int32_t *device_completion = nullptr;
  CHECK_RT(Rt, Rt::Malloc(reinterpret_cast<void **>(&device_completion),
                          sizeof(completion_token)));
  CHECK_RT(Rt, Rt::Memcpy(device_completion, &completion_token,
                          sizeof(completion_token), Rt::MemcpyHostToDevice));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_INFINI(infinicclBroadcast(device_completion, device_completion, 1,
                                  infinicclInt32, kReceiverRank, comm,
                                  nullptr));
  CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
  CHECK_RT(Rt, Rt::Memcpy(&completion_token, device_completion,
                          sizeof(completion_token), Rt::MemcpyDeviceToHost));
  correct = completion_token == 1;

  if (transfers_data) {
    CHECK_RT(Rt, Rt::Free(device_buffer));
  }
  CHECK_RT(Rt, Rt::Free(device_completion));
  CHECK_INFINI(infinicclCommDestroy(comm));
  CHECK_INFINI(infinicclFinalize());

  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
