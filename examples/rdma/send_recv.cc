/**
 * InfiniCCL Example: RDMA Backend Host-buffer Send/Recv.
 *
 * The example distributes `infinicclUniqueId` over TCP. The backend uses a
 * separate TCP bootstrap socket and transfers payloads through verbs RC QPs.
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "infiniccl.h"

#if defined(INFINICCL_WITH_NVIDIA_RDMA)
#include "devices/nvidia/runtime_.h"
#elif defined(INFINICCL_WITH_METAX_RDMA)
#include "devices/metax/runtime_.h"
#endif

#if defined(INFINICCL_WITH_NVIDIA_RDMA) || defined(INFINICCL_WITH_METAX_RDMA)
#define INFINICCL_RDMA_GPU
#endif

namespace {

constexpr int kSenderRank = 0;
constexpr int kReceiverRank = 1;
constexpr int kRankCount = 2;
constexpr int kControlTimeoutMs = 30000;
constexpr uint16_t kDefaultRdmaPort = 7471;

#if defined(INFINICCL_WITH_NVIDIA_RDMA)
constexpr const char *kDeviceName = "nvidia";
#elif defined(INFINICCL_WITH_METAX_RDMA)
constexpr const char *kDeviceName = "metax";
#else
constexpr const char *kDeviceName = "cpu";
#endif

void CloseSocket(int socket_fd) {
  if (socket_fd >= 0) {
    (void)close(socket_fd);
  }
}

bool SendAll(int socket_fd, const void *data, size_t bytes) {
  const auto *current = static_cast<const uint8_t *>(data);
  while (bytes > 0) {
    const ssize_t sent = send(socket_fd, current, bytes, MSG_NOSIGNAL);
    if (sent < 0 && errno == EINTR) {
      continue;
    }
    if (sent <= 0) {
      return false;
    }
    current += sent;
    bytes -= static_cast<size_t>(sent);
  }
  return true;
}

bool ReceiveAll(int socket_fd, void *data, size_t bytes) {
  auto *current = static_cast<uint8_t *>(data);
  while (bytes > 0) {
    const ssize_t received = recv(socket_fd, current, bytes, 0);
    if (received < 0 && errno == EINTR) {
      continue;
    }
    if (received <= 0) {
      return false;
    }
    current += received;
    bytes -= static_cast<size_t>(received);
  }
  return true;
}

int CreateControlListener(const char *address, uint16_t port) {
  const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    return -1;
  }

  int reuse = 1;
  if (setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) !=
      0) {
    CloseSocket(socket_fd);
    return -1;
  }

  sockaddr_in endpoint{};
  endpoint.sin_family = AF_INET;
  endpoint.sin_port = htons(port);
  if (inet_pton(AF_INET, address, &endpoint.sin_addr) != 1) {
    CloseSocket(socket_fd);
    return -1;
  }

  if (bind(socket_fd, reinterpret_cast<sockaddr *>(&endpoint),
           sizeof(endpoint)) != 0 ||
      listen(socket_fd, 1) != 0) {
    CloseSocket(socket_fd);
    return -1;
  }

  return socket_fd;
}

int ConnectControl(const char *address, uint16_t port) {
  const int socket_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd < 0) {
    return -1;
  }

  sockaddr_in endpoint{};
  endpoint.sin_family = AF_INET;
  endpoint.sin_port = htons(port);
  if (inet_pton(AF_INET, address, &endpoint.sin_addr) != 1 ||
      connect(socket_fd, reinterpret_cast<sockaddr *>(&endpoint),
              sizeof(endpoint)) != 0) {
    CloseSocket(socket_fd);
    return -1;
  }

  return socket_fd;
}

int AcceptControl(int listener) {
  pollfd descriptor{listener, POLLIN, 0};
  int result;
  do {
    result = poll(&descriptor, 1, kControlTimeoutMs);
  } while (result < 0 && errno == EINTR);
  if (result <= 0 || !(descriptor.revents & POLLIN)) {
    return -1;
  }
  return accept(listener, nullptr, nullptr);
}

bool ParsePort(const char *value, uint16_t *port) {
  char *end = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
      parsed > 65535) {
    return false;
  }

  *port = static_cast<uint16_t>(parsed);
  return true;
}

bool ParseCount(const char *value, size_t *count) {
  char *end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed == 0 ||
      parsed > 262144 || value[0] == '-') {
    return false;
  }

  *count = static_cast<size_t>(parsed);
  return true;
}

bool CheckBuffer(const std::vector<int32_t> &buffer, int expected_rank,
                 size_t iteration) {
  for (size_t i = 0; i < buffer.size(); ++i) {
    const int32_t expected =
        static_cast<int32_t>(expected_rank * 1000000 + iteration + i);
    if (buffer[i] != expected) {
      std::cerr << "[Check] Mismatch at index " << i << ": expected "
                << expected << ", actual " << buffer[i] << std::endl;
      return false;
    }
  }
  return true;
}

bool CheckInfini(infinicclResult_t result, const char *operation) {
  if (result == infinicclSuccess) {
    return true;
  }

  std::cerr << "[InfiniCCL Error] " << operation << " returned " << result
            << "." << std::endl;
  return false;
}

#if defined(INFINICCL_WITH_NVIDIA_RDMA)
using GpuRuntime = infini::ccl::Runtime<infini::ccl::Device::Type::kNvidia>;
using GpuError = cudaError_t;
using GpuStream = cudaStream_t;
using GpuUuid = cudaUUID_t;
constexpr auto kGpuSuccess = cudaSuccess;
constexpr auto GpuGetErrorString = cudaGetErrorString;
constexpr auto GpuStreamCreate = cudaStreamCreate;
constexpr auto GpuStreamDestroy = cudaStreamDestroy;
constexpr const char *kGpuName = "NVIDIA";

GpuError GpuGetUuid(GpuUuid *uuid, int device) {
  cudaDeviceProp properties{};
  const auto status = cudaGetDeviceProperties(&properties, device);
  if (status == cudaSuccess) {
    *uuid = properties.uuid;
  }
  return status;
}
#elif defined(INFINICCL_WITH_METAX_RDMA)
using GpuRuntime = infini::ccl::Runtime<infini::ccl::Device::Type::kMetax>;
using GpuError = mcError_t;
using GpuStream = mcStream_t;
using GpuUuid = mcUuid_t;
constexpr auto kGpuSuccess = mcSuccess;
constexpr auto GpuGetErrorString = mcGetErrorString;
constexpr auto GpuGetUuid = mcDeviceGetUuid;
constexpr auto GpuStreamCreate = mcStreamCreate;
constexpr auto GpuStreamDestroy = mcStreamDestroy;
constexpr const char *kGpuName = "MetaX";
#endif

#if defined(INFINICCL_RDMA_GPU)
bool CheckGpu(GpuError status, const char *operation) {
  return CheckInfini(static_cast<infinicclResult_t>(GpuRuntime::Check(status)),
                     operation);
}

void PrintGpuDevice(int rank) {
  int current_device = -1;
  if (!CheckGpu(GpuRuntime::GetDevice(&current_device), "GetDevice")) {
    return;
  }
  GpuUuid uuid{};
  const auto status = GpuGetUuid(&uuid, current_device);
  if (status != kGpuSuccess) {
    std::cerr << "[Rank " << rank << "] " << kGpuName
              << " GPU device=" << current_device
              << ", UUID unavailable: " << GpuGetErrorString(status)
              << std::endl;
    return;
  }
  char uuid_text[sizeof(uuid) * 2 + 1]{};
  const auto *bytes = reinterpret_cast<const unsigned char *>(&uuid);
  for (size_t i = 0; i < sizeof(uuid); ++i) {
    std::snprintf(uuid_text + i * 2, 3, "%02x",
                  static_cast<unsigned int>(bytes[i]));
  }
  std::cout << "[Rank " << rank << "] GPU UUID(raw)=" << uuid_text << std::endl;
}

#endif

void PrintDeviceInfo(int rank, int device, const std::string &memory_mode) {
  char hostname[256]{};
  (void)gethostname(hostname, sizeof(hostname));
  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << kDeviceName << " | Device " << device
            << " | Memory: " << memory_mode << std::endl;
#if defined(INFINICCL_RDMA_GPU)
  PrintGpuDevice(rank);
#endif
}

void PrintTestSummary(size_t count, size_t iterations,
                      const std::string &memory_mode) {
  const size_t total_bytes = count * sizeof(int32_t);
  std::cout << "\n=== Performing RDMA Send/Recv ===" << std::endl;
  std::cout << "Sender rank: " << kSenderRank << std::endl;
  std::cout << "Receiver rank: " << kReceiverRank << std::endl;
  std::cout << "Data size: " << count << " int32 elements (" << total_bytes
            << " bytes)" << std::endl;
  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << "Memory mode: " << memory_mode << std::endl;
}

void PrintResult(int rank, const std::vector<int32_t> &buffer, int peer_rank,
                 size_t iteration) {
  const int32_t expected =
      static_cast<int32_t>(peer_rank * 1000000 + iteration);
  std::cout << "\n=== Send/Recv Results ===" << std::endl;
  std::cout << "Rank: " << rank << std::endl;
  std::cout << "Peer rank: " << peer_rank << std::endl;
  std::cout << "Iteration: " << iteration << std::endl;
  std::cout << "Correct: \033[32mYES\033[0m" << std::endl;
  std::cout << "Expect:  " << expected << std::endl;
  std::cout << "Actual:  " << buffer[0] << std::endl;
}

class ExampleBuffer {
 public:
  explicit ExampleBuffer(std::vector<int32_t> *host) : host_(host) {}

  ~ExampleBuffer() {
#if defined(INFINICCL_RDMA_GPU)
    if (device_) {
      (void)GpuRuntime::StreamSynchronize(stream_);
      (void)GpuRuntime::Free(device_);
    }
    if (stream_) {
      (void)GpuStreamDestroy(stream_);
    }
#endif
  }

  bool Initialize(const std::string &mode) {
    if (mode == "host") {
      return true;
    }
#if defined(INFINICCL_RDMA_GPU)
    return CheckGpu(GpuRuntime::Malloc(&device_, Bytes()), "Malloc") &&
           CheckGpu(GpuStreamCreate(&stream_), "StreamCreate");
#else
    std::cerr << "Device buffers require a NVIDIA or MetaX + RDMA build."
              << std::endl;
    return false;
#endif
  }

  bool Upload() {
#if defined(INFINICCL_RDMA_GPU)
    if (device_) {
      return CheckGpu(GpuRuntime::Memcpy(device_, host_->data(), Bytes(),
                                         GpuRuntime::MemcpyHostToDevice),
                      "Memcpy(H2D)");
    }
#endif
    return true;
  }

  bool Download() {
#if defined(INFINICCL_RDMA_GPU)
    if (device_) {
      if (!Synchronize()) {
        return false;
      }
      return CheckGpu(GpuRuntime::Memcpy(host_->data(), device_, Bytes(),
                                         GpuRuntime::MemcpyDeviceToHost),
                      "Memcpy(D2H validation)");
    }
#endif
    return true;
  }

  bool Synchronize() {
#if defined(INFINICCL_RDMA_GPU)
    if (device_) {
      return CheckGpu(GpuRuntime::StreamSynchronize(stream_),
                      "StreamSynchronize");
    }
#endif
    return true;
  }

  void *data() { return device_ ? device_ : host_->data(); }

  void *stream() {
#if defined(INFINICCL_RDMA_GPU)
    return reinterpret_cast<void *>(stream_);
#else
    return nullptr;
#endif
  }

 private:
  size_t Bytes() const { return host_->size() * sizeof(int32_t); }

  std::vector<int32_t> *host_;

  void *device_ = nullptr;

#if defined(INFINICCL_RDMA_GPU)
  GpuStream stream_ = nullptr;
#endif
};

bool ExchangeUniqueId(bool server, const char *control_address,
                      uint16_t control_port, infinicclUniqueId *id) {
  int listener = -1;
  int connection = -1;
  if (server) {
    listener = CreateControlListener(control_address, control_port);
    if (listener < 0) {
      std::cerr << "[Control] Failed to listen on " << control_address << ':'
                << control_port << "." << std::endl;
      return false;
    }

    std::cout << "[Control] Waiting for rank 1 on " << control_address << ':'
              << control_port << "..." << std::endl;
    connection = AcceptControl(listener);
    CloseSocket(listener);
    if (connection < 0) {
      std::cerr << "[Control] Failed to accept rank 1 within the control "
                   "timeout."
                << std::endl;
      return false;
    }

    const bool success = SendAll(connection, id, sizeof(*id));
    CloseSocket(connection);
    return success;
  }

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::milliseconds(kControlTimeoutMs);
  do {
    connection = ConnectControl(control_address, control_port);
    if (connection >= 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } while (std::chrono::steady_clock::now() < deadline);
  if (connection < 0) {
    std::cerr << "[Control] Failed to connect to " << control_address << ':'
              << control_port << "." << std::endl;
    return false;
  }

  const bool success = ReceiveAll(connection, id, sizeof(*id));
  CloseSocket(connection);
  return success;
}

struct Session {
  infinicclComm_t comm = nullptr;

  ~Session() {
    if (comm) {
      (void)infinicclCommDestroy(comm);
    }
    (void)infinicclFinalize();
  }
};

int Run(bool server, const char *bootstrap_address, const char *control_address,
        uint16_t control_port, size_t count, size_t iterations,
        const std::string &memory_mode, int device) {
#if defined(INFINICCL_RDMA_GPU)
  if (!CheckGpu(GpuRuntime::SetDevice(device), "SetDevice")) {
    return EXIT_FAILURE;
  }
#else
  (void)device;
  if (memory_mode != "host") {
    std::cerr << "Device buffers require a NVIDIA or MetaX + RDMA build."
              << std::endl;
    return EXIT_FAILURE;
  }
#endif
  if (server && setenv("INFINICCL_RDMA_ADDR", bootstrap_address, 1) != 0) {
    std::cerr << "[Setup] Failed to set `INFINICCL_RDMA_ADDR`." << std::endl;
    return EXIT_FAILURE;
  }

  const char *configured_port = std::getenv("INFINICCL_RDMA_PORT");
  const std::string rdma_port =
      configured_port ? configured_port : std::to_string(kDefaultRdmaPort);
  uint16_t bootstrap_port = 0;
  if (!ParsePort(rdma_port.c_str(), &bootstrap_port) ||
      bootstrap_port == control_port) {
    std::cerr << "The bootstrap port must be valid and distinct from the "
                 "Unique ID control port."
              << std::endl;
    return EXIT_FAILURE;
  }
  if (server && setenv("INFINICCL_RDMA_PORT", rdma_port.c_str(), 1) != 0) {
    std::cerr << "[Setup] Failed to set `INFINICCL_RDMA_PORT`." << std::endl;
    return EXIT_FAILURE;
  }

  int argc = 1;
  char program_name[] = "rdma_send_recv";
  char *init_argv[] = {program_name, nullptr};
  char **init_argv_pointer = init_argv;
  if (!CheckInfini(infinicclInit(&argc, &init_argv_pointer), "infinicclInit")) {
    return EXIT_FAILURE;
  }
  Session session;

  const int rank = server ? kSenderRank : kReceiverRank;
  PrintDeviceInfo(rank, device, memory_mode);
  std::vector<int32_t> send_buffer(count);
  std::vector<int32_t> recv_buffer(count, -1);
  ExampleBuffer send_memory(&send_buffer);
  ExampleBuffer recv_memory(&recv_buffer);
  for (size_t i = 0; i < count; ++i) {
    send_buffer[i] = static_cast<int32_t>(rank * 1000000 + i);
  }
  if (!send_memory.Initialize(memory_mode) ||
      !recv_memory.Initialize(memory_mode) || !send_memory.Upload() ||
      !send_memory.Synchronize() || !recv_memory.Upload() ||
      !recv_memory.Download()) {
    return EXIT_FAILURE;
  }
#if defined(INFINICCL_RDMA_GPU)
  if (!CheckGpu(GpuRuntime::StreamSynchronize(nullptr),
                "StreamSynchronize(default warm-up)")) {
    return EXIT_FAILURE;
  }
#endif

  infinicclUniqueId id{};
  if (server) {
    if (!CheckInfini(infinicclGetUniqueId(&id), "infinicclGetUniqueId")) {
      return EXIT_FAILURE;
    }
    std::cout << "[Rank 0] Generated TCP bootstrap ID for " << bootstrap_address
              << ':' << rdma_port << ". Payloads use verbs, not TCP."
              << std::endl;
  }

  if (!ExchangeUniqueId(server, control_address, control_port, &id)) {
    std::cerr << "[Control] Unique ID exchange failed." << std::endl;
    return EXIT_FAILURE;
  }

  if (!server) {
    std::cout << "[Rank 1] Received RDMA unique ID." << std::endl;
  }

  if (!CheckInfini(infinicclCommInitRank(&session.comm, kRankCount, id, rank),
                   "infinicclCommInitRank")) {
    return EXIT_FAILURE;
  }
  std::cout << "[Rank " << rank << "] RDMA communicator initialized."
            << std::endl;
  const infinicclComm_t comm = session.comm;

  if (rank == kSenderRank) {
    PrintTestSummary(count, iterations, memory_mode);
  }
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    std::fill(recv_buffer.begin(), recv_buffer.end(), -1);
    for (size_t i = 0; i < count; ++i) {
      send_buffer[i] = static_cast<int32_t>(rank * 1000000 + iteration + i);
    }
    if (!send_memory.Upload() || !recv_memory.Upload()) {
      return EXIT_FAILURE;
    }

    if (server) {
      if (!CheckInfini(infinicclRecv(recv_memory.data(), count, infinicclInt32,
                                     kReceiverRank, comm, recv_memory.stream()),
                       "infinicclRecv")) {
        return EXIT_FAILURE;
      }
      if (!recv_memory.Download()) {
        return EXIT_FAILURE;
      }
      if (!CheckBuffer(recv_buffer, kReceiverRank, iteration)) {
        return EXIT_FAILURE;
      }

      if (!CheckInfini(infinicclSend(send_memory.data(), count, infinicclInt32,
                                     kReceiverRank, comm, send_memory.stream()),
                       "infinicclSend")) {
        return EXIT_FAILURE;
      }
    } else {
      if (!CheckInfini(infinicclSend(send_memory.data(), count, infinicclInt32,
                                     kSenderRank, comm, send_memory.stream()),
                       "infinicclSend")) {
        return EXIT_FAILURE;
      }

      if (!CheckInfini(infinicclRecv(recv_memory.data(), count, infinicclInt32,
                                     kSenderRank, comm, recv_memory.stream()),
                       "infinicclRecv")) {
        return EXIT_FAILURE;
      }
      if (!recv_memory.Download()) {
        return EXIT_FAILURE;
      }
      if (!CheckBuffer(recv_buffer, kSenderRank, iteration)) {
        return EXIT_FAILURE;
      }
    }
  }

  // Rank 0 waits until rank 1 has verified the final reply before teardown.
  int32_t verified = 1;
  const auto ack_status =
      server ? infinicclRecv(&verified, 1, infinicclInt32, 1, comm, nullptr)
             : infinicclSend(&verified, 1, infinicclInt32, 0, comm, nullptr);
  if (!CheckInfini(ack_status, "Final verification acknowledgement") ||
      verified != 1) {
    return EXIT_FAILURE;
  }

  if (!CheckInfini(infinicclCommDestroy(comm), "infinicclCommDestroy")) {
    return EXIT_FAILURE;
  }
  session.comm = nullptr;

  PrintResult(rank, recv_buffer, 1 - rank, iterations - 1);
  if (rank == kSenderRank) {
    std::cout << "InfiniCCL finalized." << std::endl;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc >= 2 && std::string(argv[1]) == "auto") {
    const char *rank = std::getenv("OMPI_COMM_WORLD_RANK");
    const char *size = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!rank || !size || std::string(size) != "2" ||
        (std::string(rank) != "0" && std::string(rank) != "1")) {
      std::cerr << "The `auto` mode requires exactly two OpenMPI processes."
                << std::endl;
      return EXIT_FAILURE;
    }
    static char server_role[] = "server";
    static char client_role[] = "client";
    argv[1] = std::string(rank) == "0" ? server_role : client_role;
  }

  if (argc < 4 || argc > 8 ||
      (std::string(argv[1]) != "server" && std::string(argv[1]) != "client")) {
    std::cerr << "Usage: " << argv[0]
              << " server <bootstrap-ipv4> <control-port> [count] [iterations] "
                 "[memory] [gpu]\n"
              << "       " << argv[0]
              << " client <server-ipv4> <control-port> [count] [iterations] "
                 "[memory] [gpu]\n"
              << "       " << argv[0]
              << " auto <server-ipv4> <control-port> [count] [iterations] "
                 "[memory] [gpu]\n";
    return EXIT_FAILURE;
  }

  uint16_t control_port = 0;
  if (!ParsePort(argv[3], &control_port)) {
    std::cerr << "Invalid control port: " << argv[3] << "." << std::endl;
    return EXIT_FAILURE;
  }

  size_t count = 8;
  if (argc >= 5 && !ParseCount(argv[4], &count)) {
    std::cerr << "Invalid element count: " << argv[4] << "." << std::endl;
    return EXIT_FAILURE;
  }
  size_t iterations = 1;
  if (argc >= 6 && (!ParseCount(argv[5], &iterations) || iterations > 1000)) {
    std::cerr << "Iteration count must be between 1 and 1000." << std::endl;
    return EXIT_FAILURE;
  }
  const std::string memory_mode = argc >= 7 ? argv[6] : "host";
  if (memory_mode != "host" && memory_mode != "device") {
    std::cerr << "Memory mode must be host or device." << std::endl;
    return EXIT_FAILURE;
  }
  int device = 0;
  if (argc == 8) {
    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(argv[7], &end, 10);
    if (errno || end == argv[7] || *end || parsed < 0 || parsed > 255) {
      std::cerr << "Invalid GPU device index." << std::endl;
      return EXIT_FAILURE;
    }
    device = static_cast<int>(parsed);
  }

  const bool server = std::string(argv[1]) == "server";
  const char *bootstrap_address = argv[2];
  const char *control_address = argv[2];
  return Run(server, bootstrap_address, control_address, control_port, count,
             iterations, memory_mode, device);
}
