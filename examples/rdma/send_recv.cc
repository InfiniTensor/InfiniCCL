/**
 * InfiniCCL Example: RDMA Backend Host-buffer Send/Recv.
 *
 * The example distributes `infinicclUniqueId` over TCP. The backend uses a
 * separate TCP bootstrap socket and transfers payloads through verbs RC QPs.
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
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

#if defined(INFINICCL_WITH_METAX_RDMA)
#include "devices/metax/runtime_.h"
#endif

namespace {

constexpr int kSenderRank = 0;
constexpr int kReceiverRank = 1;
constexpr int kRankCount = 2;
constexpr uint16_t kDefaultRdmaPort = 7471;
constexpr size_t kPreviewCount = 8;

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

void PrintBuffer(const char *label, const std::vector<int32_t> &buffer) {
  std::cout << label << " [";
  const size_t preview_count = std::min(kPreviewCount, buffer.size());
  for (size_t i = 0; i < preview_count; ++i) {
    if (i != 0) {
      std::cout << ", ";
    }
    std::cout << buffer[i];
  }
  if (buffer.size() > preview_count) {
    std::cout << ", ...";
  }
  std::cout << "]" << std::endl;
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

#if defined(INFINICCL_WITH_METAX_RDMA)
using MetaxRuntime = infini::ccl::Runtime<infini::ccl::Device::Type::kMetax>;

bool CheckMetax(mcError_t status, const char *operation) {
  return CheckInfini(
      static_cast<infinicclResult_t>(MetaxRuntime::Check(status)), operation);
}

void PrintMetaxDevice(int rank) {
  int current_device = -1;
  if (!CheckMetax(MetaxRuntime::GetDevice(&current_device), "mcGetDevice")) {
    return;
  }
  mcUuid_t uuid{};
  const auto status = mcDeviceGetUuid(&uuid, current_device);
  if (status != mcSuccess) {
    std::cerr << "[Rank " << rank << "] MetaX GPU device=" << current_device
              << ", UUID unavailable: " << mcGetErrorString(status)
              << std::endl;
    return;
  }
  char uuid_text[sizeof(uuid) * 2 + 1]{};
  const auto *bytes = reinterpret_cast<const unsigned char *>(&uuid);
  for (size_t i = 0; i < sizeof(uuid); ++i) {
    std::snprintf(uuid_text + i * 2, 3, "%02x",
                  static_cast<unsigned int>(bytes[i]));
  }
  std::cout << "[Rank " << rank << "] MetaX GPU device=" << current_device
            << ", UUID(raw)=" << uuid_text << std::endl;
}

__global__ void FillBuffer(int32_t *buffer, size_t count, int32_t first,
                           bool sequential) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    buffer[index] = first + (sequential ? static_cast<int32_t>(index) : 0);
  }
}

__global__ void ConsumeBuffer(const int32_t *source, int32_t *result,
                              size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    result[index] = source[index];
  }
}
#endif

class ExampleBuffer {
 public:
  explicit ExampleBuffer(std::vector<int32_t> *host) : host_(host) {}

  ~ExampleBuffer() {
#if defined(INFINICCL_WITH_METAX_RDMA)
    if (device_) {
      (void)mcStreamSynchronize(stream_);
      (void)MetaxRuntime::Free(device_);
    }
    if (consumed_) {
      (void)MetaxRuntime::Free(consumed_);
    }
    if (stream_) {
      (void)mcStreamDestroy(stream_);
    }
#endif
  }

  bool Initialize(const std::string &mode) {
    if (mode == "host") {
      return true;
    }
#if defined(INFINICCL_WITH_METAX_RDMA)
    kernels_ = mode == "device-kernel";
    if (!CheckMetax(MetaxRuntime::Malloc(&device_, Bytes()), "mcMalloc") ||
        !CheckMetax(mcStreamCreate(&stream_), "mcStreamCreate")) {
      return false;
    }
    return !kernels_ ||
           CheckMetax(MetaxRuntime::Malloc(&consumed_, Bytes()), "mcMalloc");
#else
    std::cerr << "Device buffers require a MetaX + RDMA build." << std::endl;
    return false;
#endif
  }

  bool Upload(bool sequential) {
#if defined(INFINICCL_WITH_METAX_RDMA)
    if (device_) {
      if (kernels_) {
        FillBuffer<<<(host_->size() + 255) / 256, 256, 0, stream_>>>(
            static_cast<int32_t *>(device_), host_->size(), (*host_)[0],
            sequential);
        return CheckMetax(mcGetLastError(), "FillBuffer");
      }
      return CheckMetax(MetaxRuntime::Memcpy(device_, host_->data(), Bytes(),
                                             MetaxRuntime::MemcpyHostToDevice),
                        "mcMemcpy(H2D)");
    }
#else
    (void)sequential;
#endif
    return true;
  }

  bool Download() {
#if defined(INFINICCL_WITH_METAX_RDMA)
    if (device_) {
      void *source = device_;
      if (kernels_) {
        ConsumeBuffer<<<(host_->size() + 255) / 256, 256, 0, stream_>>>(
            static_cast<const int32_t *>(device_),
            static_cast<int32_t *>(consumed_), host_->size());
        if (!CheckMetax(mcGetLastError(), "ConsumeBuffer")) {
          return false;
        }
        source = consumed_;
      }
      if (!Synchronize()) {
        return false;
      }
      return CheckMetax(MetaxRuntime::Memcpy(host_->data(), source, Bytes(),
                                             MetaxRuntime::MemcpyDeviceToHost),
                        "mcMemcpy(D2H validation)");
    }
#endif
    return true;
  }

  bool Synchronize() {
#if defined(INFINICCL_WITH_METAX_RDMA)
    if (device_) {
      return CheckMetax(mcStreamSynchronize(stream_), "mcStreamSynchronize");
    }
#endif
    return true;
  }

  void *data() { return device_ ? device_ : host_->data(); }

  void *stream() {
#if defined(INFINICCL_WITH_METAX_RDMA)
    return reinterpret_cast<void *>(stream_);
#else
    return nullptr;
#endif
  }

 private:
  size_t Bytes() const { return host_->size() * sizeof(int32_t); }

  std::vector<int32_t> *host_;

  void *device_ = nullptr;

#if defined(INFINICCL_WITH_METAX_RDMA)
  void *consumed_ = nullptr;

  mcStream_t stream_ = nullptr;

  bool kernels_ = false;
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
    connection = accept(listener, nullptr, nullptr);
    CloseSocket(listener);
    if (connection < 0) {
      std::cerr << "[Control] Failed to accept rank 1." << std::endl;
      return false;
    }

    const bool success = SendAll(connection, id, sizeof(*id));
    CloseSocket(connection);
    return success;
  }

  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
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
#if defined(INFINICCL_WITH_METAX_RDMA)
  if (!CheckMetax(MetaxRuntime::SetDevice(device), "mcSetDevice")) {
    return EXIT_FAILURE;
  }
#else
  (void)device;
  if (memory_mode != "host") {
    std::cerr << "Device buffers require a MetaX + RDMA build." << std::endl;
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
#if defined(INFINICCL_WITH_METAX_RDMA)
  PrintMetaxDevice(rank);
#endif
  std::vector<int32_t> send_buffer(count);
  std::vector<int32_t> recv_buffer(count, -1);
  ExampleBuffer send_memory(&send_buffer);
  ExampleBuffer recv_memory(&recv_buffer);
  for (size_t i = 0; i < count; ++i) {
    send_buffer[i] = static_cast<int32_t>(rank * 1000000 + i);
  }
  if (!send_memory.Initialize(memory_mode) ||
      !recv_memory.Initialize(memory_mode) || !send_memory.Upload(true) ||
      !send_memory.Synchronize() || !recv_memory.Upload(false) ||
      !recv_memory.Download()) {
    return EXIT_FAILURE;
  }
#if defined(INFINICCL_WITH_METAX_RDMA)
  if (!CheckMetax(MetaxRuntime::StreamSynchronize(nullptr),
                  "mcStreamSynchronize(default warm-up)")) {
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

  std::cout << "[Rank " << rank << "] Local memory mode: " << memory_mode << "."
            << std::endl;
  for (size_t iteration = 0; iteration < iterations; ++iteration) {
    std::cout << "[Rank " << rank << "] Iteration " << iteration << "."
              << std::endl;
    std::fill(recv_buffer.begin(), recv_buffer.end(), -1);
    for (size_t i = 0; i < count; ++i) {
      send_buffer[i] = static_cast<int32_t>(rank * 1000000 + iteration + i);
    }
    if (!send_memory.Upload(true) || !recv_memory.Upload(false) ||
        !recv_memory.Download()) {
      return EXIT_FAILURE;
    }

    if (server) {
      PrintBuffer("[Rank 0] Receive buffer before Recv", recv_buffer);
      if (!CheckInfini(infinicclRecv(recv_memory.data(), count, infinicclInt32,
                                     kReceiverRank, comm, recv_memory.stream()),
                       "infinicclRecv")) {
        return EXIT_FAILURE;
      }
      if (!recv_memory.Download()) {
        return EXIT_FAILURE;
      }
      PrintBuffer("[Rank 0] Received from rank 1", recv_buffer);
      if (!CheckBuffer(recv_buffer, kReceiverRank, iteration)) {
        return EXIT_FAILURE;
      }
      std::cout << "[Rank 0] Receive check: PASS." << std::endl;

      PrintBuffer("[Rank 0] Sending to rank 1", send_buffer);
      if (!CheckInfini(infinicclSend(send_memory.data(), count, infinicclInt32,
                                     kReceiverRank, comm, send_memory.stream()),
                       "infinicclSend")) {
        return EXIT_FAILURE;
      }
      std::cout << "[Rank 0] Send complete." << std::endl;
    } else {
      PrintBuffer("[Rank 1] Sending to rank 0", send_buffer);
      if (!CheckInfini(infinicclSend(send_memory.data(), count, infinicclInt32,
                                     kSenderRank, comm, send_memory.stream()),
                       "infinicclSend")) {
        return EXIT_FAILURE;
      }
      std::cout << "[Rank 1] Send complete." << std::endl;

      PrintBuffer("[Rank 1] Receive buffer before Recv", recv_buffer);
      if (!CheckInfini(infinicclRecv(recv_memory.data(), count, infinicclInt32,
                                     kSenderRank, comm, recv_memory.stream()),
                       "infinicclRecv")) {
        return EXIT_FAILURE;
      }
      if (!recv_memory.Download()) {
        return EXIT_FAILURE;
      }
      PrintBuffer("[Rank 1] Received from rank 0", recv_buffer);
      if (!CheckBuffer(recv_buffer, kSenderRank, iteration)) {
        return EXIT_FAILURE;
      }
      std::cout << "[Rank 1] Receive check: PASS." << std::endl;
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

  if (memory_mode == "host") {
    std::cout << "[Rank " << rank << "] PASS: RDMA Host-buffer Send/Recv."
              << std::endl;
  } else {
    std::cout << "[Rank " << rank
              << "] PASS: RDMA Send/Recv, local memory=" << memory_mode
              << ". DeviceDirect visibility remains experimental." << std::endl;
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
  if (memory_mode != "host" && memory_mode != "device" &&
      memory_mode != "device-kernel") {
    std::cerr << "Memory mode must be host, device or device-kernel."
              << std::endl;
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
