/**
 * InfiniCCL Example: Two-Process Kunlun P2P + AllReduce
 *
 * Pipeline-parallel smoke test without an MPI launcher: rank 0 creates the
 * XCCL unique id and hands it to rank 1 over a TCP side channel, then both
 * ranks run Send/Recv and AllReduce through the BKCL backend.
 *
 * Usage:
 *   XPU_VISIBLE_DEVICES=0 ./pp_p2p --role sender   --master-port 29530
 *   XPU_VISIBLE_DEVICES=1 ./pp_p2p --role receiver --master-port 29530
 */

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "backend_manifest.h"
#include "infiniccl.h"
#include "utils.h"

using namespace infini::ccl;

namespace {

constexpr int kRankCount = 2;
constexpr size_t kNumElements = 1 << 16;
constexpr float kSendValue = 7.0f;

struct Options {
  std::string role = "sender";
  std::string master_addr = "127.0.0.1";
  int master_port = 29530;
  int device = -1;  // defaults to rank
};

Options ParseArgs(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      return (i + 1 < argc) ? std::string(argv[++i]) : std::string();
    };
    if (arg == "--role") {
      opts.role = next();
    } else if (arg == "--master-addr") {
      opts.master_addr = next();
    } else if (arg == "--master-port") {
      opts.master_port = std::stoi(next());
    } else if (arg == "--device") {
      opts.device = std::stoi(next());
    }
  }
  return opts;
}

int RankFromRole(const std::string& role) {
  if (role == "sender") {
    return 0;
  }
  if (role == "receiver") {
    return 1;
  }
  std::cerr << "Unknown role: " << role << " (use sender|receiver)"
            << std::endl;
  std::exit(EXIT_FAILURE);
}

// Rank 0 hosts the side channel; rank 1 dials in and fetches the unique id.
void ExchangeUniqueId(int rank, const Options& opts, char* id_bytes,
                      size_t id_len) {
  if (rank == 0) {
    const int server = ::socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) {
      std::cerr << "socket() failed" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    int reuse = 1;
    ::setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(opts.master_port));
    if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0 ||
        ::listen(server, 1) < 0) {
      std::cerr << "bind/listen failed on port " << opts.master_port
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    std::cout << "[rank0] waiting for rank1 on port " << opts.master_port
              << std::endl;
    const int conn = ::accept(server, nullptr, nullptr);
    if (conn < 0) {
      std::cerr << "accept() failed" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    size_t sent = 0;
    while (sent < id_len) {
      const ssize_t n = ::send(conn, id_bytes + sent, id_len - sent, 0);
      if (n <= 0) {
        std::cerr << "send(unique_id) failed" << std::endl;
        std::exit(EXIT_FAILURE);
      }
      sent += static_cast<size_t>(n);
    }
    ::close(conn);
    ::close(server);
    return;
  }

  const int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0) {
    std::cerr << "socket() failed" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(opts.master_port));
  if (::inet_pton(AF_INET, opts.master_addr.c_str(), &addr.sin_addr) != 1) {
    std::cerr << "invalid master addr: " << opts.master_addr << std::endl;
    std::exit(EXIT_FAILURE);
  }
  for (int attempt = 1; attempt <= 30; ++attempt) {
    if (::connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) ==
        0) {
      break;
    }
    if (attempt == 30) {
      std::cerr << "connect() to " << opts.master_addr << ":"
                << opts.master_port << " failed" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    ::sleep(1);
  }
  size_t received = 0;
  while (received < id_len) {
    const ssize_t n = ::recv(sock, id_bytes + received, id_len - received, 0);
    if (n <= 0) {
      std::cerr << "recv(unique_id) failed" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    received += static_cast<size_t>(n);
  }
  ::close(sock);
}

}  // namespace

int main(int argc, char** argv) {
  const Options opts = ParseArgs(argc, argv);
  const int rank = RankFromRole(opts.role);
  const int device = opts.device >= 0 ? opts.device : rank;

  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});
  using Rt = Runtime<kDevType>;

  CHECK_RT(Rt, Rt::SetDevice(device));
  std::cout << "[rank" << rank << "] using device " << device << std::endl;

  // 1. Bootstrap: distribute the XCCL unique id over the TCP side channel.
  infinicclUniqueId shared_id{};
  if (rank == 0) {
    CHECK_INFINI(infinicclGetUniqueId(&shared_id));
  }
  ExchangeUniqueId(rank, opts, shared_id.internal, sizeof(shared_id.internal));
  std::cout << "[rank" << rank << "] unique id exchanged" << std::endl;

  // 2. Build the communicator.
  infinicclComm_t comm = nullptr;
  CHECK_INFINI(infinicclCommInitRank(&comm, kRankCount, shared_id, rank));
  std::cout << "[rank" << rank << "] communicator ready" << std::endl;

  Rt::Stream stream = nullptr;
  CHECK_RT(Rt, Rt::StreamCreate(&stream));

  std::atomic_bool all_correct{true};

  float* device_buffer = nullptr;
  const size_t total_bytes = kNumElements * sizeof(float);
  CHECK_RT(Rt,
           Rt::Malloc(reinterpret_cast<void**>(&device_buffer), total_bytes));

  // 3. P2P send/recv: rank0 -> rank1.
  std::cout << "[rank" << rank << "] posting p2p " << std::endl;
  std::vector<float> host(kNumElements, rank == 0 ? kSendValue : 0.0f);
  CHECK_RT(Rt, Rt::Memcpy(device_buffer, host.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  std::cout << "[rank" << rank << "] h2d done, calling ccl" << std::endl;
  if (rank == 0) {
    CHECK_INFINI(infinicclSend(device_buffer, kNumElements, infinicclFloat32, 1,
                               comm, stream));
  } else {
    CHECK_INFINI(infinicclRecv(device_buffer, kNumElements, infinicclFloat32, 0,
                               comm, stream));
  }
  CHECK_RT(Rt, Rt::StreamSynchronize(stream));
  if (rank == 1) {
    CHECK_RT(Rt, Rt::Memcpy(host.data(), device_buffer, total_bytes,
                            Rt::MemcpyDeviceToHost));
    const bool correct = Validator::ValidateResult(
        host.data(), kNumElements, kSendValue, rank, true, "Send/Recv");
    all_correct.store(correct, std::memory_order_relaxed);
  }
  std::cout << "[rank" << rank << "] send/recv done" << std::endl;

  // 4. AllReduce(sum): each rank contributes rank+1 -> expect 3.0.
  std::cout << "[rank" << rank << "] posting all_reduce" << std::endl;
  const float own = static_cast<float>(rank + 1);
  std::fill(host.begin(), host.end(), own);
  CHECK_RT(Rt, Rt::Memcpy(device_buffer, host.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclAllReduce(device_buffer, device_buffer, kNumElements,
                                  infinicclFloat32, infinicclSum, comm,
                                  stream));
  CHECK_RT(Rt, Rt::StreamSynchronize(stream));
  CHECK_RT(Rt, Rt::Memcpy(host.data(), device_buffer, total_bytes,
                          Rt::MemcpyDeviceToHost));
  {
    const bool correct = Validator::ValidateResult(
        host.data(), kNumElements, 3.0f, rank, true, "AllReduce(sum)");
    if (!correct) {
      all_correct.store(false, std::memory_order_relaxed);
    }
  }

  // 5. Broadcast from root 0: everyone must observe kSendValue.
  std::cout << "[rank" << rank << "] posting broadcast" << std::endl;
  std::fill(host.begin(), host.end(), rank == 0 ? kSendValue : -1.0f);
  CHECK_RT(Rt, Rt::Memcpy(device_buffer, host.data(), total_bytes,
                          Rt::MemcpyHostToDevice));
  CHECK_INFINI(infinicclBroadcast(device_buffer, device_buffer, kNumElements,
                                  infinicclFloat32, 0, comm, stream));
  CHECK_RT(Rt, Rt::StreamSynchronize(stream));
  CHECK_RT(Rt, Rt::Memcpy(host.data(), device_buffer, total_bytes,
                          Rt::MemcpyDeviceToHost));
  {
    const bool correct = Validator::ValidateResult(
        host.data(), kNumElements, kSendValue, rank, true, "Broadcast");
    if (!correct) {
      all_correct.store(false, std::memory_order_relaxed);
    }
  }

  CHECK_RT(Rt, Rt::StreamDestroy(stream));
  CHECK_RT(Rt, Rt::Free(device_buffer));
  CHECK_INFINI(infinicclCommDestroy(comm));

  if (!all_correct.load(std::memory_order_relaxed)) {
    std::cerr << "[rank" << rank << "] PP P2P validation FAILED." << std::endl;
    return EXIT_FAILURE;
  }
  std::cout
      << "[rank" << rank
      << "] PP P2P validation passed (send/recv + all_reduce + broadcast)."
      << std::endl;
  return EXIT_SUCCESS;
}
