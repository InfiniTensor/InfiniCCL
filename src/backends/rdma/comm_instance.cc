#include "backends/rdma/comm_instance.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <netdb.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#if defined(INFINICCL_WITH_MLX5DV)
#include <infiniband/mlx5dv.h>
#endif

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "backends/rdma/memory_adaptor.h"
#include "logging.h"

namespace infini::ccl {

struct RdmaResources {
  ibv_context *context = nullptr;

  ibv_pd *pd = nullptr;

  ibv_cq *send_cq = nullptr;

  ibv_cq *recv_cq = nullptr;

  ibv_qp *qp = nullptr;

  ibv_port_attr port_attr{};

  ibv_gid gid{};

  int socket = -1;

  int rank = -1;

  int port = 1;

  int gid_index = -1;

  int pkey_index = 0;

  size_t max_bytes = 0;

  std::mutex mutex;
};

namespace {

constexpr int kTimeoutMs = 30000;

struct PeerMetadata {
  uint32_t rank;

  uint32_t qp_num;

  uint32_t lid;

  uint32_t mtu;

  uint32_t link_layer;

  uint32_t has_gid;

  uint32_t max_bytes;

  uint32_t pkey;

  uint8_t gid[16];

  char group_id[INFINICCL_RDMA_GROUP_ID_BYTES];
};

static_assert(sizeof(PeerMetadata) ==
                  8 * sizeof(uint32_t) + 16 + INFINICCL_RDMA_GROUP_ID_BYTES,
              "`PeerMetadata` must not contain padding.");

ReturnStatus WaitForRdmaCompletion(ibv_cq *cq, bool receive,
                                   size_t expected_bytes) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    ibv_wc completion{};
    const int count = ibv_poll_cq(cq, 1, &completion);
    if (count < 0) {
      LOG("Failed to poll the RDMA completion queue.");
      return ReturnStatus::kSystemError;
    }
    if (count == 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      continue;
    }
    if (completion.status != IBV_WC_SUCCESS) {
      std::cerr << "RDMA completion failed: "
                << ibv_wc_status_str(completion.status)
                << ", vendor error: " << completion.vendor_err << ".\n";
      return ReturnStatus::kRemoteError;
    }
    if (completion.wr_id != 1 ||
        completion.opcode != (receive ? IBV_WC_RECV : IBV_WC_SEND) ||
        (receive && completion.byte_len != expected_bytes)) {
      LOG("Unexpected RDMA completion or message size.");
      return ReturnStatus::kRemoteError;
    }
    return ReturnStatus::kSuccess;
  }
  LOG("Timed out while waiting for an RDMA completion.");
  return ReturnStatus::kSystemError;
}

ReturnStatus ReportVerbsError(const char *operation, int error) {
  std::cerr << "RDMA operation `" << operation
            << "` failed: " << std::strerror(error) << " (" << error << ").\n";
  return ReturnStatus::kSystemError;
}

bool ReadOption(const char *name, int default_value, int minimum, int maximum,
                int *result) {
  const char *value = std::getenv(name);
  if (!value) {
    *result = default_value;
    return true;
  }
  char *end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno || end == value || *end || parsed < minimum || parsed > maximum) {
    std::cerr << "Invalid option `" << name << "`.\n";
    return false;
  }
  *result = static_cast<int>(parsed);
  return true;
}

bool WaitSocket(int fd, short events, int timeout_ms) {
  pollfd descriptor{fd, events, 0};
  int result;
  do {
    result = poll(&descriptor, 1, timeout_ms);
  } while (result < 0 && errno == EINTR);
  return result > 0 && (descriptor.revents & events);
}

int AcceptBootstrap(int fd, const addrinfo &address) {
  const int reuse = 1;
  (void)setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  if (bind(fd, address.ai_addr, address.ai_addrlen) != 0 ||
      listen(fd, 1) != 0 || !WaitSocket(fd, POLLIN, kTimeoutMs)) {
    return -1;
  }
  return accept4(fd, nullptr, nullptr, SOCK_NONBLOCK);
}

bool ConnectBootstrap(int fd, const addrinfo &address) {
  if (connect(fd, address.ai_addr, address.ai_addrlen) == 0) {
    return true;
  }
  if (errno != EINPROGRESS || !WaitSocket(fd, POLLOUT, 200)) {
    return false;
  }
  int error = 0;
  socklen_t length = sizeof(error);
  return getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &length) == 0 &&
         error == 0;
}

int TryBootstrapAddresses(const addrinfo *addresses, bool server) {
  for (auto *address = addresses; address; address = address->ai_next) {
    const int fd = socket(address->ai_family, SOCK_STREAM | SOCK_NONBLOCK,
                          address->ai_protocol);
    if (fd < 0) {
      continue;
    }
    if (!server && ConnectBootstrap(fd, *address)) {
      return fd;
    }
    const int connection = server ? AcceptBootstrap(fd, *address) : -1;
    close(fd);
    if (connection >= 0) {
      return connection;
    }
  }
  return -1;
}

int OpenBootstrap(const RdmaUniqueId &id, bool server) {
  addrinfo hints{};
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = server ? AI_PASSIVE : 0;
  addrinfo *addresses = nullptr;
  const std::string service = std::to_string(id.port);
  if (getaddrinfo(id.address, service.c_str(), &hints, &addresses) != 0) {
    return -1;
  }
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  int connection = -1;
  do {
    connection = TryBootstrapAddresses(addresses, server);
    if (server || connection >= 0) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  } while (std::chrono::steady_clock::now() < deadline);
  freeaddrinfo(addresses);
  return connection;
}

ReturnStatus TransferControl(int socket, size_t bytes, bool receive,
                             void *data) {
  auto *current = static_cast<char *>(data);
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(kTimeoutMs);
  while (bytes) {
    const auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now())
            .count();
    if (remaining <= 0 || !WaitSocket(socket, receive ? POLLIN : POLLOUT,
                                      static_cast<int>(remaining))) {
      LOG("RDMA bootstrap/control channel timed out or closed.");
      return ReturnStatus::kRemoteError;
    }
    const ssize_t count = receive ? recv(socket, current, bytes, 0)
                                  : send(socket, current, bytes, MSG_NOSIGNAL);
    if (count < 0 && (errno == EINTR || errno == EAGAIN)) {
      continue;
    }
    if (count <= 0) {
      LOG("Failed to transfer RDMA control metadata.");
      return ReturnStatus::kRemoteError;
    }
    current += count;
    bytes -= static_cast<size_t>(count);
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus SelectDevice(RdmaResources &resources) {
  int count = 0;
  ibv_device **devices = ibv_get_device_list(&count);
  if (!devices) {
    return ReportVerbsError("ibv_get_device_list", errno);
  }
  const char *name = std::getenv("INFINICCL_RDMA_DEVICE");
  for (int i = 0; i < count; ++i) {
    if (name && std::strcmp(name, ibv_get_device_name(devices[i])) != 0) {
      continue;
    }
    resources.context = ibv_open_device(devices[i]);
    if (!resources.context) {
      continue;
    }
    if (ibv_query_port(resources.context, resources.port,
                       &resources.port_attr) == 0 &&
        resources.port_attr.state == IBV_PORT_ACTIVE) {
      break;
    }
    ibv_close_device(resources.context);
    resources.context = nullptr;
  }
  ibv_free_device_list(devices);
  if (!resources.context) {
    LOG("No selected RDMA device has an active port.");
    return ReturnStatus::kSystemError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus ValidatePort(RdmaResources &resources) {
  if (resources.port_attr.link_layer == IBV_LINK_LAYER_ETHERNET &&
      resources.gid_index < 0) {
    LOG("RoCE requires an explicit `INFINICCL_RDMA_GID_INDEX`.");
    return ReturnStatus::kInvalidArgument;
  }
  if ((resources.port_attr.flags & IBV_QPF_GRH_REQUIRED) &&
      resources.gid_index < 0) {
    LOG("This port requires a global route and an explicit GID index.");
    return ReturnStatus::kNotSupported;
  }
  if (resources.pkey_index >= resources.port_attr.pkey_tbl_len ||
      (resources.gid_index >= 0 &&
       (resources.gid_index >= resources.port_attr.gid_tbl_len ||
        ibv_query_gid(resources.context, resources.port, resources.gid_index,
                      &resources.gid) != 0))) {
    return ReturnStatus::kInvalidArgument;
  }
  return ReturnStatus::kSuccess;
}

#if defined(INFINICCL_WITH_MLX5DV)
ReturnStatus CreateMlx5QueuePair(const ibv_qp_init_attr &base,
                                 RdmaResources &resources) {
  ibv_qp_init_attr_ex attr{};
  attr.qp_context = base.qp_context;
  attr.send_cq = base.send_cq;
  attr.recv_cq = base.recv_cq;
  attr.srq = base.srq;
  attr.cap = base.cap;
  attr.qp_type = base.qp_type;
  attr.sq_sig_all = base.sq_sig_all;
  attr.comp_mask = IBV_QP_INIT_ATTR_PD;
  attr.pd = resources.pd;
  mlx5dv_qp_init_attr mlx5_attr{};
  mlx5_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
  mlx5_attr.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
  resources.qp = mlx5dv_create_qp(resources.context, &attr, &mlx5_attr);
  if (!resources.qp) {
    return ReportVerbsError("mlx5dv_create_qp(disable Scatter-to-CQE)", errno);
  }
  std::cout << "[Rank " << resources.rank
            << "] mlx5 Scatter-to-CQE disabled explicitly for this QP."
            << std::endl;
  return ReturnStatus::kSuccess;
}
#endif

ReturnStatus CreateCompatibleQueuePair(ibv_qp_init_attr *attr,
                                       RdmaResources &resources) {
#if defined(INFINICCL_WITH_MLX5DV)
  if (mlx5dv_is_supported(resources.context->device)) {
    return CreateMlx5QueuePair(*attr, resources);
  }
#endif
  ibv_device_attr device_attr{};
  const int status = ibv_query_device(resources.context, &device_attr);
  if (status != 0) {
    return ReportVerbsError("ibv_query_device", status);
  }
  const char *name = ibv_get_device_name(resources.context->device);
  if (device_attr.vendor_id == 0x02c9 || std::strncmp(name, "mlx5", 4) == 0) {
    LOG("Mellanox QP creation requires mlx5dv Scatter-to-CQE control. "
        "Install compatible mlx5 development headers/libraries and rebuild.");
    return ReturnStatus::kNotSupported;
  }
  resources.qp = ibv_create_qp(resources.pd, attr);
  if (!resources.qp) {
    return ReportVerbsError("ibv_create_qp", errno);
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus CreateQueuePair(RdmaResources &resources) {
  resources.pd = ibv_alloc_pd(resources.context);
  resources.send_cq = ibv_create_cq(resources.context, 16, nullptr, nullptr, 0);
  resources.recv_cq = ibv_create_cq(resources.context, 16, nullptr, nullptr, 0);
  if (!resources.pd || !resources.send_cq || !resources.recv_cq) {
    return ReportVerbsError("PD/CQ allocation", errno);
  }
  ibv_qp_init_attr attr{};
  attr.send_cq = resources.send_cq;
  attr.recv_cq = resources.recv_cq;
  attr.cap.max_send_wr = 16;
  attr.cap.max_recv_wr = 16;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  attr.qp_type = IBV_QPT_RC;
  const auto status = CreateCompatibleQueuePair(&attr, resources);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  resources.max_bytes =
      std::min<size_t>(resources.port_attr.max_msg_sz, 1 << 20);
  return ReturnStatus::kSuccess;
}

ReturnStatus OpenDevice(RdmaResources &resources) {
  if (!ReadOption("INFINICCL_RDMA_IB_PORT", 1, 1, 255, &resources.port) ||
      !ReadOption("INFINICCL_RDMA_GID_INDEX", -1, -1, 255,
                  &resources.gid_index) ||
      !ReadOption("INFINICCL_RDMA_PKEY_INDEX", 0, 0, 65535,
                  &resources.pkey_index)) {
    return ReturnStatus::kInvalidArgument;
  }
  auto status = SelectDevice(resources);
  if (status == ReturnStatus::kSuccess) {
    status = ValidatePort(resources);
  }
  if (status == ReturnStatus::kSuccess) {
    status = CreateQueuePair(resources);
  }
  if (status == ReturnStatus::kSuccess) {
    std::cout << "[Rank " << resources.rank << "] RDMA device: "
              << ibv_get_device_name(resources.context->device)
              << ", port: " << resources.port
              << ", LID: " << resources.port_attr.lid
              << ", GID index: " << resources.gid_index << "." << std::endl;
  }
  return status;
}

ReturnStatus BuildPeerMetadata(const RdmaResources &resources,
                               const RdmaUniqueId &id, PeerMetadata *local) {
  uint16_t pkey = 0;
  if (ibv_query_pkey(resources.context, resources.port, resources.pkey_index,
                     &pkey) != 0 ||
      (ntohs(pkey) & 0x7fff) == 0) {
    return ReturnStatus::kInvalidArgument;
  }
  local->rank = htonl(resources.rank);
  local->qp_num = htonl(resources.qp->qp_num);
  local->lid = htonl(resources.port_attr.lid);
  local->mtu = htonl(resources.port_attr.active_mtu);
  local->link_layer = htonl(resources.port_attr.link_layer);
  local->has_gid = htonl(resources.gid_index >= 0);
  local->max_bytes = htonl(resources.max_bytes);
  local->pkey = htonl(ntohs(pkey));
  std::memcpy(local->gid, resources.gid.raw, sizeof(local->gid));
  std::memcpy(local->group_id, id.group_id, sizeof(local->group_id));
  return ReturnStatus::kSuccess;
}

ReturnStatus ValidatePeerMetadata(const RdmaResources &resources,
                                  const PeerMetadata &local,
                                  const PeerMetadata &remote) {
  if (ntohl(remote.rank) != static_cast<uint32_t>(1 - resources.rank) ||
      remote.link_layer != local.link_layer ||
      remote.has_gid != local.has_gid ||
      (ntohl(remote.pkey) & 0x7fff) != (ntohl(local.pkey) & 0x7fff) ||
      std::memcmp(remote.group_id, local.group_id, sizeof(remote.group_id)) !=
          0 ||
      ntohl(remote.qp_num) == 0 || ntohl(remote.qp_num) > 0xffffff ||
      ntohl(remote.lid) > 65535 ||
      (resources.gid_index < 0 && ntohl(remote.lid) == 0) ||
      ntohl(remote.mtu) < IBV_MTU_256 || ntohl(remote.mtu) > IBV_MTU_4096 ||
      ntohl(remote.max_bytes) == 0) {
    LOG("Incompatible RDMA peer metadata or communication group.");
    return ReturnStatus::kRemoteError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus InitializeQueuePair(const RdmaResources &resources) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = resources.port;
  attr.pkey_index = resources.pkey_index;
  const int status = ibv_modify_qp(
      resources.qp, &attr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_modify_qp(INIT)", status);
}

ReturnStatus ReadyToReceive(const RdmaResources &resources,
                            const PeerMetadata &remote) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = static_cast<ibv_mtu>(
      std::min<uint32_t>(resources.port_attr.active_mtu, ntohl(remote.mtu)));
  attr.dest_qp_num = ntohl(remote.qp_num);
  attr.rq_psn = attr.dest_qp_num & 0xffffff;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = ntohl(remote.lid);
  attr.ah_attr.port_num = resources.port;
  if (resources.gid_index >= 0) {
    attr.ah_attr.is_global = 1;
    std::memcpy(attr.ah_attr.grh.dgid.raw, remote.gid,
                sizeof(attr.ah_attr.grh.dgid.raw));
    attr.ah_attr.grh.sgid_index = resources.gid_index;
    attr.ah_attr.grh.hop_limit = 64;
  }
  const int status = ibv_modify_qp(
      resources.qp, &attr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_modify_qp(RTR)", status);
}

ReturnStatus ReadyToSend(const RdmaResources &resources) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = resources.qp->qp_num & 0xffffff;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  const int status = ibv_modify_qp(
      resources.qp, &attr,
      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
          IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_modify_qp(RTS)", status);
}

ReturnStatus SynchronizeQueuePair(int socket) {
  uint32_t ready = htonl(1);
  auto status = TransferControl(socket, sizeof(ready), false, &ready);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  status = TransferControl(socket, sizeof(ready), true, &ready);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  return ntohl(ready) == 1 ? ReturnStatus::kSuccess
                           : ReturnStatus::kRemoteError;
}

ReturnStatus ExchangePeerMetadata(const RdmaUniqueId &id,
                                  RdmaResources &resources,
                                  PeerMetadata *remote) {
  PeerMetadata local{};
  auto status = BuildPeerMetadata(resources, id, &local);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  status = TransferControl(resources.socket, sizeof(local), false, &local);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  status = TransferControl(resources.socket, sizeof(*remote), true, remote);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  status = ValidatePeerMetadata(resources, local, *remote);
  if (status == ReturnStatus::kSuccess) {
    resources.max_bytes =
        std::min<size_t>(resources.max_bytes, ntohl(remote->max_bytes));
  }
  return status;
}

ReturnStatus ConnectQueuePair(const RdmaUniqueId &id,
                              RdmaResources &resources) {
  PeerMetadata remote{};
  auto status = ExchangePeerMetadata(id, resources, &remote);
  if (status == ReturnStatus::kSuccess) {
    status = InitializeQueuePair(resources);
  }
  if (status == ReturnStatus::kSuccess) {
    status = ReadyToReceive(resources, remote);
  }
  if (status == ReturnStatus::kSuccess) {
    status = ReadyToSend(resources);
  }
  if (status == ReturnStatus::kSuccess) {
    status = SynchronizeQueuePair(resources.socket);
  }
  return status;
}

void DestroyQueuePair(RdmaResources &resources) {
  if (resources.qp) {
    const int status = ibv_destroy_qp(resources.qp);
    if (status != 0) {
      std::cerr << __FILE__ << ':' << __LINE__ << " " << __func__
                << ": Failed to destroy the RDMA QP safely.\n";
      assert(status == 0 && "Failed to destroy the RDMA QP safely.");
      std::abort();
    }
    resources.qp = nullptr;
  }
}

ReturnStatus PostReceive(ibv_qp *qp, ibv_sge *sge) {
  ibv_recv_wr wr{};
  wr.wr_id = 1;
  wr.sg_list = sge;
  wr.num_sge = 1;
  ibv_recv_wr *bad = nullptr;
  const int status = ibv_post_recv(qp, &wr, &bad);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_post_recv", status);
}

ReturnStatus PostSend(ibv_qp *qp, ibv_sge *sge) {
  ibv_send_wr wr{};
  wr.wr_id = 1;
  wr.sg_list = sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  ibv_send_wr *bad = nullptr;
  const int status = ibv_post_send(qp, &wr, &bad);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_post_send", status);
}

ReturnStatus ExchangeReceiveReady(int socket, size_t bytes, bool receive) {
  uint32_t ready_bytes = htonl(static_cast<uint32_t>(bytes));
  const auto status =
      TransferControl(socket, sizeof(ready_bytes), !receive, &ready_bytes);
  if (status != ReturnStatus::kSuccess) {
    return status;
  }
  if (!receive && ntohl(ready_bytes) != bytes) {
    LOG("RDMA send and receive sizes do not match.");
    return ReturnStatus::kRemoteError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus TransferRegisteredBuffer(const RdmaResources &resources,
                                      void *buffer, size_t bytes, ibv_mr *mr,
                                      bool receive) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(buffer);
  sge.length = static_cast<uint32_t>(bytes);
  sge.lkey = mr->lkey;
  auto status =
      receive ? PostReceive(resources.qp, &sge) : ReturnStatus::kSuccess;
  if (status == ReturnStatus::kSuccess) {
    status = ExchangeReceiveReady(resources.socket, bytes, receive);
  }
  if (status == ReturnStatus::kSuccess && !receive) {
    status = PostSend(resources.qp, &sge);
  }
  if (status == ReturnStatus::kSuccess) {
    status = WaitForRdmaCompletion(
        receive ? resources.recv_cq : resources.send_cq, receive, bytes);
  }
  return status;
}

ReturnStatus ReleaseMemoryRegion(ReturnStatus status, RdmaResources &resources,
                                 RdmaMemoryRegion *region) {
  if (status != ReturnStatus::kSuccess) {
    // Stop outstanding DMA before releasing the registered buffer.
    DestroyQueuePair(resources);
  }
  const auto cleanup = region->Reset();
  return cleanup == ReturnStatus::kSuccess ? status : cleanup;
}

ReturnStatus PrepareTransferMemory(Device::Type device, void *buffer,
                                   size_t bytes, void *stream,
                                   const RdmaResources &resources,
                                   std::unique_ptr<RdmaMemoryAdapter> *adapter,
                                   RdmaMemoryRegion *region) {
  auto status = CreateRdmaMemoryAdapter(device, adapter);
  if (status == ReturnStatus::kSuccess) {
    status = (*adapter)->Prepare(buffer, bytes, stream);
  }
  if (status == ReturnStatus::kSuccess) {
    status = (*adapter)->Register(resources.pd, buffer, bytes, region);
  }
  return status;
}

}  // namespace

RdmaInstance::RdmaInstance() : resources_(std::make_unique<RdmaResources>()) {
  type = BackendType::kRdma;
}

RdmaInstance::~RdmaInstance() { Destroy(); }

ReturnStatus RdmaInstance::Connect(const RdmaUniqueId &id, int rank) {
  auto &resources = *resources_;
  resources.rank = rank;
  auto status = OpenDevice(resources);
  if (status == ReturnStatus::kSuccess) {
    resources.socket = OpenBootstrap(id, rank == 0);
    if (resources.socket < 0) {
      std::cerr << "[Rank " << rank << "] TCP bootstrap failed at "
                << id.address << ':' << id.port << ".\n";
      status = ReturnStatus::kSystemError;
    }
  }
  if (status == ReturnStatus::kSuccess) {
    status = ConnectQueuePair(id, resources);
  }
  if (status != ReturnStatus::kSuccess) {
    LOG("Failed to initialize the TCP bootstrap or verbs RC connection.");
    Destroy();
  } else {
    std::cout << "[Rank " << rank
              << "] Verbs RC QP ready. Maximum message: " << resources.max_bytes
              << " bytes." << std::endl;
  }
  return status;
}

ReturnStatus RdmaInstance::Transfer(void *buffer, size_t bytes, int peer,
                                    bool receive, Device::Type device,
                                    void *stream) {
  auto &resources = *resources_;
  std::lock_guard<std::mutex> lock(resources.mutex);
  if (!resources.qp || resources.socket < 0) {
    return ReturnStatus::kInvalidUsage;
  }

  // TODO: Temporarily supports 2 ranks only.
  if (peer != 1 - resources.rank) {
    return ReturnStatus::kInvalidArgument;
  }

  if (bytes > resources.max_bytes) {
    LOG("The initial RDMA backend requires messages no larger than 1 MiB "
        "and the negotiated port limit.");
    Destroy();
    return ReturnStatus::kNotSupported;
  }
  std::unique_ptr<RdmaMemoryAdapter> adapter;
  RdmaMemoryRegion region;
  auto status = PrepareTransferMemory(device, buffer, bytes, stream, resources,
                                      &adapter, &region);
  if (status == ReturnStatus::kSuccess) {
    status =
        TransferRegisteredBuffer(resources, buffer, bytes, region.mr, receive);
  }
  if (status == ReturnStatus::kSuccess) {
    status = adapter->Complete(receive);
  }
  status = ReleaseMemoryRegion(status, resources, &region);
  if (status != ReturnStatus::kSuccess) {
    Destroy();
  }
  return status;
}

void RdmaInstance::Destroy() {
  auto &resources = *resources_;
  DestroyQueuePair(resources);
  if (resources.send_cq) {
    ibv_destroy_cq(resources.send_cq);
  }
  if (resources.recv_cq) {
    ibv_destroy_cq(resources.recv_cq);
  }
  if (resources.pd) {
    ibv_dealloc_pd(resources.pd);
  }
  if (resources.context) {
    ibv_close_device(resources.context);
  }
  if (resources.socket >= 0) {
    close(resources.socket);
  }
  resources.send_cq = nullptr;
  resources.recv_cq = nullptr;
  resources.pd = nullptr;
  resources.context = nullptr;
  resources.socket = -1;
}

}  // namespace infini::ccl
