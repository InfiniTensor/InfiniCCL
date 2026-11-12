#include "backends/mpi/rdma/rdma_transport.h"

#include <arpa/inet.h>
#include <infiniband/verbs.h>
#include <sys/random.h>
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
#include <thread>

#include "backends/mpi/rdma/memory_adaptor.h"
#include "logging.h"

namespace infini::ccl {

namespace {

constexpr size_t kGroupIdBytes = 16;

// Control frames ride on a dedicated MPI tag so they never clash with the
// staged payload messages of the MPI backends (tag 0).
constexpr int kControlTag = 777;

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

  char group_id[kGroupIdBytes];
};

static_assert(sizeof(PeerMetadata) == 8 * sizeof(uint32_t) + 16 + kGroupIdBytes,
              "`PeerMetadata` must not contain padding.");

enum class StagedReason : size_t {
  kNoLink = 0,
  kNoAdapter,
  kBuffer,
  kSizeLimit,
  kPeerRejected,
  kCount
};

enum class ControlKind : uint8_t { kStaged, kPropose, kAccept, kReject };

struct ControlFrame {
  ControlKind kind;
};

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

ReturnStatus SelectDevice(ibv_context **context, ibv_port_attr *port_attr,
                          int port) {
  int count = 0;
  ibv_device **devices = ibv_get_device_list(&count);
  if (!devices) {
    return ReportVerbsError("ibv_get_device_list", errno);
  }
  const char *name = std::getenv("INFINICCL_RDMA_DEVICE");
  ibv_context *found = nullptr;
  for (int i = 0; i < count; ++i) {
    if (name && std::strcmp(name, ibv_get_device_name(devices[i])) != 0) {
      continue;
    }
    ibv_context *opened = ibv_open_device(devices[i]);
    if (!opened) {
      continue;
    }
    if (ibv_query_port(opened, port, port_attr) == 0 &&
        port_attr->state == IBV_PORT_ACTIVE) {
      found = opened;
      break;
    }
    ibv_close_device(opened);
  }
  ibv_free_device_list(devices);
  if (!found) {
    LOG("No selected RDMA device has an active port.");
    return ReturnStatus::kSystemError;
  }
  *context = found;
  return ReturnStatus::kSuccess;
}

#if defined(INFINICCL_WITH_MLX5DV)
ReturnStatus CreateMlx5QueuePair(const ibv_qp_init_attr &base, ibv_pd *pd,
                                 ibv_context *context, ibv_qp **qp) {
  ibv_qp_init_attr_ex attr{};
  attr.qp_context = base.qp_context;
  attr.send_cq = base.send_cq;
  attr.recv_cq = base.recv_cq;
  attr.srq = base.srq;
  attr.cap = base.cap;
  attr.qp_type = base.qp_type;
  attr.sq_sig_all = base.sq_sig_all;
  attr.comp_mask = IBV_QP_INIT_ATTR_PD;
  attr.pd = pd;
  mlx5dv_qp_init_attr mlx5_attr{};
  mlx5_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS;
  mlx5_attr.create_flags = MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE;
  *qp = mlx5dv_create_qp(context, &attr, &mlx5_attr);
  if (!*qp) {
    return ReportVerbsError("mlx5dv_create_qp(disable Scatter-to-CQE)", errno);
  }
  return ReturnStatus::kSuccess;
}
#endif

ReturnStatus CreateCompatibleQueuePair(const ibv_qp_init_attr &attr, ibv_pd *pd,
                                       ibv_context *context, ibv_qp **qp) {
#if defined(INFINICCL_WITH_MLX5DV)
  if (mlx5dv_is_supported(context->device)) {
    const auto status = CreateMlx5QueuePair(attr, pd, context, qp);
    if (status == ReturnStatus::kSuccess) {
      std::cout << "mlx5 Scatter-to-CQE disabled explicitly for this QP."
                << std::endl;
    }
    return status;
  }
#endif
  ibv_device_attr device_attr{};
  const int query = ibv_query_device(context, &device_attr);
  if (query != 0) {
    return ReportVerbsError("ibv_query_device", query);
  }
  const char *name = ibv_get_device_name(context->device);
  if (device_attr.vendor_id == 0x02c9 || std::strncmp(name, "mlx5", 4) == 0) {
    LOG("Mellanox QP creation requires mlx5dv Scatter-to-CQE control. "
        "Install compatible mlx5 development headers/libraries and rebuild.");
    return ReturnStatus::kNotSupported;
  }
  *qp = ibv_create_qp(pd, const_cast<ibv_qp_init_attr *>(&attr));
  if (!*qp) {
    return ReportVerbsError("ibv_create_qp", errno);
  }
  return ReturnStatus::kSuccess;
}

bool SendControl(MPI_Comm comm, int peer, ControlKind kind) {
  const ControlFrame frame{kind};
  return MPI_Send(&frame, sizeof(frame), MPI_UINT8_T, peer, kControlTag,
                  comm) == MPI_SUCCESS;
}

bool RecvControl(MPI_Comm comm, int peer, ControlKind *kind) {
  ControlFrame frame{};
  if (MPI_Recv(&frame, sizeof(frame), MPI_UINT8_T, peer, kControlTag, comm,
               MPI_STATUS_IGNORE) != MPI_SUCCESS) {
    return false;
  }
  switch (frame.kind) {
    case ControlKind::kStaged:
    case ControlKind::kPropose:
    case ControlKind::kAccept:
    case ControlKind::kReject:
      *kind = frame.kind;
      return true;
  }
  return false;
}

// Both ranks derive the same random group identifier without a distinguished
// root, so the collective cannot deadlock on a single-rank failure.
void DestroyQueuePair(ibv_qp *qp) {
  if (qp) {
    const int status = ibv_destroy_qp(qp);
    if (status != 0) {
      std::cerr << __FILE__ << ':' << __LINE__ << " " << __func__
                << ": Failed to destroy the RDMA QP safely.\n";
      assert(status == 0 && "Failed to destroy the RDMA QP safely.");
      std::abort();
    }
  }
}

ReturnStatus PostReceive(ibv_qp *qp, void *buffer, size_t bytes, ibv_mr *mr) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(buffer);
  sge.length = static_cast<uint32_t>(bytes);
  sge.lkey = mr->lkey;
  ibv_recv_wr wr{};
  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  ibv_recv_wr *bad = nullptr;
  const int status = ibv_post_recv(qp, &wr, &bad);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_post_recv", status);
}

ReturnStatus PostSend(ibv_qp *qp, void *buffer, size_t bytes, ibv_mr *mr) {
  ibv_sge sge{};
  sge.addr = reinterpret_cast<uintptr_t>(buffer);
  sge.length = static_cast<uint32_t>(bytes);
  sge.lkey = mr->lkey;
  ibv_send_wr wr{};
  wr.wr_id = 1;
  wr.sg_list = &sge;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_SEND;
  wr.send_flags = IBV_SEND_SIGNALED;
  ibv_send_wr *bad = nullptr;
  const int status = ibv_post_send(qp, &wr, &bad);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_post_send", status);
}

}  // namespace

struct RdmaTransport::Impl {
  ibv_context *context = nullptr;

  ibv_pd *pd = nullptr;

  ibv_cq *send_cq = nullptr;

  ibv_cq *recv_cq = nullptr;

  ibv_qp *qp = nullptr;

  ibv_port_attr port_attr{};

  ibv_gid gid{};

  MPI_Comm comm = MPI_COMM_NULL;

  MrCache mr_cache;

  std::mutex mutex;

  std::unique_ptr<RdmaMemoryAdapter> adapter;

  ibv_mr *pending_mr = nullptr;

  ControlKind proposal = ControlKind::kStaged;

  uint8_t group_id[kGroupIdBytes] = {};

  int rank = -1;

  int port = 1;

  int gid_index = -1;

  int pkey_index = 0;

  size_t max_bytes = 0;

  bool active = false;

  uint64_t rdma_ops = 0;

  uint64_t staged_ops = 0;

  uint64_t fallbacks[static_cast<size_t>(StagedReason::kCount)] = {};

  void RecordFallback(StagedReason reason) {
    ++staged_ops;
    ++fallbacks[static_cast<size_t>(reason)];
  }

  bool RejectSend(StagedReason reason);

  bool RejectReceive(StagedReason reason);

  bool PrepareSendPath(void *buffer, size_t bytes, Device::Type device);

  bool PrepareReceivePath(void *buffer, size_t bytes, Device::Type device,
                          void *stream);

  void DestroyLocked();
};

namespace {

// Prepares every rank-local verbs resource. All fallible setup happens before
// the availability agreement so a rank without RDMA never leaves its peers
// waiting inside a collective.
ReturnStatus ValidatePort(RdmaTransport::Impl &impl, bool *valid) {
  if (impl.port_attr.link_layer == IBV_LINK_LAYER_ETHERNET &&
      impl.gid_index < 0) {
    LOG("RoCE requires an explicit `INFINICCL_RDMA_GID_INDEX`.");
    return ReturnStatus::kInvalidArgument;
  }
  if ((impl.port_attr.flags & IBV_QPF_GRH_REQUIRED) && impl.gid_index < 0) {
    LOG("This port requires a global route and an explicit GID index.");
    return ReturnStatus::kNotSupported;
  }
  if (impl.pkey_index >= impl.port_attr.pkey_tbl_len) {
    return ReturnStatus::kInvalidArgument;
  }
  if (impl.gid_index >= 0 && (impl.gid_index >= impl.port_attr.gid_tbl_len ||
                              ibv_query_gid(impl.context, impl.port,
                                            impl.gid_index, &impl.gid) != 0)) {
    return ReturnStatus::kInvalidArgument;
  }
  *valid = true;
  return ReturnStatus::kSuccess;
}

void GenerateGroupId(RdmaTransport::Impl &impl) {
  uint64_t local[kGroupIdBytes / sizeof(uint64_t)] = {};
  size_t offset = 0;
  while (offset < sizeof(local)) {
    const ssize_t count = getrandom(reinterpret_cast<uint8_t *>(local) + offset,
                                    sizeof(local) - offset, 0);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      local[0] ^= static_cast<uint64_t>(
          std::chrono::steady_clock::now().time_since_epoch().count());
      local[1] ^= static_cast<uint64_t>(getpid()) << 32;
      break;
    }
    offset += static_cast<size_t>(count);
  }
  uint64_t shared[kGroupIdBytes / sizeof(uint64_t)] = {};
  (void)MPI_Allreduce(local, shared, 2, MPI_UINT64_T, MPI_BXOR, impl.comm);
  std::memcpy(impl.group_id, shared, kGroupIdBytes);
}

ReturnStatus BuildPeerMetadata(const RdmaTransport::Impl &impl,
                               PeerMetadata *local) {
  uint16_t pkey = 0;
  if (ibv_query_pkey(impl.context, impl.port, impl.pkey_index, &pkey) != 0 ||
      (ntohs(pkey) & 0x7fff) == 0) {
    return ReturnStatus::kInvalidArgument;
  }
  local->rank = htonl(impl.rank);
  local->qp_num = htonl(impl.qp->qp_num);
  local->lid = htonl(impl.port_attr.lid);
  local->mtu = htonl(impl.port_attr.active_mtu);
  local->link_layer = htonl(impl.port_attr.link_layer);
  local->has_gid = htonl(impl.gid_index >= 0);
  local->max_bytes = htonl(impl.max_bytes);
  local->pkey = htonl(ntohs(pkey));
  std::memcpy(local->gid, impl.gid.raw, sizeof(local->gid));
  std::memcpy(local->group_id, impl.group_id, sizeof(local->group_id));
  return ReturnStatus::kSuccess;
}

ReturnStatus ValidatePeerMetadata(const RdmaTransport::Impl &impl,
                                  const PeerMetadata &local,
                                  const PeerMetadata &remote) {
  if (ntohl(remote.rank) != static_cast<uint32_t>(1 - impl.rank) ||
      remote.link_layer != local.link_layer ||
      remote.has_gid != local.has_gid ||
      (ntohl(remote.pkey) & 0x7fff) != (ntohl(local.pkey) & 0x7fff) ||
      std::memcmp(remote.group_id, local.group_id, kGroupIdBytes) != 0 ||
      ntohl(remote.qp_num) == 0 || ntohl(remote.qp_num) > 0xffffff ||
      ntohl(remote.lid) > 65535 ||
      (impl.gid_index < 0 && ntohl(remote.lid) == 0) ||
      ntohl(remote.mtu) < IBV_MTU_256 || ntohl(remote.mtu) > IBV_MTU_4096 ||
      ntohl(remote.max_bytes) == 0) {
    LOG("Incompatible RDMA peer metadata or communication group.");
    return ReturnStatus::kRemoteError;
  }
  return ReturnStatus::kSuccess;
}

ReturnStatus InitializeQueuePair(const RdmaTransport::Impl &impl) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_INIT;
  attr.port_num = impl.port;
  attr.pkey_index = impl.pkey_index;
  const int status = ibv_modify_qp(
      impl.qp, &attr,
      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_modify_qp(INIT)", status);
}

ReturnStatus ReadyToReceive(const RdmaTransport::Impl &impl,
                            const PeerMetadata &remote) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTR;
  attr.path_mtu = static_cast<ibv_mtu>(
      std::min<uint32_t>(impl.port_attr.active_mtu, ntohl(remote.mtu)));
  attr.dest_qp_num = ntohl(remote.qp_num);
  attr.rq_psn = attr.dest_qp_num & 0xffffff;
  attr.max_dest_rd_atomic = 1;
  attr.min_rnr_timer = 12;
  attr.ah_attr.dlid = ntohl(remote.lid);
  attr.ah_attr.port_num = impl.port;
  if (impl.gid_index >= 0) {
    attr.ah_attr.is_global = 1;
    std::memcpy(attr.ah_attr.grh.dgid.raw, remote.gid,
                sizeof(attr.ah_attr.grh.dgid.raw));
    attr.ah_attr.grh.sgid_index = impl.gid_index;
    attr.ah_attr.grh.hop_limit = 64;
  }
  const int status = ibv_modify_qp(
      impl.qp, &attr,
      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
          IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_modify_qp(RTR)", status);
}

ReturnStatus ReadyToSend(const RdmaTransport::Impl &impl) {
  ibv_qp_attr attr{};
  attr.qp_state = IBV_QPS_RTS;
  attr.sq_psn = impl.qp->qp_num & 0xffffff;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;
  attr.max_rd_atomic = 1;
  const int status = ibv_modify_qp(
      impl.qp, &attr,
      IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT |
          IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC);
  return status == 0 ? ReturnStatus::kSuccess
                     : ReportVerbsError("ibv_modify_qp(RTS)", status);
}

bool PrepareLocalResources(RdmaTransport::Impl &impl) {
  if (!ReadOption("INFINICCL_RDMA_IB_PORT", 1, 1, 255, &impl.port) ||
      !ReadOption("INFINICCL_RDMA_GID_INDEX", -1, -1, 255, &impl.gid_index) ||
      !ReadOption("INFINICCL_RDMA_PKEY_INDEX", 0, 0, 65535, &impl.pkey_index)) {
    return false;
  }
  if (SelectDevice(&impl.context, &impl.port_attr, impl.port) !=
      ReturnStatus::kSuccess) {
    impl.context = nullptr;
    return false;
  }
  bool valid_port = false;
  const auto port_status = ValidatePort(impl, &valid_port);
  if (port_status != ReturnStatus::kSuccess || !valid_port) {
    ibv_close_device(impl.context);
    impl.context = nullptr;
    return false;
  }
  impl.pd = ibv_alloc_pd(impl.context);
  impl.send_cq = ibv_create_cq(impl.context, 16, nullptr, nullptr, 0);
  impl.recv_cq = ibv_create_cq(impl.context, 16, nullptr, nullptr, 0);
  ibv_qp_init_attr attr{};
  attr.send_cq = impl.send_cq;
  attr.recv_cq = impl.recv_cq;
  attr.cap.max_send_wr = 16;
  attr.cap.max_recv_wr = 16;
  attr.cap.max_send_sge = 1;
  attr.cap.max_recv_sge = 1;
  attr.qp_type = IBV_QPT_RC;
  const auto qp_status =
      CreateCompatibleQueuePair(attr, impl.pd, impl.context, &impl.qp);
  if (!impl.pd || !impl.send_cq || !impl.recv_cq ||
      qp_status != ReturnStatus::kSuccess) {
    return false;
  }
  impl.max_bytes = std::min<size_t>(impl.port_attr.max_msg_sz, 1 << 20);
  return true;
}

// Runs the collective connection phase. Every rank reaches the final
// agreement reduction regardless of intermediate failures, so asymmetric
// verbs errors degrade to host staging instead of deadlocking.
ReturnStatus ConnectPeer(RdmaTransport::Impl &impl) {
  auto status = ReturnStatus::kSuccess;
  GenerateGroupId(impl);
  PeerMetadata local{};
  PeerMetadata remote{};
  status = BuildPeerMetadata(impl, &local);
  if (status == ReturnStatus::kSuccess &&
      MPI_Sendrecv(&local, sizeof(local), MPI_BYTE, 1 - impl.rank, kControlTag,
                   &remote, sizeof(remote), MPI_BYTE, 1 - impl.rank,
                   kControlTag, impl.comm, MPI_STATUS_IGNORE) != MPI_SUCCESS) {
    LOG("Failed to exchange RDMA peer metadata over MPI.");
    status = ReturnStatus::kSystemError;
  }
  if (status == ReturnStatus::kSuccess) {
    status = ValidatePeerMetadata(impl, local, remote);
  }
  if (status == ReturnStatus::kSuccess) {
    impl.max_bytes = std::min<size_t>(impl.max_bytes, ntohl(remote.max_bytes));
    status = InitializeQueuePair(impl);
  }
  if (status == ReturnStatus::kSuccess) {
    status = ReadyToReceive(impl, remote);
  }
  if (status == ReturnStatus::kSuccess) {
    status = ReadyToSend(impl);
  }
  int ready = status == ReturnStatus::kSuccess ? 1 : 0;
  int all_ready = 0;
  if (MPI_Allreduce(&ready, &all_ready, 1, MPI_INT, MPI_MIN, impl.comm) !=
          MPI_SUCCESS ||
      all_ready != 1) {
    return ReturnStatus::kSystemError;
  }
  return status;
}

}  // namespace

bool RdmaTransport::Impl::RejectSend(StagedReason reason) {
  RecordFallback(reason);
  // The peer waits for exactly one control frame per message.
  (void)SendControl(comm, 1 - rank, ControlKind::kStaged);
  return false;
}

bool RdmaTransport::Impl::RejectReceive(StagedReason reason) {
  RecordFallback(reason);
  (void)SendControl(comm, 1 - rank, ControlKind::kReject);
  return false;
}

bool RdmaTransport::Impl::PrepareSendPath(void *buffer, size_t bytes,
                                          Device::Type device) {
  if (bytes == 0 || bytes > max_bytes) {
    return RejectSend(StagedReason::kSizeLimit);
  }
  if (CreateRdmaMemoryAdapter(device, &adapter) != ReturnStatus::kSuccess) {
    adapter.reset();
    return RejectSend(StagedReason::kNoAdapter);
  }
  if (!adapter->CanRegister(buffer, bytes)) {
    adapter.reset();
    return RejectSend(StagedReason::kBuffer);
  }
  if (!SendControl(comm, 1 - rank, ControlKind::kPropose) ||
      !RecvControl(comm, 1 - rank, &proposal)) {
    RecordFallback(StagedReason::kNoLink);
    adapter.reset();
    return false;
  }
  if (proposal != ControlKind::kAccept) {
    RecordFallback(StagedReason::kPeerRejected);
    adapter.reset();
    return false;
  }
  return true;
}

bool RdmaTransport::Impl::PrepareReceivePath(void *buffer, size_t bytes,
                                             Device::Type device,
                                             void *stream) {
  if (!RecvControl(comm, 1 - rank, &proposal)) {
    RecordFallback(StagedReason::kNoLink);
    return false;
  }
  if (proposal == ControlKind::kStaged) {
    // The peer already chose host staging; mirror it without a reply.
    ++staged_ops;
    return false;
  }
  if (bytes == 0 || bytes > max_bytes) {
    return RejectReceive(StagedReason::kSizeLimit);
  }
  if (CreateRdmaMemoryAdapter(device, &adapter) != ReturnStatus::kSuccess) {
    adapter.reset();
    return RejectReceive(StagedReason::kNoAdapter);
  }
  if (!adapter->CanRegister(buffer, bytes)) {
    adapter.reset();
    return RejectReceive(StagedReason::kBuffer);
  }
  auto status = adapter->Prepare(buffer, bytes, stream);
  if (status == ReturnStatus::kSuccess) {
    status = mr_cache.Acquire(pd, adapter.get(), buffer, bytes, &pending_mr);
  }
  if (status == ReturnStatus::kSuccess) {
    status = PostReceive(qp, buffer, bytes, pending_mr);
  }
  if (status != ReturnStatus::kSuccess) {
    adapter.reset();
    pending_mr = nullptr;
    return RejectReceive(StagedReason::kBuffer);
  }
  // The receive work request is posted; the confirmation is the point of no
  // return after which failures surface as errors instead of fallbacks.
  if (!SendControl(comm, 1 - rank, ControlKind::kAccept)) {
    RecordFallback(StagedReason::kNoLink);
    adapter.reset();
    pending_mr = nullptr;
    return false;
  }
  return true;
}

void RdmaTransport::Impl::DestroyLocked() {
  DestroyQueuePair(qp);
  qp = nullptr;
  adapter.reset();
  pending_mr = nullptr;
  const auto cache_stats = mr_cache.Stats();
  mr_cache.Reset();
  if (cache_stats.acquisitions > 0) {
    std::cout << "[Rank " << rank
              << "] RDMA MR cache: " << cache_stats.registrations
              << " registrations for " << cache_stats.acquisitions
              << " transfers." << std::endl;
  }
  if (rdma_ops > 0 || staged_ops > 0) {
    std::cout << "[Rank " << rank << "] RDMA transport: rdma=" << rdma_ops
              << " staged=" << staged_ops << " (no_link="
              << fallbacks[static_cast<size_t>(StagedReason::kNoLink)]
              << ", no_adapter="
              << fallbacks[static_cast<size_t>(StagedReason::kNoAdapter)]
              << ", buffer="
              << fallbacks[static_cast<size_t>(StagedReason::kBuffer)]
              << ", size="
              << fallbacks[static_cast<size_t>(StagedReason::kSizeLimit)]
              << ", peer_rejected="
              << fallbacks[static_cast<size_t>(StagedReason::kPeerRejected)]
              << ")" << std::endl;
  }
  rdma_ops = 0;
  staged_ops = 0;
  std::memset(fallbacks, 0, sizeof(fallbacks));
  if (send_cq) {
    ibv_destroy_cq(send_cq);
    send_cq = nullptr;
  }
  if (recv_cq) {
    ibv_destroy_cq(recv_cq);
    recv_cq = nullptr;
  }
  if (pd) {
    ibv_dealloc_pd(pd);
    pd = nullptr;
  }
  if (context) {
    ibv_close_device(context);
    context = nullptr;
  }
  active = false;
}

RdmaTransport::RdmaTransport() : impl_(std::make_unique<Impl>()) {}

RdmaTransport::~RdmaTransport() { Destroy(); }

void RdmaTransport::Initialize(int rank, int nranks, MPI_Comm comm) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto &impl = *impl_;
  impl.rank = rank;
  impl.comm = comm;
  if (nranks != 2) {
    // Multi-rank transport links are future work; staging covers them.
    return;
  }
  const bool local_ready = PrepareLocalResources(impl);
  int ready = local_ready ? 1 : 0;
  int all_ready = 0;
  if (MPI_Allreduce(&ready, &all_ready, 1, MPI_INT, MPI_MIN, comm) !=
      MPI_SUCCESS) {
    LOG("Failed to agree on RDMA transport availability.");
    impl.DestroyLocked();
    return;
  }
  if (all_ready != 1) {
    if (local_ready) {
      std::cerr << "[Rank " << rank
                << "] RDMA devices are unavailable on some rank; messages "
                   "use host staging.\n";
    }
    impl.DestroyLocked();
    return;
  }
  const auto status = ConnectPeer(impl);
  if (status != ReturnStatus::kSuccess) {
    std::cerr << "[Rank " << rank
              << "] RDMA transport setup failed; messages use host "
                 "staging.\n";
    impl.DestroyLocked();
    return;
  }
  std::cout << "[Rank " << rank << "] RDMA transport ready on "
            << ibv_get_device_name(impl.context->device) << ", port "
            << impl.port << ", LID " << impl.port_attr.lid << ", GID index "
            << impl.gid_index << "; maximum message " << impl.max_bytes
            << " bytes." << std::endl;
  impl.active = true;
}

bool RdmaTransport::SupportRdma(void *buffer, size_t bytes, int peer,
                                Device::Type device, void *stream, bool send) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->adapter.reset();
  impl_->pending_mr = nullptr;
  if (!impl_->active || peer != 1 - impl_->rank) {
    impl_->RecordFallback(StagedReason::kNoLink);
    return false;
  }
  if (send) {
    return impl_->PrepareSendPath(buffer, bytes, device);
  }
  return impl_->PrepareReceivePath(buffer, bytes, device, stream);
}

ReturnStatus RdmaTransport::Send(void *buffer, size_t bytes, int peer,
                                 void *stream) {
  (void)peer;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto &impl = *impl_;
  auto status = impl.adapter->Prepare(buffer, bytes, stream);
  if (status == ReturnStatus::kSuccess) {
    status = impl.mr_cache.Acquire(impl.pd, impl.adapter.get(), buffer, bytes,
                                   &impl.pending_mr);
  }
  if (status == ReturnStatus::kSuccess) {
    status = PostSend(impl.qp, buffer, bytes, impl.pending_mr);
  }
  if (status == ReturnStatus::kSuccess) {
    status = WaitForRdmaCompletion(impl.send_cq, false, bytes);
  }
  if (status == ReturnStatus::kSuccess) {
    status = impl.adapter->Complete(false);
    ++impl.rdma_ops;
    return status;
  }
  impl.DestroyLocked();
  return status;
}

ReturnStatus RdmaTransport::Recv(void *buffer, size_t bytes, int peer) {
  (void)buffer;
  (void)bytes;
  (void)peer;
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto &impl = *impl_;
  const auto status = WaitForRdmaCompletion(impl.recv_cq, true, 0);
  if (status != ReturnStatus::kSuccess) {
    impl.DestroyLocked();
    return status;
  }
  const auto complete = impl.adapter->Complete(true);
  ++impl.rdma_ops;
  if (complete != ReturnStatus::kSuccess) {
    impl.DestroyLocked();
  }
  return complete;
}

void RdmaTransport::Destroy() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->DestroyLocked();
}

}  // namespace infini::ccl
