/**
 * InfiniCCL Example/Test: Broadcast
 *
 * Runs a small suite of boundary cases:
 *   1. count = 0 → no-op success
 *   2. out-of-place, root = size - 1
 *   3. out-of-place, non-root sendbuff = nullptr (documented contract)
 *   4. in-place (sendbuff == recvbuff), root = 0
 *   5. in-place (sendbuff == recvbuff), root = size - 1
 *   6. count > INT_MAX bytes (chunking path), gated by INFINI_BROADCAST_LARGE=1
 *   7. invalid root (-1 and size) → infiniInvalidArgument
 */

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "infiniccl.h"
#include "utils.h"

#include "backend_manifest.h"
#include "device.h"
#include "runtime.h"
#include "traits.h"

using namespace infini::ccl;

namespace {

struct CaseResult {
  bool ok = true;
  bool skipped = false;
  std::string note;
};

bool AllRanksOk(bool local_ok) {
  int local = local_ok ? 1 : 0;
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
  return global != 0;
}

void PrintCase(int rank, const std::string &name, const CaseResult &local,
               bool global_ok) {
  if (rank != 0) {
    return;
  }
  const char *GREEN = "\033[32m";
  const char *YELLOW = "\033[33m";
  const char *RED = "\033[31m";
  const char *RESET = "\033[0m";

  std::string status;
  if (local.skipped) {
    status = std::string(YELLOW) + "SKIP" + RESET;
  } else if (global_ok) {
    status = std::string(GREEN) + "PASS" + RESET;
  } else {
    status = std::string(RED) + "FAIL" + RESET;
  }

  std::cout << "[" << name << "] " << status;
  if (!local.note.empty()) {
    std::cout << " (rank0: " << local.note << ")";
  }
  std::cout << std::endl;
}

// Broadcasts `count` floats from `root` and verifies every rank receives
// `expected`. All ranks must call with the same `root`, `count`, and `inplace`.
//
// Out-of-place: root passes a separate sendbuff; non-root passes `nullptr`.
// In-place:     every rank passes `sendbuff == recvbuff`; root pre-fills the
//               recv buffer with the source data, non-root with garbage that
//               must be overwritten by the broadcast.
template <Device::Type kDev>
CaseResult RunBasicFloat32(infiniComm_t comm, int rank, int root, size_t count,
                           float expected, bool inplace) {
  using Rt = Runtime<kDev>;
  const size_t total_bytes = count * sizeof(float);

  std::vector<float> h_init(count, expected);
  std::vector<float> h_garbage(count, -1.0f);

  float *d_recv = nullptr;
  float *d_send_owned =
      nullptr; // separate allocation, only for out-of-place root

  CHECK_RT(Rt, Rt::Malloc(&d_recv, total_bytes));

  // Pre-fill the receive buffer.
  // - In-place root: recv holds the source data (and stays unchanged).
  // - Everyone else: garbage that must be overwritten by the broadcast.
  const bool recv_holds_source = inplace && (rank == root);
  CHECK_RT(Rt, Rt::Memcpy(d_recv,
                          recv_holds_source ? h_init.data() : h_garbage.data(),
                          total_bytes, Rt::MemcpyHostToDevice));

  // Out-of-place root needs a separate sendbuff carrying the source data.
  if (!inplace && rank == root) {
    CHECK_RT(Rt, Rt::Malloc(&d_send_owned, total_bytes));
    CHECK_RT(Rt, Rt::Memcpy(d_send_owned, h_init.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
  }

  // Resolve the send pointer per mode.
  const void *send_ptr = nullptr;
  if (inplace) {
    send_ptr = d_recv; // every rank passes the same pointer as recv
  } else if (rank == root) {
    send_ptr = d_send_owned;
  }
  // Out-of-place non-root: send_ptr stays nullptr (documented contract).

  infiniResult_t status = infiniBroadcast(send_ptr, d_recv, count,
                                          infiniFloat32, root, comm, nullptr);

  CaseResult result;
  if (status != infiniSuccess) {
    result.ok = false;
    result.note =
        "infiniBroadcast returned " + std::to_string(static_cast<int>(status));
  } else {
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    std::vector<float> h_recv(count);
    CHECK_RT(Rt, Rt::Memcpy(h_recv.data(), d_recv, total_bytes,
                            Rt::MemcpyDeviceToHost));

    for (size_t i = 0; i < count; ++i) {
      if (std::fabs(h_recv[i] - expected) > 1e-3) {
        result.ok = false;
        result.note = "value mismatch at index " + std::to_string(i);
        break;
      }
    }
  }

  if (d_send_owned) {
    CHECK_RT(Rt, Rt::Free(d_send_owned));
  }
  CHECK_RT(Rt, Rt::Free(d_recv));
  return result;
}

CaseResult Case_Count0(infiniComm_t comm) {
  // All ranks pass nullptrs; the impl must short-circuit before any buffer
  // access.
  infiniResult_t status =
      infiniBroadcast(nullptr, nullptr, 0, infiniFloat32, 0, comm, nullptr);
  if (status != infiniSuccess) {
    return {false, false,
            "expected infiniSuccess, got " +
                std::to_string(static_cast<int>(status))};
  }
  return {};
}

template <Device::Type kDev>
CaseResult Case_OutOfPlaceRootLast(infiniComm_t comm, int rank, int size) {
  return RunBasicFloat32<kDev>(comm, rank, /*root=*/size - 1, /*count=*/1024,
                               /*expected=*/7.5f, /*inplace=*/false);
}

template <Device::Type kDev>
CaseResult Case_OutOfPlaceNonRootNullSend(infiniComm_t comm, int rank) {
  // Out-of-place mode passes nullptr as sendbuff on non-root ranks. This case
  // locks that contract in as an explicit, named check.
  return RunBasicFloat32<kDev>(comm, rank, /*root=*/0, /*count=*/2048,
                               /*expected=*/-3.25f, /*inplace=*/false);
}

template <Device::Type kDev>
CaseResult Case_InplaceRootZero(infiniComm_t comm, int rank) {
  // sendbuff == recvbuff on every rank; root's value must survive, non-root
  // must be overwritten.
  return RunBasicFloat32<kDev>(comm, rank, /*root=*/0, /*count=*/1024,
                               /*expected=*/11.25f, /*inplace=*/true);
}

template <Device::Type kDev>
CaseResult Case_InplaceRootLast(infiniComm_t comm, int rank, int size) {
  return RunBasicFloat32<kDev>(comm, rank, /*root=*/size - 1, /*count=*/1024,
                               /*expected=*/-42.5f, /*inplace=*/true);
}

template <Device::Type kDev>
CaseResult Case_LargeCount(infiniComm_t comm, int rank) {
  using Rt = Runtime<kDev>;
  if (std::getenv("INFINI_BROADCAST_LARGE") == nullptr) {
    return {true, true, "set INFINI_BROADCAST_LARGE=1 to enable (~2GB/rank)"};
  }
  // Force the chunked MPI_Bcast path: byte count > INT_MAX.
  const size_t count = static_cast<size_t>(std::numeric_limits<int>::max()) +
                       static_cast<size_t>(1024);
  const std::int8_t expected = 0x5A;
  const size_t total_bytes = count * sizeof(std::int8_t);

  std::vector<std::int8_t> h_send;
  if (rank == 0) {
    h_send.assign(count, expected);
  }

  std::int8_t *d_send = nullptr;
  std::int8_t *d_recv = nullptr;
  if (rank == 0) {
    CHECK_RT(Rt, Rt::Malloc(&d_send, total_bytes));
    CHECK_RT(Rt, Rt::Memcpy(d_send, h_send.data(), total_bytes,
                            Rt::MemcpyHostToDevice));
  }
  CHECK_RT(Rt, Rt::Malloc(&d_recv, total_bytes));

  infiniResult_t status =
      infiniBroadcast(rank == 0 ? d_send : nullptr, d_recv, count, infiniChar,
                      /*root=*/0, comm, nullptr);

  CaseResult result;
  if (status != infiniSuccess) {
    result.ok = false;
    result.note =
        "infiniBroadcast returned " + std::to_string(static_cast<int>(status));
  } else {
    CHECK_RT(Rt, Rt::StreamSynchronize(nullptr));
    // Sample head, middle, and tail to avoid scanning ~2GB.
    std::int8_t probes[3] = {-1, -1, -1};
    CHECK_RT(Rt, Rt::Memcpy(&probes[0], d_recv, sizeof(std::int8_t),
                            Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::Memcpy(&probes[1], d_recv + (count / 2),
                            sizeof(std::int8_t), Rt::MemcpyDeviceToHost));
    CHECK_RT(Rt, Rt::Memcpy(&probes[2], d_recv + (count - 1),
                            sizeof(std::int8_t), Rt::MemcpyDeviceToHost));
    if (probes[0] != expected || probes[1] != expected ||
        probes[2] != expected) {
      result.ok = false;
      result.note = "head/mid/tail mismatch";
    }
  }

  if (d_send) {
    CHECK_RT(Rt, Rt::Free(d_send));
  }
  CHECK_RT(Rt, Rt::Free(d_recv));
  return result;
}

CaseResult Case_InvalidRoot(infiniComm_t comm, int size) {
  // Tiny dummy buffers — the validator must reject `root` before touching
  // them. Passing `count=1` with valid datatype ensures no other early exit
  // (count=0, dtype) preempts the root check.
  float dummy_send = 0.f;
  float dummy_recv = 0.f;

  for (int bad_root : {-1, size}) {
    infiniResult_t status = infiniBroadcast(
        &dummy_send, &dummy_recv, 1, infiniFloat32, bad_root, comm, nullptr);
    if (status != infiniInvalidArgument) {
      return {false, false,
              "root=" + std::to_string(bad_root) + " expected " +
                  std::to_string(static_cast<int>(infiniInvalidArgument)) +
                  ", got " + std::to_string(static_cast<int>(status))};
    }
  }
  return {};
}

} // namespace

int main(int argc, char **argv) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});

  CHECK_INFINI(infiniInit(&argc, &argv));

  int rank = 0;
  int size = 0;
  CHECK_INFINI(infiniGetRank(&rank));
  CHECK_INFINI(infiniGetSize(&size));

  if (rank == 0) {
    std::cout << "=== Broadcast Test Suite ===" << std::endl;
    std::cout << "Device: " << Device::StringFromType(kDevType) << std::endl;
    std::cout << "Ranks:  " << size << std::endl;
  }

  infiniComm_t comm = nullptr;
  CHECK_INFINI(infiniCommInitAll(&comm, size, nullptr));

  bool overall_ok = true;

  auto run = [&](const std::string &name, CaseResult local) {
    bool global_ok = AllRanksOk(local.ok);
    PrintCase(rank, name, local, global_ok);
    if (!local.skipped) {
      overall_ok = overall_ok && global_ok;
    }
  };

  run("count=0", Case_Count0(comm));
  run("out-of-place, root=size-1",
      Case_OutOfPlaceRootLast<kDevType>(comm, rank, size));
  run("out-of-place, non-root sendbuff=nullptr",
      Case_OutOfPlaceNonRootNullSend<kDevType>(comm, rank));
  run("in-place, root=0", Case_InplaceRootZero<kDevType>(comm, rank));
  run("in-place, root=size-1",
      Case_InplaceRootLast<kDevType>(comm, rank, size));
  run("large count (>INT_MAX bytes)", Case_LargeCount<kDevType>(comm, rank));
  run("invalid root", Case_InvalidRoot(comm, size));

  if (rank == 0) {
    const char *GREEN = "\033[32m";
    const char *RED = "\033[31m";
    const char *RESET = "\033[0m";
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << (overall_ok ? (std::string(GREEN) + "ALL PASS" + RESET)
                             : (std::string(RED) + "FAILED" + RESET))
              << std::endl;
  }

  CHECK_INFINI(infiniCommDestroy(comm));
  CHECK_INFINI(infiniFinalize());

  return overall_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}