/**
 * InfiniCCL Example/Test: Point-to-Point Send/Recv.
 *
 * Runs a small suite of cases covering blocking P2P:
 *   1. count=0 blocking ping (rank 0 -> 1)
 *   2. blocking ping, rank 0 -> 1
 *   3. blocking ping, rank 0 -> size-1
 *   4. blocking ping-pong, rank 0 <-> 1
 *   5. large count (>INT_MAX bytes), gated by INFINI_SENDRECV_LARGE=1
 *   6. invalid peer (-1 and size) -> infiniInvalidArgument
 */

#include <mpi.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "backend_manifest.h"
#include "device.h"
#include "infiniccl.h"
#include "runtime.h"
#include "traits.h"
#include "utils.h"

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

CaseResult SkipNeed2Ranks(int rank) {
  return {true, true, (rank == 0) ? "needs at least 2 ranks" : ""};
}

int GetEnvInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }

  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || *end != '\0') {
    return fallback;
  }

  return static_cast<int>(parsed);
}

int GetLocalRank() {
  int local_rank = GetEnvInt("OMPI_COMM_WORLD_LOCAL_RANK", -1);
  if (local_rank >= 0) {
    return local_rank;
  }

  local_rank = GetEnvInt("MPI_LOCALRANKID", -1);
  if (local_rank >= 0) {
    return local_rank;
  }

  local_rank = GetEnvInt("SLURM_LOCALID", -1);
  if (local_rank >= 0) {
    return local_rank;
  }

  return 0;
}

template <Device::Type kDev>
void PrintRankMapping(int rank, int size) {
  for (int r = 0; r < size; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == r) {
      char host[256] = {};
      if (gethostname(host, sizeof(host)) != 0) {
        std::snprintf(host, sizeof(host), "unknown");
      }

      std::cout << "[Rank " << rank << "] Host: " << host
                << " | GPU: " << Device::StringFromType(kDev) << " | Device "
                << GetLocalRank() << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}

// Allocate a device buffer holding `count` floats == `value`.
template <Device::Type kDev>
float *AllocFilled(size_t count, float value) {
  float *d = nullptr;
  Runtime<kDev>::Malloc(&d, count * sizeof(float));
  std::vector<float> h(count, value);
  Runtime<kDev>::Memcpy(d, h.data(), count * sizeof(float),
                        Runtime<kDev>::MemcpyHostToDevice);
  return d;
}

// Verify `count` floats on device equal `expected`.
template <Device::Type kDev>
bool DeviceEqualsFloat(const float *d, size_t count, float expected) {
  std::vector<float> h(count);
  Runtime<kDev>::Memcpy(h.data(), d, count * sizeof(float),
                        Runtime<kDev>::MemcpyDeviceToHost);
  for (size_t i = 0; i < count; ++i) {
    if (std::fabs(h[i] - expected) > 1e-3) {
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Case 1: count=0 blocking ping
// ---------------------------------------------------------------------------
CaseResult Case_Count0Ping(infiniComm_t comm, int rank, int size) {
  if (size < 2) {
    return SkipNeed2Ranks(rank);
  }
  if (rank == 0) {
    infiniResult_t s = infiniSend(nullptr, 0, infiniFloat32, 1, comm, nullptr);
    if (s != infiniSuccess) {
      return {false, false,
              "send returned " + std::to_string(static_cast<int>(s))};
    }
  } else if (rank == 1) {
    infiniResult_t s = infiniRecv(nullptr, 0, infiniFloat32, 0, comm, nullptr);
    if (s != infiniSuccess) {
      return {false, false,
              "recv returned " + std::to_string(static_cast<int>(s))};
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Helper: run a basic blocking ping `sender → receiver` for `count` floats
// of value `value`, and have the receiver verify on the device side.
// ---------------------------------------------------------------------------
template <Device::Type kDev>
CaseResult RunBlockingPing(infiniComm_t comm, int rank, int sender,
                           int receiver, size_t count, float value) {
  if (rank == sender) {
    float *d_send = AllocFilled<kDev>(count, value);
    infiniResult_t s =
        infiniSend(d_send, count, infiniFloat32, receiver, comm, nullptr);
    Runtime<kDev>::Free(d_send);
    if (s != infiniSuccess) {
      return {false, false,
              "send returned " + std::to_string(static_cast<int>(s))};
    }
  } else if (rank == receiver) {
    float *d_recv = AllocFilled<kDev>(count, -1.0f);
    infiniResult_t s =
        infiniRecv(d_recv, count, infiniFloat32, sender, comm, nullptr);
    if (s != infiniSuccess) {
      Runtime<kDev>::Free(d_recv);
      return {false, false,
              "recv returned " + std::to_string(static_cast<int>(s))};
    }
    bool ok = DeviceEqualsFloat<kDev>(d_recv, count, value);
    Runtime<kDev>::Free(d_recv);
    if (!ok) {
      return {false, false, "received data did not match expected value"};
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Case 2/3: blocking ping rank 0 → 1, rank 0 → size-1
// ---------------------------------------------------------------------------
template <Device::Type kDev>
CaseResult Case_BlockingPing01(infiniComm_t comm, int rank, int size) {
  if (size < 2) return SkipNeed2Ranks(rank);
  return RunBlockingPing<kDev>(comm, rank, /*sender=*/0, /*receiver=*/1,
                               /*count=*/1024, /*value=*/3.5f);
}

template <Device::Type kDev>
CaseResult Case_BlockingPing0Last(infiniComm_t comm, int rank, int size) {
  if (size < 2) return SkipNeed2Ranks(rank);
  return RunBlockingPing<kDev>(comm, rank, /*sender=*/0,
                               /*receiver=*/size - 1,
                               /*count=*/2048, /*value=*/-7.25f);
}

// ---------------------------------------------------------------------------
// Case 4: blocking ping-pong, rank 0 ↔ 1
// ---------------------------------------------------------------------------
template <Device::Type kDev>
CaseResult Case_BlockingPingPong01(infiniComm_t comm, int rank, int size) {
  if (size < 2) return SkipNeed2Ranks(rank);
  constexpr size_t kCount = 512;
  constexpr float kForward = 11.0f;
  constexpr float kReply = -22.0f;

  if (rank == 0) {
    float *d_out = AllocFilled<kDev>(kCount, kForward);
    float *d_in = AllocFilled<kDev>(kCount, -1.0f);

    infiniResult_t s =
        infiniSend(d_out, kCount, infiniFloat32, 1, comm, nullptr);
    if (s != infiniSuccess) {
      Runtime<kDev>::Free(d_out);
      Runtime<kDev>::Free(d_in);
      return {false, false,
              "rank0 send returned " + std::to_string(static_cast<int>(s))};
    }
    s = infiniRecv(d_in, kCount, infiniFloat32, 1, comm, nullptr);
    if (s != infiniSuccess) {
      Runtime<kDev>::Free(d_out);
      Runtime<kDev>::Free(d_in);
      return {false, false,
              "rank0 recv returned " + std::to_string(static_cast<int>(s))};
    }
    bool ok = DeviceEqualsFloat<kDev>(d_in, kCount, kReply);
    Runtime<kDev>::Free(d_out);
    Runtime<kDev>::Free(d_in);
    if (!ok) return {false, false, "rank0 reply mismatch"};
  } else if (rank == 1) {
    float *d_in = AllocFilled<kDev>(kCount, -1.0f);
    infiniResult_t s =
        infiniRecv(d_in, kCount, infiniFloat32, 0, comm, nullptr);
    if (s != infiniSuccess) {
      Runtime<kDev>::Free(d_in);
      return {false, false,
              "rank1 recv returned " + std::to_string(static_cast<int>(s))};
    }
    bool ok = DeviceEqualsFloat<kDev>(d_in, kCount, kForward);
    Runtime<kDev>::Free(d_in);
    if (!ok) return {false, false, "rank1 forward mismatch"};

    float *d_out = AllocFilled<kDev>(kCount, kReply);
    s = infiniSend(d_out, kCount, infiniFloat32, 0, comm, nullptr);
    Runtime<kDev>::Free(d_out);
    if (s != infiniSuccess) {
      return {false, false,
              "rank1 send returned " + std::to_string(static_cast<int>(s))};
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Case 7: large count chunking (blocking, gated)
// ---------------------------------------------------------------------------
template <Device::Type kDev>
CaseResult Case_LargeCount(infiniComm_t comm, int rank, int size) {
  if (size < 2) return SkipNeed2Ranks(rank);
  if (std::getenv("INFINI_SENDRECV_LARGE") == nullptr) {
    return {
        true, true,
        (rank == 0) ? "set INFINI_SENDRECV_LARGE=1 to enable (~2GB/rank)" : ""};
  }
  const size_t count = static_cast<size_t>(std::numeric_limits<int>::max()) +
                       static_cast<size_t>(1024);
  const std::int8_t expected = 0x5A;
  const size_t total_bytes = count * sizeof(std::int8_t);

  if (rank == 0) {
    std::int8_t *d_out = nullptr;
    Runtime<kDev>::Malloc(&d_out, total_bytes);
    std::vector<std::int8_t> h_out(count, expected);
    Runtime<kDev>::Memcpy(d_out, h_out.data(), total_bytes,
                          Runtime<kDev>::MemcpyHostToDevice);
    infiniResult_t s = infiniSend(d_out, count, infiniChar, 1, comm, nullptr);
    Runtime<kDev>::Free(d_out);
    if (s != infiniSuccess) {
      return {false, false,
              "Send returned " + std::to_string(static_cast<int>(s))};
    }
  } else if (rank == 1) {
    std::int8_t *d_in = nullptr;
    Runtime<kDev>::Malloc(&d_in, total_bytes);
    infiniResult_t s = infiniRecv(d_in, count, infiniChar, 0, comm, nullptr);
    if (s != infiniSuccess) {
      Runtime<kDev>::Free(d_in);
      return {false, false,
              "Recv returned " + std::to_string(static_cast<int>(s))};
    }
    std::int8_t probes[3] = {-1, -1, -1};
    Runtime<kDev>::Memcpy(&probes[0], d_in, sizeof(std::int8_t),
                          Runtime<kDev>::MemcpyDeviceToHost);
    Runtime<kDev>::Memcpy(&probes[1], d_in + (count / 2), sizeof(std::int8_t),
                          Runtime<kDev>::MemcpyDeviceToHost);
    Runtime<kDev>::Memcpy(&probes[2], d_in + (count - 1), sizeof(std::int8_t),
                          Runtime<kDev>::MemcpyDeviceToHost);
    Runtime<kDev>::Free(d_in);
    if (probes[0] != expected || probes[1] != expected ||
        probes[2] != expected) {
      return {false, false, "head/mid/tail mismatch"};
    }
  }
  return {};
}

// ---------------------------------------------------------------------------
// Case 8: invalid peer
// ---------------------------------------------------------------------------
CaseResult Case_InvalidPeer(infiniComm_t comm, int rank, int size) {
  if (size < 2) return SkipNeed2Ranks(rank);
  // Only rank 0 attempts the bad calls; the impl rejects them at the base
  // validator before any MPI traffic, so other ranks don't need to mirror.
  if (rank != 0) {
    return {};
  }
  float dummy = 0.f;
  for (int bad_peer : {-1, size}) {
    infiniResult_t s =
        infiniSend(&dummy, 1, infiniFloat32, bad_peer, comm, nullptr);
    if (s != infiniInvalidArgument) {
      return {false, false,
              "Send peer=" + std::to_string(bad_peer) + " expected " +
                  std::to_string(static_cast<int>(infiniInvalidArgument)) +
                  ", got " + std::to_string(static_cast<int>(s))};
    }
    s = infiniRecv(&dummy, 1, infiniFloat32, bad_peer, comm, nullptr);
    if (s != infiniInvalidArgument) {
      return {false, false,
              "Recv peer=" + std::to_string(bad_peer) + " expected " +
                  std::to_string(static_cast<int>(infiniInvalidArgument)) +
                  ", got " + std::to_string(static_cast<int>(s))};
    }
  }
  return {};
}

}  // namespace

int main(int argc, char **argv) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});

  CHECK_INFINI(infiniInit(&argc, &argv));

  int rank = 0;
  int size = 0;
  CHECK_INFINI(infiniGetRank(&rank));
  CHECK_INFINI(infiniGetSize(&size));

  if (rank == 0) {
    std::cout << "=== Send/Recv Test Suite ===" << std::endl;
    std::cout << "Device: " << Device::StringFromType(kDevType) << std::endl;
    std::cout << "Ranks:  " << size << std::endl;
  }

  PrintRankMapping<kDevType>(rank, size);

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

  run("count=0 ping (blocking)", Case_Count0Ping(comm, rank, size));
  run("blocking ping, 0 -> 1", Case_BlockingPing01<kDevType>(comm, rank, size));
  run("blocking ping, 0 -> size-1",
      Case_BlockingPing0Last<kDevType>(comm, rank, size));
  run("blocking ping-pong, 0 <-> 1",
      Case_BlockingPingPong01<kDevType>(comm, rank, size));
  run("large count (>INT_MAX bytes)",
      Case_LargeCount<kDevType>(comm, rank, size));
  run("invalid peer", Case_InvalidPeer(comm, rank, size));

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
