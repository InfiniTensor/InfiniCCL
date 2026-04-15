/**
 * InfiniCCL Example: AllReduce
 * * This example demonstrates the planned API for performing a
 * collective sum-reduction across multiple GPUs and nodes.
 */

#include <iostream>
#include <unistd.h>
#include <vector>

// Public API
#include "infiniccl.h"

// Example-specific utilities
#include "utils.h"

// Internal Headers (Accessible via example-specific include paths, technically
// not public APIs)
#include "backend_manifest.h"
#include "device.h"
#include "runtime.h"
#include "traits.h"

using namespace infini::ccl;

int main(int argc, char **argv) {
  constexpr Device::Type kDevType =
      ListGetBest<DevicePriority>(EnabledDevices{});

  CHECK_INFINI(infiniInit(&argc, &argv));

  int rank, size;
  CHECK_INFINI(infiniGetRank(&rank));
  CHECK_INFINI(infiniGetSize(&size));

  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  // Map local rank to GPU device
  const char *local_rank_str = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  int local_rank = 0;
  if (local_rank_str != nullptr) {
    local_rank = std::atoi(local_rank_str);
  }

  std::cout << "[Rank " << rank << "] Host: " << hostname
            << " | GPU: " << Device::StringFromType(kDevType) << " "
            << " | Device " << local_rank << std::endl;

  std::cout << "Rank initialized successfully." << std::endl;

  // 2. Setup Device (NVIDIA/MetaX)
  // In a real scenario, we'd use local_rank to pick a GPU
  // int local_rank = rank % gpus_per_node;
  // Runtime::setDevice(local_rank);

  // 3. Setup Communicator (Planned)
  // infiniComm_t comm;
  int *devlist = new int[size];
  int num_gpus_per_node = (size > 1) ? (size >> 1) : 1;

  for (int i = 0; i < size; i++) {
    devlist[i] = i % num_gpus_per_node;
  }
  // CHECK_INFINI(infiniCommInitAll(&comm, size, nullptr));

  // 4. Prepare Data
  const int kNumElements = 1024;
  std::vector<float> h_send(kNumElements, 1.0f);
  std::vector<float> h_recv(kNumElements, 0.0f);

  float *d_send, *d_recv;
  Runtime<kDevType>::Malloc(&d_send, kNumElements * sizeof(*d_send));
  Runtime<kDevType>::Malloc(&d_recv, kNumElements * sizeof(*d_recv));
  Runtime<kDevType>::Memcpy(d_send, h_send.data(),
                            kNumElements * sizeof(*d_send),
                            Runtime<kDevType>::MemcpyHostToDevice);
  Runtime<kDevType>::Memcpy(d_recv, h_recv.data(),
                            kNumElements * sizeof(*d_recv),
                            Runtime<kDevType>::MemcpyHostToDevice);

  // 5. Perform AllReduce (Conceptual API)
  std::cout << "Starting AllReduce operation (Placeholder)..." << std::endl;

  /*
  CHECK_INFINI(infiniAllReduce(
      d_send,
      d_recv,
      kNumElements,
      infiniFloat32,
      infiniSum,
      comm,
      nullptr
  ));
  */

  Runtime<kDevType>::Memcpy(h_recv.data(), d_recv, kNumElements * sizeof(float),
                            Runtime<kDevType>::MemcpyDeviceToHost);

  // 6. Cleanup
  Runtime<kDevType>::Free(d_send);
  Runtime<kDevType>::Free(d_recv);
  delete[] devlist;

  // CHECK_INFINI(infiniCommDestroy(comm));
  CHECK_INFINI(infiniFinalize());

  std::cout << "InfiniCCL finalized." << std::endl;

  return 0;
}
