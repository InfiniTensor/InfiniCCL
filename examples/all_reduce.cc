/**
 * InfiniCCL Example: AllReduce
 * * This example demonstrates the planned API for performing a
 * collective sum-reduction across multiple GPUs and nodes.
 */

#include <iostream>
#include <numeric>
#include <vector>

#include "infiniccl.h"

// Simple check macro for the C-API
#define CHECK_INFINI(cmd)                                                      \
  do {                                                                         \
    infiniResult_t res = (cmd);                                                \
    if (res != infiniSuccess) {                                                \
      std::cerr << "InfiniCCL Error at " << __LINE__ << std::endl;             \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

int main(int argc, char **argv) {
  // 1. Initialize the Runtime
  // Currently, this is the only fully implemented part!
  // It bootstraps the underlying MPI/OMPI environment.
  CHECK_INFINI(infiniInit(&argc, &argv));

  int rank, size;
  // Planned API to get global rank info
  // For now, these would be wrappers around MPI_Comm_rank/size
  // infiniGetRank(&rank);
  // infiniGetSize(&size);

  std::cout << "Rank initialized successfully." << std::endl;

  // 2. Setup Device (NVIDIA/MetaX)
  // In a real scenario, we'd use local_rank to pick a GPU
  // int local_rank = rank % gpus_per_node;
  // Runtime::setDevice(local_rank);

  // 3. Setup Communicator (Planned)
  // infiniComm_t comm;
  // CHECK_INFINI(infiniCommInitAll(&comm, size, nullptr));

  // 4. Prepare Data
  const int count = 1024;
  std::vector<float> h_send(count, 1.0f);
  std::vector<float> h_recv(count, 0.0f);

  /* Note: In a final version, we would allocate GPU memory here:
     float *d_send, *d_recv;
     Runtime::malloc(&d_send, count * sizeof(float));
     Runtime::memcpy(d_send, h_send.data(), ...);
  */

  // 5. Perform AllReduce (Conceptual API)
  std::cout << "Starting AllReduce operation (Placeholder)..." << std::endl;

  /*
  CHECK_INFINI(infiniAllReduce(
      h_send.data(),  // Input (eventually d_send)
      h_recv.data(),  // Output (eventually d_recv)
      count,
      infiniFloat32,
      infiniSum,
      comm,
      nullptr         // Stream
  ));
  */

  // 6. Cleanup
  // CHECK_INFINI(infiniCommDestroy(comm));
  //   CHECK_INFINI(infiniFinalize());

  std::cout << "InfiniCCL finalized." << std::endl;
  return 0;
}
