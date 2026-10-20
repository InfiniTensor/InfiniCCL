#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_COMM_INSTANCE_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_COMM_INSTANCE_H_

#include <mpi.h>

#if defined(INFINICCL_WITH_RDMA)
#include "backends/mpi/rdma/rdma_transport.h"
#endif

#include "checks.h"
#include "communicator.h"

namespace infini::ccl {

struct OmpiInstance : public BackendCommInstance {
  MPI_Comm handle = MPI_COMM_NULL;

#if defined(INFINICCL_WITH_RDMA)
  RdmaTransport rdma_transport;
#endif

  OmpiInstance() { type = BackendType::kOmpi; }
  ~OmpiInstance() override { Destroy(); }

  void Destroy() {
#if defined(INFINICCL_WITH_RDMA)
    rdma_transport.Destroy();
#endif
    if (handle != MPI_COMM_WORLD && handle != MPI_COMM_NULL) {
      INFINI_CHECK_MPI(MPI_Comm_free(&handle));
      handle = MPI_COMM_NULL;
    }
    return;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_COMM_INSTANCE_H_
