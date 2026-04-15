#ifndef INFINI_CCL_OMPI_COMM_INSTANCE_H_
#define INFINI_CCL_OMPI_COMM_INSTANCE_H_

#include <mpi.h>

#include "communicator.h"

namespace infini::ccl {

struct OmpiInstance : public BackendCommInstance {
  MPI_Comm handle = MPI_COMM_NULL;

  OmpiInstance() { type = BackendType::kOmpi; }

  void Destroy() {
    // Ensure we don't accidentally leak if a backend duplicates a communicator.
    if (handle != MPI_COMM_WORLD && handle != MPI_COMM_NULL) {
      MPI_Comm_free(&handle);
    }
  }
};

} // namespace infini::ccl

#endif // INFINI_CCL_OMPI_COMM_INSTANCE_H_
