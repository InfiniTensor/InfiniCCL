#ifndef INFINI_CCL_OMPI_COMM_INSTANCE_H_
#define INFINI_CCL_OMPI_COMM_INSTANCE_H_

#include <mpi.h>

#include "communicator.h"

namespace infini::ccl {

struct OmpiInstance : public BackendCommInstance {
  MPI_Comm handle;
  OmpiInstance() { type = BackendType::kOmpi; }
};

} // namespace infini::ccl

#endif
