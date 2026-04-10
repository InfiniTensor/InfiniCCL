#ifndef INFINI_CCL_NVIDIA_NCCL_COMM_INSTANCE_H_
#define INFINI_CCL_NVIDIA_NCCL_COMM_INSTANCE_H_

#include <nccl.h>

#include "communicator.h"

namespace infini::ccl {

struct NcclInstance : public BackendCommInstance {
  ncclComm_t handle;
  NcclInstance() { type = BackendType::kNccl; }
};

} // namespace infini::ccl

#endif
