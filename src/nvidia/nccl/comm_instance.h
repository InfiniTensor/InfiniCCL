#ifndef INFINI_CCL_NVIDIA_NCCL_COMM_INSTANCE_H_
#define INFINI_CCL_NVIDIA_NCCL_COMM_INSTANCE_H_

#include <nccl.h>

#include "checks.h"
#include "communicator.h"

namespace infini::ccl {

struct NcclInstance : public BackendCommInstance {
  ncclComm_t handle;
  NcclInstance() { type = BackendType::kNccl; }

  void Destroy() {
    if (handle != nullptr) {
      INFINI_CHECK_NCCL(ncclCommDestroy(handle));
      handle = nullptr;
    }
  }
};

}  // namespace infini::ccl

#endif
