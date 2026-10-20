#ifndef INFINI_CCL_BACKENDS_RDMA_COMM_INSTANCE_H_
#define INFINI_CCL_BACKENDS_RDMA_COMM_INSTANCE_H_

#include <cstddef>
#include <memory>

#include "backends/rdma/rdma_common.h"
#include "communicator.h"
#include "return_status_impl.h"

namespace infini::ccl {

struct RdmaResources;

struct RdmaInstance : public BackendCommInstance {
  RdmaInstance();

  ~RdmaInstance() override;

  ReturnStatus Connect(const RdmaUniqueId &id, int rank);

  ReturnStatus Transfer(void *buffer, size_t bytes, int peer, bool receive,
                        Device::Type device = Device::Type::kCpu,
                        void *stream = nullptr);

  void Destroy();

 private:
  std::unique_ptr<RdmaResources> resources_;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_RDMA_COMM_INSTANCE_H_
