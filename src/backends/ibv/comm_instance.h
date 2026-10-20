#ifndef INFINI_CCL_BACKENDS_IBV_COMM_INSTANCE_H_
#define INFINI_CCL_BACKENDS_IBV_COMM_INSTANCE_H_

#include <cstddef>
#include <memory>

#include "backends/ibv/ibv_common.h"
#include "communicator.h"
#include "return_status_impl.h"

namespace infini::ccl {

struct IbvResources;

struct IbvInstance : public BackendCommInstance {
  IbvInstance();

  ~IbvInstance() override;

  ReturnStatus Connect(const IbvUniqueId &id, int rank);

  ReturnStatus Transfer(void *buffer, size_t bytes, int peer, bool receive,
                        Device::Type device = Device::Type::kCpu,
                        void *stream = nullptr);

  void Destroy();

 private:
  std::unique_ptr<IbvResources> resources_;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_IBV_COMM_INSTANCE_H_
