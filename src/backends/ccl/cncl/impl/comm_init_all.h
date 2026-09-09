#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_ALL_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_ALL_H_

#include <memory>
#include <numeric>
#include <vector>

#include "backends/ccl/common/comm_instance.h"
#include "base/comm_init_all.h"
#include "communicator.h"
#include "runtime.h"

namespace infini::ccl {

template <Device::Type device>
class CommInitAllImpl<BackendType::kCncl, device> {
 public:
  static ReturnStatus Apply(void** comm_handles, int n_dev,
                            const int* dev_list) {
    using Api = CclApi<BackendType::kCncl, device>;
    using CommInstance = CclCommInstance<Api>;
    using Rt = Runtime<device>;

    if (!comm_handles || !dev_list || n_dev <= 0) {
      return ReturnStatus::kInvalidArgument;
    }

    auto** comms = reinterpret_cast<Communicator**>(comm_handles);
    for (int i = 0; i < n_dev; ++i) {
      if (comms[i]) {
        return ReturnStatus::kInvalidArgument;
      }
    }

    std::vector<std::unique_ptr<Communicator>> wrappers;
    std::vector<std::unique_ptr<CommInstance>> instances;
    std::vector<typename Api::Comm> backend_comms(n_dev);
    std::vector<int> rank_list(n_dev);
    wrappers.reserve(n_dev);
    instances.reserve(n_dev);
    std::iota(rank_list.begin(), rank_list.end(), 0);

    for (int i = 0; i < n_dev; ++i) {
      auto status = Rt::Check(Rt::SetDevice(dev_list[i]));
      if (status != ReturnStatus::kSuccess) {
        return status;
      }
      wrappers.emplace_back(
          std::make_unique<Communicator>(device, dev_list[i]));
      instances.emplace_back(std::make_unique<CommInstance>());
    }

    auto status = Api::Check(Api::CommInitAll(backend_comms.data(), n_dev,
                                              dev_list, rank_list.data()));
    if (status != ReturnStatus::kSuccess) {
      return status;
    }

    for (int i = 0; i < n_dev; ++i) {
      instances[i]->handle = backend_comms[i];
      wrappers[i]->set_world_info(i, n_dev);
      wrappers[i]->set_intra_comm(std::move(instances[i]));
      comms[i] = wrappers[i].release();
    }

    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<CommInitAll, BackendType::kCncl> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_IMPL_COMM_INIT_ALL_H_
