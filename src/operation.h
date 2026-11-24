#ifndef INFINI_CCL_OPERATION_H_
#define INFINI_CCL_OPERATION_H_

#include <memory>
#include <utility>

#include "backend.h"
#include "backend_device_map.h"
#include "communicator.h"
#include "device.h"
#include "dispatch_policy.h"
#include "dispatcher.h"
#include "return_status_impl.h"
#include "traits.h"

namespace infini::ccl {

template <typename Key, BackendType backend_type = BackendType::kCount,
          Device::Type device_type = Device::Type::kCount>
class Operation {
 public:
  template <typename... Args>
  static auto Call(Args &&...args) {
    constexpr BackendType kBestBack =
        ListGetBest<BackendPriority>(ActiveBackends<Key>{});
    constexpr Device::Type kBestDev =
        ListGetBest<DevicePriority>(ActiveDevices<Key>{});

    return Call(kBestBack, kBestDev, std::forward<Args>(args)...);
  }

  template <typename... Args>
  static auto Call(BackendType backend, Device::Type device, Args &&...args) {
    return DispatchFunc<ActiveBackends<Key>, ActiveDevices<Key>>(
        {static_cast<int64_t>(backend), static_cast<int64_t>(device)},
        [&](auto resolved_list) {
          constexpr BackendType kBackend =
              static_cast<BackendType>(ListGet<0>(resolved_list));
          constexpr Device::Type kDevice =
              static_cast<Device::Type>(ListGet<1>(resolved_list));

          if constexpr (IsSupportedCombination<kBackend, kDevice>::value) {
            return Key::template Execute<kBackend, kDevice>(
                std::forward<Args>(args)...);
          } else {
            return ReturnStatus::kNotSupported;
          }
        },
        "Operation::Call");
  }

 protected:
  // Carrier-aware dispatch for `Policy::kCarrierAware` operations: when the
  // priority winner is attached to neither communicator slot, re-dispatch to
  // the best enabled backend the communicator carries. Only the priority
  // winner can enter this path, so the redirected call never redirects
  // again. Dependent on `Key`, the body is instantiated only at real call
  // sites, after every `BackendEnabled` specialization is visible.
  template <BackendType backend, typename... Args>
  static bool RedirectToCarrier(void *comm_handle, ReturnStatus *status,
                                Args &&...args) {
    if constexpr (DispatchPolicy<Key>::value == Policy::kCarrierAware) {
      auto *comm = static_cast<Communicator *>(comm_handle);
      constexpr auto enabled_backends = ActiveBackends<Key>{};
      if (backend != ListGetBest<BackendPriority>(enabled_backends)) {
        return false;
      }

      // The communicator decides: a slot already carrying the winner means
      // the call belongs to that backend.
      if (comm->intra_comm_backend() == backend ||
          comm->inter_comm_backend() == backend) {
        return false;
      }

      for (BackendType carrier :
           {comm->intra_comm_backend(), comm->inter_comm_backend()}) {
        if (carrier != BackendType::kCount &&
            IsBackendEnabled(enabled_backends, carrier)) {
          *status = Key::template Call(carrier, comm->device_type(),
                                       std::forward<Args>(args)...);
          return true;
        }
      }
    }

    return false;
  }

 private:
  template <auto... backends>
  static bool IsBackendEnabled(List<backends...>, BackendType backend) {
    return ((backend == backends) || ...);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_OPERATION_H_
