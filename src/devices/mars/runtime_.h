#ifndef INFINI_CCL_DEVICES_MARS_RUNTIME_H_
#define INFINI_CCL_DEVICES_MARS_RUNTIME_H_

#include <hcr/hc_runtime.h>

#include <utility>

#include "devices/cuda/runtime_.h"
#include "devices/mars/device_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kMars>
    : CudaRuntime<Runtime<Device::Type::kMars>> {
  using Stream = hcStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kMars;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != hcSuccess) {
          LOG(hcGetErrorString(static_cast<hcError_t>(status)));
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](auto&&... args) {
    return hcMalloc(std::forward<decltype(args)>(args)...);
  };
  static constexpr auto Free = [](auto&&... args) {
    return hcFree(std::forward<decltype(args)>(args)...);
  };
  static constexpr auto Memcpy = [](auto&&... args) {
    return hcMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyHostToDevice = hcMemcpyHostToDevice;
  static constexpr auto MemcpyDeviceToHost = hcMemcpyDeviceToHost;
  static constexpr auto Memset = hcMemset;
  static constexpr auto GetDevice = hcGetDevice;
  static constexpr auto SetDevice = hcSetDevice;
  static constexpr auto DeviceSynchronize = hcDeviceSynchronize;
  static constexpr auto StreamSynchronize = hcStreamSynchronize;
};

static_assert(Runtime<Device::Type::kMars>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_MARS_RUNTIME_H_
