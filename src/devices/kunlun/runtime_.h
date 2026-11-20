#ifndef INFINI_CCL_DEVICES_KUNLUN_RUNTIME_H_
#define INFINI_CCL_DEVICES_KUNLUN_RUNTIME_H_

#include <cstring>

// clang-format off
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>
// clang-format on

#include "devices/kunlun/device_.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

/// Kunlun XPU runtime (XRE `libxpurt`), mirroring the InfiniRT kunlun
/// backend. XRE has no whole-device pointer query and no memset; neither is
/// required by the InfiniCCL device interface.
template <>
struct Runtime<Device::Type::kKunlun>
    : DeviceRuntime<Runtime<Device::Type::kKunlun>> {
  using Error = int;

  using Stream = XPUStream;

  static constexpr Device::Type kDeviceType = Device::Type::kKunlun;

  // Follows the InfiniRT/backend-neutral 0..3 MemcpyKind convention.
  static constexpr int MemcpyHostToHost = 0;
  static constexpr int MemcpyHostToDevice = 1;
  static constexpr int MemcpyDeviceToHost = 2;
  static constexpr int MemcpyDeviceToDevice = 3;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != XPU_SUCCESS) {
          LOG(("Kunlun XRE error code: " +
               std::to_string(static_cast<int>(status)))
                  .c_str());
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](void** ptr, size_t size) {
    return xpu_malloc(ptr, static_cast<std::uint64_t>(size));
  };

  static constexpr auto Free = xpu_free;

  static Error Memcpy(void* dst, const void* src, std::size_t count,
                      int kind) {
    switch (kind) {
      case MemcpyHostToHost:
        std::memcpy(dst, src, count);
        return XPU_SUCCESS;
      case MemcpyHostToDevice:
        return xpu_memcpy(dst, src, static_cast<std::uint64_t>(count),
                          XPU_HOST_TO_DEVICE);
      case MemcpyDeviceToHost:
        return xpu_memcpy(dst, src, static_cast<std::uint64_t>(count),
                          XPU_DEVICE_TO_HOST);
      case MemcpyDeviceToDevice:
        return xpu_memcpy(dst, src, static_cast<std::uint64_t>(count),
                          XPU_DEVICE_TO_DEVICE);
      default:
        return XPUERR_INVALID_PARAM;
    }
  }

  static constexpr auto GetDevice = xpu_current_device;

  static constexpr auto SetDevice = xpu_set_device;

  static Error DeviceSynchronize() {
    // XRE has no whole-device synchronize; `xpu_wait()` waits on the
    // default stream, matching the InfiniRT kunlun backend.
    return xpu_wait();
  }

  static constexpr auto StreamCreate = [](Stream* stream) -> Error {
    return xpu_stream_create(stream);
  };

  static constexpr auto StreamDestroy = [](Stream stream) -> Error {
    return xpu_stream_destroy(stream);
  };

  static constexpr auto StreamSynchronize = [](Stream stream) {
    return xpu_wait(stream);
  };
};

static_assert(Runtime<Device::Type::kKunlun>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_KUNLUN_RUNTIME_H_
