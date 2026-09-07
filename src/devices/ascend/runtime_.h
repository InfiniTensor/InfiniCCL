#ifndef INFINI_CCL_DEVICES_ASCEND_RUNTIME_H_
#define INFINI_CCL_DEVICES_ASCEND_RUNTIME_H_

#include <acl/acl.h>

#include <string>

#include "devices/ascend/device_.h"
#include "logging.h"
#include "return_status_impl.h"
#include "runtime.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kAscend>
    : DeviceRuntime<Runtime<Device::Type::kAscend>> {
  using Stream = aclrtStream;

  static constexpr Device::Type kDeviceType = Device::Type::kAscend;

  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != ACL_SUCCESS) {
          const char* message = aclGetRecentErrMsg();
          if (message) {
            LOG(message);
          } else {
            LOG(("Ascend ACL error code: " +
                 std::to_string(static_cast<int>(status)))
                    .c_str());
          }
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](void** ptr, size_t size) {
    return aclrtMalloc(ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  };

  static constexpr auto Memcpy = [](void* dst, const void* src, size_t count,
                                    aclrtMemcpyKind kind) {
    return aclrtMemcpy(dst, count, src, count, kind);
  };

  static constexpr auto Free = aclrtFree;

  static constexpr auto MemcpyHostToDevice = ACL_MEMCPY_HOST_TO_DEVICE;

  static constexpr auto MemcpyDeviceToHost = ACL_MEMCPY_DEVICE_TO_HOST;

  static constexpr auto Memset = [](void* ptr, int value, size_t count) {
    return aclrtMemset(ptr, count, value, count);
  };

  static constexpr auto GetDevice = aclrtGetDevice;

  static constexpr auto SetDevice = aclrtSetDevice;

  static aclError EnsureDeviceContext() {
    int32_t device_id = 0;
    const aclError status = aclrtGetDevice(&device_id);
    if (status == ACL_SUCCESS) {
      return ACL_SUCCESS;
    }
    return aclrtSetDevice(0);
  }

  static constexpr auto DeviceSynchronize = aclrtSynchronizeDevice;

  static constexpr auto StreamCreate = aclrtCreateStream;

  static constexpr auto StreamDestroy = aclrtDestroyStream;

  static constexpr auto StreamSynchronize = [](aclrtStream stream) {
    return stream ? aclrtSynchronizeStream(stream) : aclrtSynchronizeDevice();
  };
};

static_assert(Runtime<Device::Type::kAscend>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ASCEND_RUNTIME_H_
