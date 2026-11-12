#ifndef INFINI_CCL_DEVICES_QY_RUNTIME_H_
#define INFINI_CCL_DEVICES_QY_RUNTIME_H_

#include <utility>

// clang-format off
#include <cuda_runtime.h>
// clang-format on

#include "devices/cuda/runtime_.h"
#include "devices/qy/device_.h"
#include "logging.h"
#include "return_status_impl.h"

namespace infini::ccl {

template <>
struct Runtime<Device::Type::kQy> : CudaRuntime<Runtime<Device::Type::kQy>> {
  using Stream = cudaStream_t;

  static constexpr Device::Type kDeviceType = Device::Type::kQy;

  // Denglin overloads several cuda* entry points; wrap in forwarding lambdas
  // so a single candidate is selected (same pattern as InfiniRT QY).
  static constexpr auto Check =
      [](auto status, ReturnStatus err_code = ReturnStatus::kSystemError) {
        if (status != cudaSuccess) {
          LOG(cudaGetErrorString(static_cast<cudaError_t>(status)));
          return err_code;
        }
        return ReturnStatus::kSuccess;
      };

  static constexpr auto Malloc = [](auto &&...args) {
    return cudaMalloc(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Memcpy = [](auto &&...args) {
    return cudaMemcpy(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto Free = [](auto &&...args) {
    return cudaFree(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto MemcpyHostToDevice = cudaMemcpyHostToDevice;

  static constexpr auto MemcpyDeviceToHost = cudaMemcpyDeviceToHost;

  static constexpr auto Memset = [](auto &&...args) {
    return cudaMemset(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto GetDevice = [](auto &&...args) {
    return cudaGetDevice(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto SetDevice = [](auto &&...args) {
    return cudaSetDevice(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto DeviceSynchronize = [](auto &&...args) {
    return cudaDeviceSynchronize(std::forward<decltype(args)>(args)...);
  };

  static constexpr auto StreamSynchronize = [](auto &&...args) {
    return cudaStreamSynchronize(std::forward<decltype(args)>(args)...);
  };
};

static_assert(Runtime<Device::Type::kQy>::Validate());

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_QY_RUNTIME_H_
