#ifndef INFINI_CCL_BACKEND_DEVICE_MAP_H_
#define INFINI_CCL_BACKEND_DEVICE_MAP_H_

#include "backend.h"
#include "device.h"

namespace infini::ccl {

template <BackendType backend, Device::Type device>
struct IsSupportedCombination : std::false_type {};

// OpenMPI is device-agnostic.
template <Device::Type D>
struct IsSupportedCombination<BackendType::kOmpi, D> : std::true_type {};

// Mars device communication uses HCCL; OpenMPI has no Mars data type caster.
template <>
struct IsSupportedCombination<BackendType::kOmpi, Device::Type::kMars>
    : std::false_type {};

template <>
struct IsSupportedCombination<BackendType::kNccl, Device::Type::kNvidia>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kNccl, Device::Type::kThead>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kNccl, Device::Type::kIluvatar>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kNccl, Device::Type::kHygon>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kMccl, Device::Type::kMetax>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kMccl, Device::Type::kMoore>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kCncl, Device::Type::kCambricon>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kHccl, Device::Type::kAscend>
    : std::true_type {};

template <>
struct IsSupportedCombination<BackendType::kHccl, Device::Type::kMars>
    : std::true_type {};

};  // namespace infini::ccl

#endif  // INFINI_CCL_BACKEND_DEVICE_MAP_H_
