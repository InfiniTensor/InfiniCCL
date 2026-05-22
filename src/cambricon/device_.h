#ifndef INFINI_CCL_CAMBRICON_DEVICE__H_
#define INFINI_CCL_CAMBRICON_DEVICE__H_

#include "device.h"

namespace infini::ccl {

template <>
struct DeviceEnabled<Device::Type::kCambricon> : std::true_type {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_CAMBRICON_DEVICE__H_
