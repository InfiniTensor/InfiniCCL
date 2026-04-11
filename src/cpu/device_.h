#ifndef INFINI_CCL_CPU_DEVICE__H_
#define INFINI_CCL_CPU_DEVICE__H_

#include "device.h"

namespace infini::ccl {

template <> struct DeviceEnabled<Device::Type::kCpu> : std::true_type {};

} // namespace infini::ccl

#endif
