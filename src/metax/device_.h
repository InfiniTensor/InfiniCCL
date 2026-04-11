#ifndef INFINI_CCL_METAX_DEVICE__H_
#define INFINI_CCL_METAX_DEVICE__H_

#include "device.h"

namespace infini::ccl {

template <> struct DeviceEnabled<Device::Type::kMetax> : std::true_type {};

} // namespace infini::ccl

#endif // INFINI_CCL_METAX_DEVICE__H_
