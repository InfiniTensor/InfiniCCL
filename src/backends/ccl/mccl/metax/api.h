#ifndef INFINI_CCL_BACKENDS_CCL_MCCL_METAX_API_H_
#define INFINI_CCL_BACKENDS_CCL_MCCL_METAX_API_H_

#include "backends/ccl/mccl/api.h"
#include "devices/metax/runtime_.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kMccl, Device::Type::kMetax>
    : McclApi<Device::Type::kMetax> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_MCCL_METAX_API_H_
