#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_KUNLUN_API_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_KUNLUN_API_H_

#include "backends/ccl/bkcl/api.h"
#include "devices/kunlun/runtime_.h"

namespace infini::ccl {

template <>
struct CclApi<BackendType::kBkcl, Device::Type::kKunlun>
    : BkclApi<Device::Type::kKunlun> {};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_KUNLUN_API_H_
