#ifndef INFINI_CCL_CAMBRICON_DATA_TYPE__H_
#define INFINI_CCL_CAMBRICON_DATA_TYPE__H_

#if defined(__mlu__) || defined(__BANG__)
#include "bang_bf16.h"
#else
// If compiling on the host side using standard GCC, mock the storage types
// to prevent header poisoning and avoid `bang_fp16.h` dependency altogether.
#ifndef HAS_CAMBRICON_HOST_HALF_MOCKS
#define HAS_CAMBRICON_HOST_HALF_MOCKS

typedef uint16_t half;
typedef uint16_t bfloat16_t;

#endif
#endif

#include "cambricon/device_.h"
#include "data_type_impl.h"

namespace infini::ccl {

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kFloat16> {
  using type = half;
};

template <>
struct TypeMap<Device::Type::kCambricon, DataType::kBFloat16> {
  using type = bfloat16_t;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_CAMBRICON_DATA_TYPE__H_
