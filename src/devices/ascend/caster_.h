#ifndef INFINI_CCL_DEVICES_ASCEND_CASTER_H_
#define INFINI_CCL_DEVICES_ASCEND_CASTER_H_

#include "caster.h"
#include "devices/ascend/data_type_.h"

namespace infini::ccl {

template <>
struct HardwareCastImpl<Device::Type::kAscend, float, op::fp16_t> {
  static float Apply(op::fp16_t x) { return static_cast<float>(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kAscend, op::fp16_t, float> {
  static op::fp16_t Apply(float x) { return op::fp16_t(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kAscend, float, op::bfloat16> {
  static float Apply(op::bfloat16 x) { return static_cast<float>(x); }
};

template <>
struct HardwareCastImpl<Device::Type::kAscend, op::bfloat16, float> {
  static op::bfloat16 Apply(float x) { return op::bfloat16(x); }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ASCEND_CASTER_H_
