#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_MARS_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_MARS_TYPE_MAP_H_

#include <hccl.h>

#include "backends/ccl/common/api.h"
#include "comm_impl.h"
#include "constexpr_map.h"
#include "data_type_impl.h"

namespace infini::ccl {

static constexpr ConstexprMap<DataType, hcclDataType_t, 12>
    kMarsHcclDataTypeMap{{{
        {DataType::kInt8, hcclInt8},
        {DataType::kInt16, hcclNumTypes},
        {DataType::kInt32, hcclInt32},
        {DataType::kInt64, hcclInt64},
        {DataType::kUInt8, hcclUint8},
        {DataType::kUInt16, hcclNumTypes},
        {DataType::kUInt32, hcclUint32},
        {DataType::kUInt64, hcclUint64},
        {DataType::kFloat32, hcclFloat32},
        {DataType::kFloat64, hcclFloat64},
        {DataType::kFloat16, hcclFloat16},
        {DataType::kBFloat16, hcclBfloat16},
    }}};

static constexpr ConstexprMap<ReductionOpType, hcclRedOp_t, 5> kMarsHcclOpMap{{{
    {ReductionOpType::kSum, hcclSum},
    {ReductionOpType::kProd, hcclProd},
    {ReductionOpType::kMax, hcclMax},
    {ReductionOpType::kMin, hcclMin},
    {ReductionOpType::kAvg, hcclAvg},
}}};

template <>
struct CclTypeMap<BackendType::kHccl, Device::Type::kMars> {
  using Api = CclApi<BackendType::kHccl, Device::Type::kMars>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType* backend_dtype) {
    auto hccl_dtype = kMarsHcclDataTypeMap.at(dtype);
    if (hccl_dtype == hcclNumTypes) {
      return false;
    }
    *backend_dtype = hccl_dtype;
    return true;
  }

  static bool ToBackendRedOp(ReductionOpType op,
                             typename Api::RedOp* backend_op) {
    *backend_op = kMarsHcclOpMap.at(op);
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_MARS_TYPE_MAP_H_
