#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_

#include <hccl/hccl.h>

#include "backends/ccl/common/api.h"
#include "comm_impl.h"
#include "constexpr_map.h"
#include "data_type_impl.h"

namespace infini::ccl {

static constexpr ConstexprMap<DataType, HcclDataType, 12> kHcclDataTypeMap{{{
    {DataType::kInt8, HCCL_DATA_TYPE_INT8},
    {DataType::kInt16, HCCL_DATA_TYPE_INT16},
    {DataType::kInt32, HCCL_DATA_TYPE_INT32},
    {DataType::kInt64, HCCL_DATA_TYPE_INT64},
    {DataType::kUInt8, HCCL_DATA_TYPE_UINT8},
    {DataType::kUInt16, HCCL_DATA_TYPE_UINT16},
    {DataType::kUInt32, HCCL_DATA_TYPE_UINT32},
    {DataType::kUInt64, HCCL_DATA_TYPE_UINT64},
    {DataType::kFloat16, HCCL_DATA_TYPE_FP16},
    {DataType::kBFloat16, HCCL_DATA_TYPE_BFP16},
    {DataType::kFloat32, HCCL_DATA_TYPE_FP32},
    {DataType::kFloat64, HCCL_DATA_TYPE_FP64},
}}};

static constexpr ConstexprMap<ReductionOpType, HcclReduceOp, 4> kHcclOpMap{{{
    {ReductionOpType::kSum, HCCL_REDUCE_SUM},
    {ReductionOpType::kProd, HCCL_REDUCE_PROD},
    {ReductionOpType::kMax, HCCL_REDUCE_MAX},
    {ReductionOpType::kMin, HCCL_REDUCE_MIN},
}}};

inline bool DataTypeToHcclType(DataType dtype, HcclDataType* hccl_dtype) {
  if (dtype < DataType::kInt8 || dtype >= DataType::kNumTypes) {
    return false;
  }
  *hccl_dtype = kHcclDataTypeMap.at(dtype);
  return true;
}

inline bool RedOpToHcclOp(ReductionOpType red_op, HcclReduceOp* hccl_op) {
  if (red_op < ReductionOpType::kSum || red_op > ReductionOpType::kMin) {
    return false;
  }
  *hccl_op = kHcclOpMap.at(red_op);
  return true;
}

template <>
struct CclTypeMap<BackendType::kHccl, Device::Type::kAscend> {
  using Api = CclApi<BackendType::kHccl, Device::Type::kAscend>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType* backend_dtype) {
    return DataTypeToHcclType(dtype, backend_dtype);
  }

  static bool ToBackendRedOp(ReductionOpType red_op,
                             typename Api::RedOp* backend_op) {
    return RedOpToHcclOp(red_op, backend_op);
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_
