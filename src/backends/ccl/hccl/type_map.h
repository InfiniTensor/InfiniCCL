#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_TYPE_MAP_H_

#include <hccl/hccl.h>

#include "backends/ccl/common/api.h"
#include "comm_impl.h"
#include "data_type_impl.h"

namespace infini::ccl {

inline bool DataTypeToHcclType(DataType dtype, HcclDataType* hccl_dtype) {
  switch (dtype) {
    case DataType::kInt8:
      *hccl_dtype = HCCL_DATA_TYPE_INT8;
      return true;
    case DataType::kInt16:
      *hccl_dtype = HCCL_DATA_TYPE_INT16;
      return true;
    case DataType::kInt32:
      *hccl_dtype = HCCL_DATA_TYPE_INT32;
      return true;
    case DataType::kInt64:
      *hccl_dtype = HCCL_DATA_TYPE_INT64;
      return true;
    case DataType::kUInt8:
      *hccl_dtype = HCCL_DATA_TYPE_UINT8;
      return true;
    case DataType::kUInt16:
      *hccl_dtype = HCCL_DATA_TYPE_UINT16;
      return true;
    case DataType::kUInt32:
      *hccl_dtype = HCCL_DATA_TYPE_UINT32;
      return true;
    case DataType::kUInt64:
      *hccl_dtype = HCCL_DATA_TYPE_UINT64;
      return true;
    case DataType::kFloat16:
      *hccl_dtype = HCCL_DATA_TYPE_FP16;
      return true;
    case DataType::kBFloat16:
      *hccl_dtype = HCCL_DATA_TYPE_BFP16;
      return true;
    case DataType::kFloat32:
      *hccl_dtype = HCCL_DATA_TYPE_FP32;
      return true;
    case DataType::kFloat64:
      *hccl_dtype = HCCL_DATA_TYPE_FP64;
      return true;
    default:
      return false;
  }
}

inline bool RedOpToHcclOp(ReductionOpType red_op, HcclReduceOp* hccl_op) {
  switch (red_op) {
    case ReductionOpType::kSum:
      *hccl_op = HCCL_REDUCE_SUM;
      return true;
    case ReductionOpType::kProd:
      *hccl_op = HCCL_REDUCE_PROD;
      return true;
    case ReductionOpType::kMax:
      *hccl_op = HCCL_REDUCE_MAX;
      return true;
    case ReductionOpType::kMin:
      *hccl_op = HCCL_REDUCE_MIN;
      return true;
    default:
      return false;
  }
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
