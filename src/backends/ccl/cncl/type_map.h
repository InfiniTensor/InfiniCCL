#ifndef INFINI_CCL_BACKENDS_CCL_CNCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_CNCL_TYPE_MAP_H_

#include <cncl.h>

#include "backends/ccl/common/api.h"
#include "comm_impl.h"
#include "data_type_impl.h"

namespace infini::ccl {

inline bool DataTypeToCnclType(DataType dtype, cnclDataType_t* cncl_dtype) {
  switch (dtype) {
    case DataType::kInt8:
      *cncl_dtype = cnclInt8;
      return true;
    case DataType::kInt16:
      *cncl_dtype = cnclInt16;
      return true;
    case DataType::kInt32:
      *cncl_dtype = cnclInt32;
      return true;
    case DataType::kInt64:
      *cncl_dtype = cnclInt64;
      return true;
    case DataType::kUInt8:
      *cncl_dtype = cnclUint8;
      return true;
    case DataType::kUInt16:
      *cncl_dtype = cnclUint16;
      return true;
    case DataType::kUInt32:
      *cncl_dtype = cnclUint32;
      return true;
    case DataType::kUInt64:
      *cncl_dtype = cnclUint64;
      return true;
    case DataType::kFloat16:
      *cncl_dtype = cnclFloat16;
      return true;
    case DataType::kBFloat16:
      *cncl_dtype = cnclBfloat16;
      return true;
    case DataType::kFloat32:
      *cncl_dtype = cnclFloat32;
      return true;
    default:
      return false;
  }
}

template <>
struct CclTypeMap<BackendType::kCncl, Device::Type::kCambricon> {
  using Api = CclApi<BackendType::kCncl, Device::Type::kCambricon>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType* backend_dtype) {
    return DataTypeToCnclType(dtype, backend_dtype);
  }

  static bool ToBackendRedOp(ReductionOpType red_op,
                             typename Api::RedOp* backend_op) {
    if (red_op == ReductionOpType::kAvg) {
      return false;
    }
    *backend_op = static_cast<cnclReduceOp_t>(red_op);
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_CNCL_TYPE_MAP_H_
