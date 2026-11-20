#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_TYPE_MAP_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_TYPE_MAP_H_

#include <xccl.h>

#include <string>

#include "backends/ccl/common/api.h"
#include "backends/ccl/bkcl/api.h"
#include "comm_impl.h"
#include "data_type_impl.h"
#include "logging.h"

namespace infini::ccl {

// XCCL supports float/float16/bfloat16/float64/int32/int64/uint8 only. There
// is no "unsupported" enumerator, so -1 is used as the ConstexprMap sentinel
// and rejected before reaching the library.
constexpr XCCLDataType kBkclUnsupported = static_cast<XCCLDataType>(-1);

static constexpr ConstexprMap<DataType, XCCLDataType, 12> kBkclDataTypeMap{{{
    {DataType::kInt8, kBkclUnsupported},
    {DataType::kInt16, kBkclUnsupported},
    {DataType::kInt32, XCCL_INT32},
    {DataType::kInt64, XCCL_INT64},
    {DataType::kUInt8, XCCL_UINT8},
    {DataType::kUInt16, kBkclUnsupported},
    {DataType::kUInt32, kBkclUnsupported},
    {DataType::kUInt64, kBkclUnsupported},
    {DataType::kFloat16, XCCL_FLOAT16},
    {DataType::kBFloat16, XCCL_BFLOAT16},
    {DataType::kFloat32, XCCL_FLOAT},
    {DataType::kFloat64, XCCL_FLOAT64},
}}};

static constexpr ConstexprMap<ReductionOpType, XCCLOp, 5> kBkclOpMap{{{
    {ReductionOpType::kSum, XCCL_ADD},
    {ReductionOpType::kProd, XCCL_PRODUCT},
    {ReductionOpType::kMax, XCCL_MAX},
    {ReductionOpType::kMin, XCCL_MIN},
    {ReductionOpType::kAvg, static_cast<XCCLOp>(-1)},
}}};

template <Device::Type device>
inline XCCLDataType DataTypeToBkclType(DataType dtype) {
  auto bkcl_dtype = kBkclDataTypeMap.at(dtype);

  if (bkcl_dtype == kBkclUnsupported) {
    // The requested data type is not supported by XCCL.
    LOG(("DataType '" + std::string(kDataTypeToDesc.at(dtype)) +
         "' is not supported by the BKCL backend")
            .c_str());
  }

  return bkcl_dtype;
}

inline XCCLOp RedOpToBkclOp(ReductionOpType red_op) {
  return kBkclOpMap.at(red_op);
}

template <Device::Type device>
struct CclTypeMap<BackendType::kBkcl, device> {
  using Api = CclApi<BackendType::kBkcl, device>;

  static bool ToBackendDataType(DataType dtype,
                                typename Api::DataType* backend_dtype) {
    auto bkcl_dtype = DataTypeToBkclType<device>(dtype);
    if (bkcl_dtype == kBkclUnsupported) {
      return false;
    }
    *backend_dtype = bkcl_dtype;
    return true;
  }

  static bool ToBackendRedOp(ReductionOpType red_op,
                             typename Api::RedOp* backend_op) {
    if (red_op == ReductionOpType::kAvg) {
      // XCCL has no average reduction; reject it explicitly.
      return false;
    }
    *backend_op = RedOpToBkclOp(red_op);
    return true;
  }
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_TYPE_MAP_H_
