#ifndef INFINI_CCL_DATA_TYPE_IMPL_H_
#define INFINI_CCL_DATA_TYPE_IMPL_H_

#include <cstddef>
#include <string_view>

#include "constexpr_map.h"
#include "data_type.h"

namespace infini::ccl {

enum class DataType : int8_t {
  kChar = infiniChar,
  kInt8 = infiniInt8,
  kInt16 = infiniInt16,
  kInt32 = infiniInt32,
  kInt64 = infiniInt64,
  kUInt8 = infiniUInt8,
  kUInt16 = infiniUInt16,
  kUInt32 = infiniUInt32,
  kUInt64 = infiniUInt64,
  kFloat16 = infiniFloat16,
  kBFloat16 = infiniBFloat16,
  kFloat32 = infiniFloat32,
  kFloat64 = infiniFloat64,
  kNumTypes = infiniNumTypes,
};

constexpr ConstexprMap<DataType, std::size_t, 12> kDataTypeToSize{{{
    {DataType::kInt8, 1},
    {DataType::kInt16, 2},
    {DataType::kInt32, 4},
    {DataType::kInt64, 8},
    {DataType::kUInt8, 1},
    {DataType::kUInt16, 2},
    {DataType::kUInt32, 4},
    {DataType::kUInt64, 8},
    {DataType::kFloat16, 2},
    {DataType::kBFloat16, 2},
    {DataType::kFloat32, 4},
    {DataType::kFloat64, 8},
}}};

constexpr ConstexprMap<DataType, std::string_view, 12> kDataTypeToDesc{{{
    {DataType::kInt8, "int8"},
    {DataType::kInt16, "int16"},
    {DataType::kInt32, "int32"},
    {DataType::kInt64, "int64"},
    {DataType::kUInt8, "uint8"},
    {DataType::kUInt16, "uint16"},
    {DataType::kUInt32, "uint32"},
    {DataType::kUInt64, "uint64"},
    {DataType::kFloat16, "float16"},
    {DataType::kBFloat16, "bfloat16"},
    {DataType::kFloat32, "float32"},
    {DataType::kFloat64, "float64"},
}}};

constexpr ConstexprMap<std::string_view, DataType, 12> kStringToDataType{{{
    {"int8", DataType::kInt8},
    {"int16", DataType::kInt16},
    {"int32", DataType::kInt32},
    {"int64", DataType::kInt64},
    {"uint8", DataType::kUInt8},
    {"uint16", DataType::kUInt16},
    {"uint32", DataType::kUInt32},
    {"uint64", DataType::kUInt64},
    {"float16", DataType::kFloat16},
    {"bfloat16", DataType::kBFloat16},
    {"float32", DataType::kFloat32},
    {"float64", DataType::kFloat64},
}}};

} // namespace infini::ccl

#endif // INFINI_CCL_DATA_TYPE_IMPL_H_
