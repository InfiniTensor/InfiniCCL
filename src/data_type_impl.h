#ifndef INFINI_CCL_DATA_TYPE_IMPL_H_
#define INFINI_CCL_DATA_TYPE_IMPL_H_

#include <cstddef>
#include <string_view>

#include "constexpr_map.h"
#include "data_type.h"

namespace infini::ccl {

using DataType = ::infiniDataType_t;

constexpr DataType kChar = infiniChar;
constexpr DataType kInt8 = infiniInt8;
constexpr DataType kInt16 = infiniInt16;
constexpr DataType kInt32 = infiniInt32;
constexpr DataType kInt64 = infiniInt64;
constexpr DataType kUInt8 = infiniUInt8;
constexpr DataType kUInt16 = infiniUInt16;
constexpr DataType kUInt32 = infiniUInt32;
constexpr DataType kUInt64 = infiniUInt64;
constexpr DataType kFloat16 = infiniFloat16;
constexpr DataType kBFloat16 = infiniBFloat16;
constexpr DataType kFloat32 = infiniFloat32;
constexpr DataType kFloat64 = infiniFloat64;
constexpr DataType kNumTypes = infiniNumTypes;

constexpr ConstexprMap<DataType, std::size_t, 12> kDataTypeToSize{{{
    {kInt8, 1},
    {kInt16, 2},
    {kInt32, 4},
    {kInt64, 8},
    {kUInt8, 1},
    {kUInt16, 2},
    {kUInt32, 4},
    {kUInt64, 8},
    {kFloat16, 2},
    {kBFloat16, 2},
    {kFloat32, 4},
    {kFloat64, 8},
}}};

constexpr ConstexprMap<DataType, std::string_view, 12> kDataTypeToDesc{{{
    {kInt8, "int8"},
    {kInt16, "int16"},
    {kInt32, "int32"},
    {kInt64, "int64"},
    {kUInt8, "uint8"},
    {kUInt16, "uint16"},
    {kUInt32, "uint32"},
    {kUInt64, "uint64"},
    {kFloat16, "float16"},
    {kBFloat16, "bfloat16"},
    {kFloat32, "float32"},
    {kFloat64, "float64"},
}}};

constexpr ConstexprMap<std::string_view, DataType, 12> kStringToDataType{{{
    {"int8", kInt8},
    {"int16", kInt16},
    {"int32", kInt32},
    {"int64", kInt64},
    {"uint8", kUInt8},
    {"uint16", kUInt16},
    {"uint32", kUInt32},
    {"uint64", kUInt64},
    {"float16", kFloat16},
    {"bfloat16", kBFloat16},
    {"float32", kFloat32},
    {"float64", kFloat64},
}}};

} // namespace infini::ccl

#endif // INFINI_CCL_DATA_TYPE_IMPL_H_
