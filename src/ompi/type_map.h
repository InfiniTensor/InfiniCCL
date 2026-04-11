#ifndef INFINI_CCL_OMPI_TYPE_MAPPING_H_
#define INFINI_CCL_OMPI_TYPE_MAPPING_H_

#include <mpi.h>

#include "data_type_impl.h"

namespace infini::ccl {

static const ConstexprMap<DataType, MPI_Datatype, 12> kOmpiTypeMap{{{
    {kInt8, MPI_INT8_T},
    {kInt16, MPI_INT16_T},
    {kInt32, MPI_INT32_T},
    {kInt64, MPI_INT64_T},
    {kUInt8, MPI_UINT8_T},
    {kUInt16, MPI_UINT16_T},
    {kUInt32, MPI_UINT32_T},
    {kUInt64, MPI_UINT64_T},
    {kFloat32, MPI_FLOAT},
    {kFloat64, MPI_DOUBLE},
    {kFloat16, MPI_BYTE},
    {kBFloat16, MPI_BYTE},
}}};

inline MPI_Datatype DataTypeToOmpiType(DataType dtype) {
  return kOmpiTypeMap.at(dtype);
}

} // namespace infini::ccl

#endif // INFINI_CCL_OMPI_TYPE_MAPPING_H_
