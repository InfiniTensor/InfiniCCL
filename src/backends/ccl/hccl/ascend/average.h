#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_ASCEND_AVERAGE_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_ASCEND_AVERAGE_H_

#include <acl/acl.h>
#include <hccl/hccl.h>

#include <cstddef>
#include <cstdint>

namespace infini::ccl {

class HcclAveragePlan;

bool HcclAverageSupported(HcclDataType dtype);

HcclResult PrepareHcclAverage(void* buffer, size_t count, HcclDataType dtype,
                              uint32_t nranks, aclrtStream stream,
                              HcclAveragePlan** plan);

HcclResult RunHcclAverage(HcclAveragePlan* plan);

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_ASCEND_AVERAGE_H_
