#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_REDUCTION_OP_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_REDUCTION_OP_H_

#include <hccl/hccl.h>

namespace infini::ccl {

// Preserve AVG through the common CCL adapter; HCCL itself only sees SUM.
struct HcclReductionOp {
  HcclReduceOp native_op;
  bool average;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_REDUCTION_OP_H_
