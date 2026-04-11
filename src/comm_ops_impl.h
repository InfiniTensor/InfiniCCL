#ifndef INFINI_CCL_COMM_OPS_IMPL_H_
#define INFINI_CCL_COMM_OPS_IMPL_H_

#include <cstdint>

#include "comm_ops.h"

namespace infini::ccl {

enum class ReductionOpType : int8_t {
  kSum = infiniSum,
  kProd = infiniProd,
  kMax = infiniMax,
  kMin = infiniMin,
  kAvg = infiniAvg,
  kNumRedOps = infiniNumOps,
};

} // namespace infini::ccl

#endif // INFINI_CCL_COMM_OPS_IMPL_H_
