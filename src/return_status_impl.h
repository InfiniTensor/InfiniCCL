#ifndef INFINI_CCL_RETURN_STATUS_IMPL_H_
#define INFINI_CCL_RETURN_STATUS_IMPL_H_

#include <cstdint>

#include "return_status.h"

namespace infini::ccl {

enum class ReturnStatus : int8_t {
  kSuccess = infiniSuccess,
  kUnhandledError = infiniUnhandledError,
  kSystemError = infiniSystemError,
  kInternalError = infiniInternalError,
  kInvalidArgument = infiniInvalidArgument,
  kInvalidUsage = infiniInvalidUsage,
  kRemoteError = infiniRemoteError,
  kInProgress = infiniInProgress,
  kNumResults = infiniNumResults,
};

} // namespace infini::ccl

#endif // INFINI_CCL_RETURN_STATUS_IMPL_H_
