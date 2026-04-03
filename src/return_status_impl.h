#ifndef INFINI_CCL_RETURN_STATUS_IMPL_H_
#define INFINI_CCL_RETURN_STATUS_IMPL_H_

#include "return_status.h"

namespace infini::ccl {

using ReturnStatus = ::infiniResult_t;

constexpr ReturnStatus kSuccess = infiniSuccess;
constexpr ReturnStatus kUnhandledError = infiniUnhandledError;
constexpr ReturnStatus kSystemError = infiniSystemError;
constexpr ReturnStatus kInternalError = infiniInternalError;
constexpr ReturnStatus kInvalidArgument = infiniInvalidArgument;
constexpr ReturnStatus kInvalidUsage = infiniInvalidUsage;
constexpr ReturnStatus kRemoteError = infiniRemoteError;
constexpr ReturnStatus kInProgress = infiniInProgress;
constexpr ReturnStatus kNumResults = infiniNumResults;

} // namespace infini::ccl

#endif // INFINI_CCL_RETURN_STATUS_IMPL_H_
