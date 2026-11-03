#ifndef INFINI_CCL_BACKENDS_CCL_HCCL_CHECKS_H_
#define INFINI_CCL_BACKENDS_CCL_HCCL_CHECKS_H_

#include <hccl/hccl.h>

#include <cstdlib>
#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_HCCL(result)                                             \
  ::infini::ccl::detail::CheckHcclImpl((result), #result, __FILE__, __LINE__, \
                                       __func__)

namespace infini::ccl {

namespace detail {

// Use this for critical internal calls; forward native API results directly.
inline ReturnStatus CheckHcclImpl(HcclResult result, const char* operation,
                                  const char* file, int line,
                                  const char* function) {
  if (result != HCCL_SUCCESS) {
    std::cerr << "backend(hccl): `" << operation << "` failed with code "
              << result << " at " << file << ":" << line << " in `" << function
              << "`: " << HcclGetErrorString(result) << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_HCCL_CHECKS_H_
