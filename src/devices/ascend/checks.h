#ifndef INFINI_CCL_DEVICES_ASCEND_CHECKS_H_
#define INFINI_CCL_DEVICES_ASCEND_CHECKS_H_

#include <acl/acl.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_ACL(result)                                             \
  ::infini::ccl::detail::CheckAscendImpl((result), "ACL", #result, __FILE__, \
                                         __LINE__, __func__)

#define INFINI_CHECK_ACLNN(result)                                             \
  ::infini::ccl::detail::CheckAscendImpl((result), "ACLNN", #result, __FILE__, \
                                         __LINE__, __func__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckAscendImpl(int64_t result, const char* api,
                                    const char* operation, const char* file,
                                    int line, const char* function) {
  if (result != 0) {
    std::cerr << api << ": `" << operation << "` failed with code " << result
              << " at " << file << ":" << line << " in `" << function << "`";
    if (const char* detail = aclGetRecentErrMsg()) {
      std::cerr << " " << detail;
    }
    std::cerr << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ASCEND_CHECKS_H_
