#ifndef INFINI_CCL_DEVICES_ASCEND_CHECKS_H_
#define INFINI_CCL_DEVICES_ASCEND_CHECKS_H_

#include <acl/acl.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_ACL(result) \
  ::infini::ccl::detail::CheckAclImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckAclImpl(aclError acl_result, const char *file,
                                 int line) {
  if (acl_result != ACL_SUCCESS) {
    std::cerr << "ACL error code: " << acl_result << " at line " << line
              << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_DEVICES_ASCEND_CHECKS_H_
