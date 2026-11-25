#ifndef INFINI_CCL_BACKENDS_CCL_BKCL_CHECKS_H_
#define INFINI_CCL_BACKENDS_CCL_BKCL_CHECKS_H_

#include <xccl.h>

#include <iostream>

#include "return_status_impl.h"

#define INFINI_CHECK_BKCL(result) \
  ::infini::ccl::detail::CheckBkclImpl((result), __FILE__, __LINE__)

namespace infini::ccl {

namespace detail {

inline ReturnStatus CheckBkclImpl(XCCLResult_t bkcl_result, const char* file,
                                  int line) {
  if (bkcl_result != XCCL_SUCCESS) {
    std::cerr << "backend(bkcl) BKCL error code: " << bkcl_result
              << " at line " << line << " in " << file << std::endl;
    std::abort();
  }
  return ReturnStatus::kSuccess;
}

}  // namespace detail

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_CCL_BKCL_CHECKS_H_
