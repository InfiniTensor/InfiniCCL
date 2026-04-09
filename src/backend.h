#ifndef INFINI_CCL_BACKEND_H_
#define INFINI_CCL_BACKEND_H_

#include <cstdint>

namespace infini::ccl {

enum class BackendType : int8_t {
  kOmpi = 0,
  kGloo = 1,
  kNccl = 2,
  kMccl = 3,
  kRccl = 4,
  kCncl = 5,
  kHccl = 6,
  kCount
};

} // namespace infini::ccl

#endif // INFINI_CCL_BACKEND_H_
