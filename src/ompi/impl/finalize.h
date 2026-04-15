#ifndef INFINI_CCL_OMPI_IMPL_FINALIZE_H_
#define INFINI_CCL_OMPI_IMPL_FINALIZE_H_

#include "base/finalize.h"
#include "ompi/checks.h"

namespace infini::ccl {

template <Device::Type device_type>
class FinalizeImpl<BackendType::kOmpi, device_type> {
public:
  static ReturnStatus Apply() {
    int finalized;
    INFINI_CHECK_MPI(MPI_Finalized(&finalized));
    return ReturnStatus::kSuccess;
  }
};

template <>
struct BackendEnabled<Finalize, BackendType::kOmpi> : std::true_type {};

} // namespace infini::ccl

#endif // INFINI_CCL_OMPI_IMPL_FINALIZE_H_
