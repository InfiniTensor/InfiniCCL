#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_VENDOR_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_VENDOR_H_

#if defined(INFINI_CCL_USE_RCCL)
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_VENDOR_H_
