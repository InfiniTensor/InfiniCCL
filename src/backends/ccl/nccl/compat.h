#ifndef INFINI_CCL_BACKENDS_CCL_NCCL_COMPAT_H_
#define INFINI_CCL_BACKENDS_CCL_NCCL_COMPAT_H_

#if defined(INFINI_CCL_USE_RCCL)
#include <rccl/rccl.h>
#else
#include <nccl.h>
#endif

// Denglin/vendor NCCL 2.12 (and some RCCL builds) lack Gather/Scatter entry
// points that newer InfiniCCL master expects. Detect at compile time.
#if defined(NCCL_VERSION_CODE) && !defined(INFINI_CCL_NCCL_HAS_GATHER_SCATTER)
#if NCCL_VERSION_CODE >= 22300
#define INFINI_CCL_NCCL_HAS_GATHER_SCATTER 1
#else
#define INFINI_CCL_NCCL_HAS_GATHER_SCATTER 0
#endif
#endif

#ifndef INFINI_CCL_NCCL_HAS_GATHER_SCATTER
#define INFINI_CCL_NCCL_HAS_GATHER_SCATTER 0
#endif

#endif  // INFINI_CCL_BACKENDS_CCL_NCCL_COMPAT_H_
