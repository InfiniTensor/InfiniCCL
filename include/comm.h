#ifndef INFINI_CCL_COMM_H_
#define INFINI_CCL_COMM_H_

#include <cstddef>

#include "data_type.h"
#include "return_status.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *infiniComm_t;

// Initialization
infiniResult_t infiniInit(int *argc, char ***argv);

#ifdef __cplusplus
}
#endif

#endif // INFINI_CCL_COMM_H_
