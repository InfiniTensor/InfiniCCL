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
infiniResult_t infiniFinalize(void);

// Rank/Size Query
infiniResult_t infiniGetRank(int *rank);
infiniResult_t infiniGetSize(int *size);

// Communicator Management
infiniResult_t infiniCommInitAll(infiniComm_t *comm, int ndev,
                                 const int *devlist);
infiniResult_t infiniCommDestroy(infiniComm_t comm);

// --- Reduction Operations ---
typedef enum {
  infiniSum = 0,
  infiniProd = 1,
  infiniMax = 2,
  infiniMin = 3,
  infiniAvg = 4,
  infiniNumOps
} infiniRedOp_t;

// Collective Communication Functions
infiniResult_t infiniAllReduce(const void *sendbuff, void *recvbuff,
                               size_t count, infiniDataType_t datatype,
                               infiniRedOp_t op, infiniComm_t comm,
                               void *stream);

#ifdef __cplusplus
}
#endif

#endif // INFINI_CCL_COMM_H_
