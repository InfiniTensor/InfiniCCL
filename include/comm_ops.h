#ifndef INFINI_CCL_COMM_OPS_H_
#define INFINI_CCL_COMM_OPS_H_

#include <cstddef>

#include "comm.h"
#include "data_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// --- Reduction Operations ---
typedef enum {
  infiniSum = 0,
  infiniProd = 1,
  infiniMax = 2,
  infiniMin = 3,
  infiniAvg = 4,
  infiniNumOps = 5
} infiniRedOp_t;

infiniResult_t infiniAllReduce(const void *sendbuff, void *recvbuff,
                               size_t count, infiniDataType_t datatype,
                               infiniRedOp_t op, infiniComm_t comm,
                               void *stream);
infiniResult_t infiniBroadcast(const void *sendbuff, void *recvbuff,
                               size_t count, infiniDataType_t datatype,
                               int root, infiniComm_t comm, void *stream);
infiniResult_t infiniReduce(const void *sendbuff, void *recvbuff, size_t count,
                            infiniDataType_t datatype, infiniRedOp_t op,
                            int root, infiniComm_t comm, void *stream);
infiniResult_t infiniRecv(void *recvbuff, size_t count,
                          infiniDataType_t datatype, int peer,
                          infiniComm_t comm, void *stream);
infiniResult_t infiniSend(const void *sendbuff, size_t count,
                          infiniDataType_t datatype, int peer,
                          infiniComm_t comm, void *stream);
infiniResult_t infiniAllGather(const void *sendbuff, void *recvbuff,
                               size_t count, infiniDataType_t datatype,
                               infiniComm_t comm, void *stream);
infiniResult_t infiniReduceScatter(const void *sendbuff, void *recvbuff,
                                   size_t count, infiniDataType_t datatype,
                                   infiniRedOp_t op, infiniComm_t comm,
                                   void *stream);

#ifdef __cplusplus
}
#endif

#endif // INFINI_CCL_COMM_OPS_H_
