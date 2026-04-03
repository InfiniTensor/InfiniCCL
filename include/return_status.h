#ifndef INFINI_CCL_RETURN_STATUS_H_
#define INFINI_CCL_RETURN_STATUS_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  infiniSuccess = 0,
  infiniUnhandledError = 1,
  infiniSystemError = 2,
  infiniInternalError = 3,
  infiniInvalidArgument = 4,
  infiniInvalidUsage = 5,
  infiniRemoteError = 6,
  infiniInProgress = 7,
  infiniNumResults = 8
} infiniResult_t;

#ifdef __cplusplus
}
#endif

#endif // INFINI_CCL_RETURN_STATUS_H_
