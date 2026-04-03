#ifndef INFINI_CCL_DATA_TYPE_H_
#define INFINI_CCL_DATA_TYPE_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  infiniChar = 0,
  infiniInt8 = 0,
  infiniInt16 = 1,
  infiniInt = 2,
  infiniInt32 = 2,
  infiniInt64 = 3,
  infiniUInt8 = 4,
  infiniUInt16 = 5,
  infiniUInt32 = 6,
  infiniUInt64 = 7,
  infiniHalf = 8,
  infiniFloat16 = 8,
  infiniBFloat16 = 9,
  infiniFloat = 10,
  infiniFloat32 = 10,
  infiniDouble = 11,
  infiniFloat64 = 11,
  infiniNumTypes = 12,
} infiniDataType_t;

#ifdef __cplusplus
}
#endif

#endif // INFINI_CCL_DATA_TYPE_H_
