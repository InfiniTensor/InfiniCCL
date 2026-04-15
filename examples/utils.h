#ifndef INFINI_CCL_EXAMPLES_UTILS_H_
#define INFINI_CCL_EXAMPLES_UTILS_H_

// Simple check macro for the C-API
#define CHECK_INFINI(cmd)                                                      \
  do {                                                                         \
    infiniResult_t res = (cmd);                                                \
    if (res != infiniSuccess) {                                                \
      std::cerr << "[InfiniCCL Error] received error code " << res             \
                << " at line " << __LINE__ << std::endl;                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#endif
