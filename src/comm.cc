#include "comm.h"
#include "base/init.h"
#include "comm_ops_impl.h"
#include "operation.h"

infiniResult_t infiniInit(int *argc, char ***argv) {
  return static_cast<infiniResult_t>(infini::ccl::Init::Call(argc, argv));
}
