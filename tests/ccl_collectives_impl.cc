#include <cstdlib>
#include <iostream>
#include <memory>

#include "backends/ccl/common/impl/collectives.h"

namespace infini::ccl {

struct FakeMcclApi {
  static constexpr BackendType kBackendType = BackendType::kMccl;
  using Comm = int;
  using DataType = int;
  using RedOp = int;
  using Result = int;
  using Stream = void*;
  static constexpr Result kSuccess = 0;
  static constexpr Result kFailure = 1;
  struct Call { const void* send; void* recv; size_t count; DataType type; RedOp op; int root; Comm comm; Stream stream; };
  static Call call;
  static Result next_result;
  static Result CommDestroy(Comm) { return kSuccess; }
  static void Reset() { call = {}; next_result = kSuccess; }
  static ReturnStatus Check(Result result) { return result == kSuccess ? ReturnStatus::kSuccess : ReturnStatus::kSystemError; }
  static Result Broadcast(const void* s, void* r, size_t n, DataType t, int root, Comm c, Stream x) { call = {s,r,n,t,0,root,c,x}; return next_result; }
  static Result Reduce(const void* s, void* r, size_t n, DataType t, RedOp o, int root, Comm c, Stream x) { call = {s,r,n,t,o,root,c,x}; return next_result; }
  static Result ReduceScatter(const void* s, void* r, size_t n, DataType t, RedOp o, Comm c, Stream x) { call = {s,r,n,t,o,-1,c,x}; return next_result; }
  static Result AllGather(const void* s, void* r, size_t n, DataType t, Comm c, Stream x) { call = {s,r,n,t,0,-1,c,x}; return next_result; }
  static Result Gather(const void* s, void* r, size_t n, DataType t, int root, Comm c, Stream x) { call = {s,r,n,t,0,root,c,x}; return next_result; }
  static Result Scatter(const void* s, void* r, size_t n, DataType t, int root, Comm c, Stream x) { call = {s,r,n,t,0,root,c,x}; return next_result; }
  static Result AllToAll(const void* s, void* r, size_t n, DataType t, Comm c, Stream x) { call = {s,r,n,t,0,-1,c,x}; return next_result; }
};
FakeMcclApi::Call FakeMcclApi::call{};
FakeMcclApi::Result FakeMcclApi::next_result = FakeMcclApi::kSuccess;

template <> struct CclApi<BackendType::kMccl, Device::Type::kMetax> : FakeMcclApi {};
template <> struct CclTypeMap<BackendType::kMccl, Device::Type::kMetax> {
  static bool ToBackendDataType(DataType type, int* result) { if (type == DataType::kUInt16) return false; *result = 100 + static_cast<int>(type); return true; }
  static bool ToBackendRedOp(ReductionOpType op, int* result) { if (op == ReductionOpType::kNumRedOps) return false; *result = 200 + static_cast<int>(op); return true; }
};
using Api = CclApi<BackendType::kMccl, Device::Type::kMetax>;
using Instance = CclCommInstance<Api>;
std::unique_ptr<Instance> MakeComm() { auto result = std::make_unique<Instance>(); result->handle = 7; return result; }

bool ForwardAllCollectives() {
  FakeMcclApi::Reset();
  Communicator comm(Device::Type::kMetax, 0); comm.set_intra_comm(MakeComm());
  int send = 1, recv = 0; void* stream = reinterpret_cast<void*>(0x1234);
  using B = CclBroadcastImpl<BackendType::kMccl, Device::Type::kMetax>;
  using R = CclReduceImpl<BackendType::kMccl, Device::Type::kMetax>;
  using S = CclReduceScatterImpl<BackendType::kMccl, Device::Type::kMetax>;
  using G = CclAllGatherImpl<BackendType::kMccl, Device::Type::kMetax>;
  using H = CclGatherImpl<BackendType::kMccl, Device::Type::kMetax>;
  using T = CclScatterImpl<BackendType::kMccl, Device::Type::kMetax>;
  using A = CclAllToAllImpl<BackendType::kMccl, Device::Type::kMetax>;
  if (B::Apply(&send,&recv,3,DataType::kFloat32,1,&comm,stream) != ReturnStatus::kSuccess || FakeMcclApi::call.root != 1) return false;
  if (R::Apply(&send,&recv,4,DataType::kInt32,ReductionOpType::kSum,0,&comm,stream) != ReturnStatus::kSuccess || FakeMcclApi::call.op != 200) return false;
  if (S::Apply(&send,&recv,5,DataType::kFloat32,ReductionOpType::kMax,&comm,stream) != ReturnStatus::kSuccess) return false;
  if (G::Apply(&send,&recv,6,DataType::kFloat32,&comm,stream) != ReturnStatus::kSuccess) return false;
  if (H::Apply(&send,&recv,7,DataType::kFloat32,0,&comm,stream) != ReturnStatus::kSuccess) return false;
  if (T::Apply(&send,&recv,8,DataType::kFloat32,0,&comm,stream) != ReturnStatus::kSuccess) return false;
  return A::Apply(&send,&recv,9,DataType::kFloat32,&comm,stream) == ReturnStatus::kSuccess;
}

bool RejectUnsupportedAndPropagateError() {
  FakeMcclApi::Reset(); Communicator comm(Device::Type::kMetax, 0); comm.set_intra_comm(MakeComm());
  int send = 1, recv = 0;
  using B = CclBroadcastImpl<BackendType::kMccl, Device::Type::kMetax>;
  if (B::Apply(&send,&recv,1,DataType::kUInt16,0,&comm,nullptr) != ReturnStatus::kNotSupported || FakeMcclApi::call.send != nullptr) return false;
  FakeMcclApi::next_result = FakeMcclApi::kFailure;
  return B::Apply(&send,&recv,1,DataType::kFloat32,0,&comm,nullptr) == ReturnStatus::kSystemError;
}

bool RejectInvalidCommunicator() {
  int send = 1, recv = 0;
  using A = CclAllToAllImpl<BackendType::kMccl, Device::Type::kMetax>;
  Communicator comm(Device::Type::kMetax, 0);
  return A::Apply(&send,&recv,1,DataType::kFloat32,&comm,nullptr) == ReturnStatus::kInternalError;
}

bool Run(const char* name, bool (*test)()) { if (test()) return true; std::cerr << "FAILED: " << name << std::endl; return false; }
}  // namespace infini::ccl

int main() {
  using namespace infini::ccl;
  bool passed = Run("forward all collectives", ForwardAllCollectives);
  passed = Run("unsupported type and backend error", RejectUnsupportedAndPropagateError) && passed;
  passed = Run("invalid communicator", RejectInvalidCommunicator) && passed;
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
