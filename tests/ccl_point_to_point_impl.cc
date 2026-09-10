#include <cstdlib>
#include <iostream>
#include <memory>

#include "backends/ccl/common/impl/recv.h"
#include "backends/ccl/common/impl/send.h"

namespace infini::ccl {

struct FakeMcclApi {
  static constexpr BackendType kBackendType = BackendType::kMccl;

  using Comm = int;
  using DataType = int;
  using Result = int;
  using Stream = void*;

  static constexpr Result kSuccess = 0;
  static constexpr Result kFailure = 1;

  struct Call {
    const void* buffer = nullptr;
    size_t count = 0;
    DataType data_type = 0;
    int peer = -1;
    Comm comm = 0;
    Stream stream = nullptr;
  };

  static Call send_call;
  static Call recv_call;
  static Result next_result;
  static int destroy_count;

  static void Reset() {
    send_call = {};
    recv_call = {};
    next_result = kSuccess;
  }

  static ReturnStatus Check(Result result) {
    return result == kSuccess ? ReturnStatus::kSuccess
                              : ReturnStatus::kSystemError;
  }

  static Result CommDestroy(Comm) {
    ++destroy_count;
    return kSuccess;
  }

  static Result Send(const void* buffer, size_t count, DataType data_type,
                     int peer, Comm comm, Stream stream) {
    send_call = {buffer, count, data_type, peer, comm, stream};
    return next_result;
  }

  static Result Recv(void* buffer, size_t count, DataType data_type, int peer,
                     Comm comm, Stream stream) {
    recv_call = {buffer, count, data_type, peer, comm, stream};
    return next_result;
  }
};

FakeMcclApi::Call FakeMcclApi::send_call{};
FakeMcclApi::Call FakeMcclApi::recv_call{};
FakeMcclApi::Result FakeMcclApi::next_result = FakeMcclApi::kSuccess;
int FakeMcclApi::destroy_count = 0;

template <>
struct CclApi<BackendType::kMccl, Device::Type::kMetax> : FakeMcclApi {};

template <>
struct CclTypeMap<BackendType::kMccl, Device::Type::kMetax> {
  static bool ToBackendDataType(DataType data_type, int* backend_data_type) {
    if (data_type == DataType::kUInt16) {
      return false;
    }
    *backend_data_type = 100 + static_cast<int>(data_type);
    return true;
  }
};

using Api = CclApi<BackendType::kMccl, Device::Type::kMetax>;
using CommInstance = CclCommInstance<Api>;
using SendUnderTest = CclSendImpl<BackendType::kMccl, Device::Type::kMetax>;
using RecvUnderTest = CclRecvImpl<BackendType::kMccl, Device::Type::kMetax>;

std::unique_ptr<CommInstance> MakeMcclBackend(int handle = 41) {
  auto instance = std::make_unique<CommInstance>();
  instance->handle = handle;
  return instance;
}

bool TestSendForwardsArguments() {
  FakeMcclApi::Reset();
  Communicator comm(Device::Type::kMetax, 0);
  comm.set_intra_comm(MakeMcclBackend());

  float buffer = 7.0f;
  void* stream = reinterpret_cast<void*>(0x1234);
  auto status = SendUnderTest::Apply(&buffer, 9, DataType::kFloat32, 3, &comm,
                                     stream);
  const auto& call = FakeMcclApi::send_call;

  return status == ReturnStatus::kSuccess && call.buffer == &buffer &&
         call.count == 9 &&
         call.data_type == 100 + static_cast<int>(DataType::kFloat32) &&
         call.peer == 3 && call.comm == 41 && call.stream == stream;
}

bool TestRecvForwardsArguments() {
  FakeMcclApi::Reset();
  Communicator comm(Device::Type::kMetax, 0);
  comm.set_intra_comm(MakeMcclBackend(42));

  float buffer = 0.0f;
  void* stream = reinterpret_cast<void*>(0x5678);
  auto status = RecvUnderTest::Apply(&buffer, 11, DataType::kInt32, 2, &comm,
                                     stream);
  const auto& call = FakeMcclApi::recv_call;

  return status == ReturnStatus::kSuccess && call.buffer == &buffer &&
         call.count == 11 &&
         call.data_type == 100 + static_cast<int>(DataType::kInt32) &&
         call.peer == 2 && call.comm == 42 && call.stream == stream;
}

bool TestUnsupportedDataTypeDoesNotCallBackend() {
  FakeMcclApi::Reset();
  Communicator comm(Device::Type::kMetax, 0);
  comm.set_intra_comm(MakeMcclBackend());

  float buffer = 0.0f;
  return SendUnderTest::Apply(&buffer, 1, DataType::kUInt16, 1, &comm,
                              nullptr) == ReturnStatus::kNotSupported &&
         RecvUnderTest::Apply(&buffer, 1, DataType::kUInt16, 1, &comm,
                              nullptr) == ReturnStatus::kNotSupported &&
         FakeMcclApi::send_call.buffer == nullptr &&
         FakeMcclApi::recv_call.buffer == nullptr;
}

bool TestBackendErrorIsPropagated() {
  FakeMcclApi::Reset();
  FakeMcclApi::next_result = FakeMcclApi::kFailure;
  Communicator comm(Device::Type::kMetax, 0);
  comm.set_intra_comm(MakeMcclBackend());

  float buffer = 0.0f;
  return SendUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &comm,
                              nullptr) == ReturnStatus::kSystemError &&
         RecvUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &comm,
                              nullptr) == ReturnStatus::kSystemError;
}

bool TestInvalidCommunicatorState() {
  FakeMcclApi::Reset();
  float buffer = 0.0f;

  Communicator empty(Device::Type::kMetax, 0);
  if (SendUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &empty,
                           nullptr) != ReturnStatus::kInternalError ||
      RecvUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &empty,
                           nullptr) != ReturnStatus::kInternalError) {
    return false;
  }

  Communicator wrong_device(Device::Type::kCpu, 0);
  wrong_device.set_intra_comm(MakeMcclBackend());
  if (SendUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &wrong_device,
                           nullptr) != ReturnStatus::kInternalError ||
      RecvUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &wrong_device,
                           nullptr) != ReturnStatus::kInternalError) {
    return false;
  }

  Communicator empty_handle(Device::Type::kMetax, 0);
  empty_handle.set_intra_comm(MakeMcclBackend(0));
  return SendUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &empty_handle,
                              nullptr) == ReturnStatus::kInternalError &&
         RecvUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, &empty_handle,
                              nullptr) == ReturnStatus::kInternalError;
}

bool TestNullAndWrongBackend() {
  FakeMcclApi::Reset();
  float buffer = 0.0f;
  if (SendUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, nullptr,
                           nullptr) != ReturnStatus::kInternalError ||
      RecvUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1, nullptr,
                           nullptr) != ReturnStatus::kInternalError) {
    return false;
  }

  auto backend = std::make_unique<BackendCommInstance>();
  backend->type = BackendType::kNccl;
  Communicator wrong_backend(Device::Type::kMetax, 0);
  wrong_backend.set_intra_comm(std::move(backend));
  return SendUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1,
                              &wrong_backend, nullptr) ==
             ReturnStatus::kInternalError &&
         RecvUnderTest::Apply(&buffer, 1, DataType::kFloat32, 1,
                              &wrong_backend, nullptr) ==
             ReturnStatus::kInternalError;
}

bool TestCommInstanceDestroysHandle() {
  FakeMcclApi::destroy_count = 0;
  {
    Communicator comm(Device::Type::kMetax, 0);
    comm.set_intra_comm(MakeMcclBackend());
  }
  return FakeMcclApi::destroy_count == 1;
}

bool RunTest(const char* name, bool (*test)()) {
  if (test()) {
    return true;
  }
  std::cerr << "FAILED: " << name << std::endl;
  return false;
}

}  // namespace infini::ccl

int main() {
  using namespace infini::ccl;

  bool passed = true;
  passed = RunTest("send forwards arguments", TestSendForwardsArguments) &&
           passed;
  passed = RunTest("recv forwards arguments", TestRecvForwardsArguments) &&
           passed;
  passed = RunTest("unsupported datatype",
                   TestUnsupportedDataTypeDoesNotCallBackend) &&
           passed;
  passed = RunTest("backend error propagation", TestBackendErrorIsPropagated) &&
           passed;
  passed = RunTest("invalid communicator state", TestInvalidCommunicatorState) &&
           passed;
  passed = RunTest("null and wrong backend", TestNullAndWrongBackend) && passed;
  passed = RunTest("communicator destruction", TestCommInstanceDestroysHandle) &&
           passed;
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
