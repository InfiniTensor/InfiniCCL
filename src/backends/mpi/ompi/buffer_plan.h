#ifndef INFINI_CCL_BACKENDS_MPI_OMPI_BUFFER_PLAN_H_
#define INFINI_CCL_BACKENDS_MPI_OMPI_BUFFER_PLAN_H_

#include <cstdlib>
#include <memory>

#include "runtime.h"

namespace infini::ccl {

namespace detail {

using HostBuffer = std::unique_ptr<void, decltype(&std::free)>;

inline HostBuffer AllocateHostBuffer(std::size_t bytes) {
  return HostBuffer(std::malloc(bytes), &std::free);
}

template <Device::Type kDev>
bool IsDeviceBuffer(const void *ptr) {
  return ptr != nullptr && GetMemorySpace<kDev>(ptr) == MemorySpace::kDevice;
}

}  // namespace detail

template <Device::Type kDev>
class MpiBufferPlan {
 public:
  using Rt = Runtime<kDev>;

  enum class Mode {
    kPair,
    kAlias,
  };

  ReturnStatus Prepare(const void *send_buff, void *recv_buff,
                       std::size_t send_bytes, std::size_t recv_bytes,
                       bool allow_direct, void *stream, Mode mode) {
    const bool send_active = send_bytes != 0;
    const bool recv_active = recv_bytes != 0;
    if (!send_active && !recv_active) {
      return ReturnStatus::kSuccess;
    }

    user_recv_buff_ = recv_buff;
    recv_bytes_ = recv_bytes;

    const bool send_device =
        send_active && detail::IsDeviceBuffer<kDev>(send_buff);
    const bool recv_device =
        recv_active && detail::IsDeviceBuffer<kDev>(recv_buff);
    recv_is_device_ = recv_device;

    if (mode == Mode::kPair) {
      return PreparePair(send_buff, recv_buff, send_bytes, recv_bytes,
                         allow_direct, stream, send_active, recv_active,
                         send_device, recv_device);
    }

    return PrepareAlias(recv_buff, recv_bytes, allow_direct, stream,
                        send_active, recv_device);
  }

  ReturnStatus Commit() {
    if (!host_recv_buff_) {
      return ReturnStatus::kSuccess;
    }

    CHECK_STATUS(Rt, Rt::Memcpy(user_recv_buff_, host_recv_buff_.get(),
                                recv_bytes_, Rt::MemcpyHostToDevice));
    return ReturnStatus::kSuccess;
  }

  const void *send_buff() const { return mpi_send_buff_; }

  void *recv_buff() const { return mpi_recv_buff_; }

  bool direct() const { return direct_; }

 private:
  ReturnStatus PreparePair(const void *send_buff, void *recv_buff,
                           std::size_t send_bytes, std::size_t recv_bytes,
                           bool allow_direct, void *stream, bool send_active,
                           bool recv_active, bool send_device,
                           bool recv_device) {
    direct_ = allow_direct && (!send_active || send_device) &&
              (!recv_active || recv_device);
    if (direct_) {
      if (send_device) {
        CHECK_STATUS(Rt, Rt::StreamSynchronize(
                             static_cast<typename Rt::Stream>(stream)));
      }

      mpi_send_buff_ = send_buff;
      mpi_recv_buff_ = recv_buff;
      return ReturnStatus::kSuccess;
    }

    if (send_device) {
      CHECK_STATUS(
          Rt, Rt::StreamSynchronize(static_cast<typename Rt::Stream>(stream)));
      host_send_buff_ = detail::AllocateHostBuffer(send_bytes);
      if (!host_send_buff_) {
        return ReturnStatus::kSystemError;
      }

      CHECK_STATUS(Rt, Rt::Memcpy(host_send_buff_.get(), send_buff, send_bytes,
                                  Rt::MemcpyDeviceToHost));
      mpi_send_buff_ = host_send_buff_.get();
    } else {
      mpi_send_buff_ = send_buff;
    }

    if (recv_device) {
      host_recv_buff_ = detail::AllocateHostBuffer(recv_bytes);
      if (!host_recv_buff_) {
        return ReturnStatus::kSystemError;
      }
      mpi_recv_buff_ = host_recv_buff_.get();
    } else {
      mpi_recv_buff_ = recv_buff;
    }

    return ReturnStatus::kSuccess;
  }

  ReturnStatus PrepareAlias(void *recv_buff, std::size_t recv_bytes,
                            bool allow_direct, void *stream, bool send_active,
                            bool recv_device) {
    direct_ = allow_direct && recv_device && !send_active;
    if (direct_) {
      CHECK_STATUS(
          Rt, Rt::StreamSynchronize(static_cast<typename Rt::Stream>(stream)));
      mpi_recv_buff_ = recv_buff;
      return ReturnStatus::kSuccess;
    }

    if (recv_device) {
      CHECK_STATUS(
          Rt, Rt::StreamSynchronize(static_cast<typename Rt::Stream>(stream)));
      host_recv_buff_ = detail::AllocateHostBuffer(recv_bytes);
      if (!host_recv_buff_) {
        return ReturnStatus::kSystemError;
      }

      mpi_recv_buff_ = host_recv_buff_.get();
    } else {
      mpi_recv_buff_ = recv_buff;
    }

    return ReturnStatus::kSuccess;
  }

  void *user_recv_buff_ = nullptr;
  const void *mpi_send_buff_ = nullptr;
  void *mpi_recv_buff_ = nullptr;
  detail::HostBuffer host_send_buff_{nullptr, &std::free};
  detail::HostBuffer host_recv_buff_{nullptr, &std::free};
  std::size_t recv_bytes_ = 0;
  bool recv_is_device_ = false;
  bool direct_ = false;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_MPI_OMPI_BUFFER_PLAN_H_
