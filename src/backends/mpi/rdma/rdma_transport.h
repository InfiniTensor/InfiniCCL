#ifndef INFINI_CCL_BACKENDS_RDMA_RDMA_TRANSPORT_H_
#define INFINI_CCL_BACKENDS_RDMA_RDMA_TRANSPORT_H_

#include <infiniband/verbs.h>
#include <mpi.h>

#include <cstddef>
#include <memory>

#include "device.h"
#include "return_status_impl.h"

namespace infini::ccl {

// RDMA transport attribute of the OpenMPI backend. Device-memory messages use
// a verbs RC connection directly when both peers support it; everything else
// falls back to the caller's host staging path. The class is internal to the
// library: applications keep using `infinicclSend`/`infinicclRecv`.
//
// Calls belonging to one communicator must be serialized by the caller, the
// same assumption the MPI staging path already makes.
class RdmaTransport {
 public:
  RdmaTransport();

  ~RdmaTransport();

  RdmaTransport(const RdmaTransport &) = delete;

  RdmaTransport &operator=(const RdmaTransport &) = delete;

  // The nested name is public so the implementation file can define helpers
  // over it; the definition and all members stay in the `.cc` file.
  struct Impl;

  // Creates the verbs resources and exchanges QP metadata with the peer over
  // `comm`. The transport stays inert (`SupportRdma` always returns false)
  // when the rank count is not two or no RDMA device is usable, so callers
  // keep falling back to host staging without extra branching.
  void Initialize(int rank, int nranks, MPI_Comm comm);

  // End-to-end path decision for one message. The send side runs the local
  // pre-check, proposes the RDMA path to the peer and waits for its
  // confirmation; the receive side waits for the proposal, validates its own
  // buffer, posts the receive work request and only then confirms. Every
  // fallback is recorded in the transport statistics. Returns false whenever
  // either side cannot use RDMA; the caller must then use host staging.
  bool SupportRdma(void *buffer, size_t bytes, int peer, Device::Type device,
                   void *stream, bool send);

  // Executes the RDMA send path after `SupportRdma` returned true for the
  // same message. Failures after the peer confirmed are hard errors; the
  // transport is torn down so later messages use host staging.
  ReturnStatus Send(void *buffer, size_t bytes, int peer, void *stream);

  // Waits for the receive completion after `SupportRdma` returned true; the
  // receive work request is already posted there.
  ReturnStatus Recv(void *buffer, size_t bytes, int peer);

  // Releases all resources and prints the transport statistics.
  void Destroy();

 private:
  std::unique_ptr<Impl> impl_;
};
}  // namespace infini::ccl

#endif  // INFINI_CCL_BACKENDS_RDMA_RDMA_TRANSPORT_H_
