#ifndef INFINI_CCL_DISPATCH_POLICY_H_
#define INFINI_CCL_DISPATCH_POLICY_H_

namespace infini::ccl {

// Backend selection strategy for an API implemented by several backends.
enum class Policy {
  // Highest-priority enabled backend always wins (default).
  kStatic,
  // Reserved: filter candidates by the side the API builds (world vs.
  // intra-node) before priority. For lifecycle APIs.
  kRole,
  // The communicator decides: when the priority winner is attached to
  // neither communicator slot, the best enabled carried backend runs.
  // Runtime cost: two communicator-slot reads plus a few enum compares per
  // call.
  kCarrierAware,
  // Reserved: run every enabled backend the communicator carries (e.g.
  // communicator destruction).
  kFanOut,
};

// Per-operation dispatch policy. Specialize next to the operation to opt
// out of `kStatic`.
template <typename Key>
struct DispatchPolicy {
  static constexpr Policy value = Policy::kStatic;
};

}  // namespace infini::ccl

#endif  // INFINI_CCL_DISPATCH_POLICY_H_
