#include "backends/ccl/hccl/ascend/average.h"

#include <aclnnop/aclnn_div.h>

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "backends/ccl/hccl/checks.h"
#include "constexpr_map.h"
#include "devices/ascend/checks.h"

namespace infini::ccl {

namespace {

struct AverageType {
  aclDataType acl_type;

  // `aclnnop/aclnn_div.h`: 0 = ordinary division, 1 = truncate toward zero,
  // 2 = floor. InfiniCCL uses -1 for unsupported types, never passed to ACLNN.
  int rounding_mode;
};

// CANN 9.1 division probes lose precision for large INT32/INT64 values.
// Unsigned and FP64 AVG remain disabled pending end-to-end validation.
constexpr ConstexprMap<HcclDataType, AverageType, 12> kAverageTypes{{{
    {HCCL_DATA_TYPE_INT8, {ACL_INT8, 1}},
    {HCCL_DATA_TYPE_INT16, {ACL_INT16, 1}},
    {HCCL_DATA_TYPE_INT32, {ACL_INT32, -1}},
    {HCCL_DATA_TYPE_INT64, {ACL_INT64, -1}},
    {HCCL_DATA_TYPE_UINT8, {ACL_UINT8, -1}},
    {HCCL_DATA_TYPE_UINT16, {ACL_UINT16, -1}},
    {HCCL_DATA_TYPE_UINT32, {ACL_UINT32, -1}},
    {HCCL_DATA_TYPE_UINT64, {ACL_UINT64, -1}},
    {HCCL_DATA_TYPE_FP16, {ACL_FLOAT16, 0}},
    {HCCL_DATA_TYPE_BFP16, {ACL_BF16, 0}},
    {HCCL_DATA_TYPE_FP32, {ACL_FLOAT, 0}},
    {HCCL_DATA_TYPE_FP64, {ACL_DOUBLE, -1}},
}}};

struct AverageKey {
  aclrtContext context;

  aclrtStream stream;

  void* buffer;

  size_t count;

  HcclDataType dtype;

  uint32_t nranks;

  bool operator==(const AverageKey& other) const {
    return context == other.context && stream == other.stream &&
           buffer == other.buffer && count == other.count &&
           dtype == other.dtype && nranks == other.nranks;
  }
};

}  // namespace

class HcclAveragePlan {
 public:
  explicit HcclAveragePlan(const AverageKey& key) : key_(key) {}

  HcclAveragePlan(const HcclAveragePlan&) = delete;

  HcclAveragePlan& operator=(const HcclAveragePlan&) = delete;

  ~HcclAveragePlan() {
    aclrtContext previous = nullptr;
    INFINI_CHECK_ACL(aclrtGetCurrentContext(&previous));
    INFINI_CHECK_ACL(aclrtSetCurrentContext(key_.context));
    if (submitted_) {
      // Events outlive caller-owned streams. Wait before freeing resources.
      INFINI_CHECK_ACL(aclrtSynchronizeEvent(completion_));
    }
    if (executor_ != nullptr) {
      INFINI_CHECK_ACLNN(aclDestroyAclOpExecutor(executor_));
    }
    if (tensor_ != nullptr) {
      INFINI_CHECK_ACLNN(aclDestroyTensor(tensor_));
    }
    if (scalar_ != nullptr) {
      INFINI_CHECK_ACLNN(aclDestroyScalar(scalar_));
    }
    if (workspace_ != nullptr) {
      INFINI_CHECK_ACL(aclrtFree(workspace_));
    }
    if (completion_ != nullptr) {
      INFINI_CHECK_ACL(aclrtDestroyEvent(completion_));
    }
    if (previous != nullptr) {
      INFINI_CHECK_ACL(aclrtSetCurrentContext(previous));
    }
  }

  bool Matches(const AverageKey& key) const { return key_ == key; }

  bool CanEvict(aclrtContext context) const {
    return key_.context == context && Ready();
  }

  void Initialize() {
    const auto type = kAverageTypes.at(key_.dtype);
    const int64_t length = static_cast<int64_t>(key_.count);
    const int64_t stride = 1;
    tensor_ = aclCreateTensor(&length, 1, type.acl_type, &stride, 0,
                              ACL_FORMAT_ND, &length, 1, key_.buffer);
    int64_t divisor = key_.nranks;
    float floating_divisor = static_cast<float>(key_.nranks);

    // An INT64 scalar narrows to the floating input dtype. Use FP32 so rank
    // counts such as 257 remain exact for BF16.
    scalar_ = type.rounding_mode == 0
                  ? aclCreateScalar(&floating_divisor, ACL_FLOAT)
                  : aclCreateScalar(&divisor, ACL_INT64);
    if (tensor_ == nullptr || scalar_ == nullptr) {
      INFINI_CHECK_HCCL(HCCL_E_MEMORY);
    }

    // Direct division avoids rounding a low-precision reciprocal first.
    INFINI_CHECK_ACLNN(aclnnInplaceDivModsGetWorkspaceSize(
        tensor_, scalar_, type.rounding_mode, &workspace_bytes_, &executor_));
    INFINI_CHECK_ACLNN(aclSetAclOpExecutorRepeatable(executor_));
    if (workspace_bytes_ != 0) {
      INFINI_CHECK_ACL(aclrtMalloc(&workspace_, workspace_bytes_,
                                   ACL_MEM_MALLOC_HUGE_FIRST));
    }
    INFINI_CHECK_ACL(aclrtCreateEventExWithFlag(
        &completion_, ACL_EVENT_CAPTURE_STREAM_PROGRESS));
  }

  void Run() {
    INFINI_CHECK_ACLNN(aclnnInplaceDivMods(workspace_, workspace_bytes_,
                                           executor_, key_.stream));
    INFINI_CHECK_ACL(aclrtRecordEvent(completion_, key_.stream));
    submitted_ = true;
  }

 private:
  bool Ready() const {
    if (!submitted_) {
      return true;
    }

    aclrtEventRecordedStatus status{};
    INFINI_CHECK_ACL(aclrtQueryEventStatus(completion_, &status));

    return status == ACL_EVENT_RECORDED_STATUS_COMPLETE;
  }

  AverageKey key_;

  aclTensor* tensor_ = nullptr;

  aclScalar* scalar_ = nullptr;

  aclOpExecutor* executor_ = nullptr;

  void* workspace_ = nullptr;

  uint64_t workspace_bytes_ = 0;

  aclrtEvent completion_ = nullptr;

  bool submitted_ = false;
};

namespace {

class AverageCache {
 public:
  AverageCache() = default;

  AverageCache(const AverageCache&) = delete;

  AverageCache& operator=(const AverageCache&) = delete;

  HcclAveragePlan* GetOrCreate(const AverageKey& key) {
    if (auto* cached = Find(key)) {
      return cached;
    }
    EvictCompleted(key.context);
    auto plan = std::make_unique<HcclAveragePlan>(key);
    plan->Initialize();
    auto* result = plan.get();
    plans_.push_back(std::move(plan));
    return result;
  }

 private:
  HcclAveragePlan* Find(const AverageKey& key) const {
    for (const auto& plan : plans_) {
      if (plan->Matches(key)) {
        return plan.get();
      }
    }
    return nullptr;
  }

  void EvictCompleted(aclrtContext context) {
    // Keep eight warm plans when possible; never wait for in-flight plans.
    constexpr size_t kWarmPlans = 8;
    for (auto it = plans_.begin();
         plans_.size() >= kWarmPlans && it != plans_.end();) {
      if ((*it)->CanEvict(context)) {
        it = plans_.erase(it);
      } else {
        ++it;
      }
    }
  }

  std::vector<std::unique_ptr<HcclAveragePlan>> plans_;
};

}  // namespace

bool HcclAverageSupported(HcclDataType dtype) {
  return dtype >= HCCL_DATA_TYPE_INT8 && dtype <= HCCL_DATA_TYPE_BFP16 &&
         kAverageTypes.at(dtype).rounding_mode >= 0;
}

HcclResult PrepareHcclAverage(void* buffer, size_t count, HcclDataType dtype,
                              uint32_t nranks, aclrtStream stream,
                              HcclAveragePlan** output) {
  if (output == nullptr) {
    return HCCL_E_PARA;
  }

  *output = nullptr;

  if (!HcclAverageSupported(dtype)) {
    return HCCL_E_NOT_SUPPORT;
  }
  if (nranks == 0 ||
      count > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
      (buffer == nullptr && count != 0)) {
    return HCCL_E_PARA;
  }
  if (count == 0 || nranks == 1) {
    return HCCL_SUCCESS;
  }

  aclrtContext context = nullptr;
  INFINI_CHECK_ACL(aclrtGetCurrentContext(&context));
  const AverageKey key{context, stream, buffer, count, dtype, nranks};
  thread_local AverageCache cache;
  *output = cache.GetOrCreate(key);

  return HCCL_SUCCESS;
}

HcclResult RunHcclAverage(HcclAveragePlan* plan) {
  if (plan == nullptr) {
    return HCCL_E_PARA;
  }

  plan->Run();

  return HCCL_SUCCESS;
}

}  // namespace infini::ccl
