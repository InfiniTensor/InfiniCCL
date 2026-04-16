#ifndef INFINI_CCL_UTILS_H_
#define INFINI_CCL_UTILS_H_

#include "constexpr_map.h"
#include <iostream>

namespace infini::ccl {

class Logger {
public:
  enum class LogLevel : int8_t { kInfo, kWarning, kError, kFatal, kCount };

  static void PrintMsg(const char *msg, LogLevel level = LogLevel::kError,
                       const char *file = __FILE__, int line = __LINE__) {
    std::cerr << "[InfiniCCL " << kLogLevelToDesc.at(level) << "] " << file
              << ":" << line << " - " << msg << std::endl;
    if (level == LogLevel::kFatal) {
      std::abort();
    }
  }

private:
  static constexpr ConstexprMap<LogLevel, std::string_view,
                                static_cast<std::size_t>(LogLevel::kCount)>
      kLogLevelToDesc{{{
          {LogLevel::kInfo, "INFO"},
          {LogLevel::kWarning, "WARNING"},
          {LogLevel::kError, "ERROR"},
          {LogLevel::kFatal, "FATAL"},
      }}};
};

} // namespace infini::ccl

#endif // INFINI_CCL_UTILS_H_
