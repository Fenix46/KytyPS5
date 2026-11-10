#ifndef EMULATOR_SRC_COMMON_DEBUGSAMPLER_H_
#define EMULATOR_SRC_COMMON_DEBUGSAMPLER_H_

// Diagnostics only: samples one thread every ~1 ms and prints, every 10 s, the functions it was
// most often in (self) and under (inclusive). Addresses are exe-relative RVAs of function starts,
// to be resolved against the linker map.

namespace Common::DebugSampler {

// Starts sampling the calling thread (once). No-op when diagnostics are disabled or off Windows.
void StartForCurrentThread(const char* label);

} // namespace Common::DebugSampler

#endif
