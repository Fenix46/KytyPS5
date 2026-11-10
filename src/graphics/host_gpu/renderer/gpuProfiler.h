#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_GPUPROFILER_H_

// Diagnostics only: GPU time per draw/dispatch from timestamp queries, summed per shader set and
// printed every 10 s ("[gpu]" lines). Active only with the perf overlay and host query reset.

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandScheduler;

namespace GpuProfiler {

void Init(vk::Device device, float period_ns);
[[nodiscard]] bool Active();
// Writes the start timestamp; returns a slot for End(), or -1.
int  Begin(vk::CommandBuffer command, CommandScheduler& scheduler, uint64_t key, uint64_t a,
           uint64_t b, uint32_t kind, uint32_t work, uint32_t width = 0, uint32_t height = 0);
void End(vk::CommandBuffer command, int slot);
// Reads finished timestamps (never waits) and prints the report when it is due.
void Poll();

} // namespace GpuProfiler
} // namespace Libs::Graphics

#endif
