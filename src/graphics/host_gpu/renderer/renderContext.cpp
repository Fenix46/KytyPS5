#include "graphics/host_gpu/renderer/renderContext.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/debugStack.h"
#include "graphics/host_gpu/renderer/gpuProfiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "kernel/memory.h"
#include "graphics/presentation/videoOut.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <optional>
#include <vector>
#include <string>
#include <mutex>
#include <map>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Libs::Graphics {

RenderContext::RenderContext(GraphicContext& graphics, const std::filesystem::path& shader_store)
    : m_graphics(graphics), m_render_executor(*this), m_command_scheduler(*this, graphics),
      m_descriptor_heap(graphics, m_command_scheduler.GetMasterSemaphore()),
      m_pipeline_cache(graphics, &m_image_heap, shader_store), m_sampler_cache(graphics),
      m_buffer_cache(graphics, m_command_scheduler, m_page_manager, m_texture_cache),
      m_texture_cache(graphics, m_command_scheduler, m_page_manager, m_buffer_cache),
      m_image_heap(*this, m_page_manager), m_occlusion_queries(graphics, m_command_scheduler) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	GpuProfiler::Init(graphics.device, graphics.physical_device_properties.limits.timestampPeriod);
}

RenderContext::~RenderContext() {
	ShutdownGpu();
	m_command_scheduler.Shutdown();
}

void RenderContext::InitializeGpu(VideoOut::VideoOutDriver* video_out) {
	EXIT_IF(m_gpu != nullptr);
	m_video_out = video_out;
	m_gpu       = std::make_unique<GuestGpu>(*this);
}

void RenderContext::ShutdownGpu() {
	if (m_gpu != nullptr) {
		m_gpu->Shutdown();
		m_gpu.reset();
	}
	if (m_video_out != nullptr) {
		if (m_command_scheduler.Active()) {
			m_command_scheduler.Finish();
		}
		m_command_scheduler.DrainPriorityOperations();
		m_video_out = nullptr;
	}
}

GuestGpu& RenderContext::GetGpu() const {
	EXIT_IF(m_gpu == nullptr);
	return *m_gpu;
}

VideoOut::VideoOutDriver& RenderContext::GetVideoOut() const {
	EXIT_IF(m_video_out == nullptr);
	return *m_video_out;
}

namespace {
// Diagnostics: which megabyte of guest memory faults, how, on which thread and at what cost.
struct FaultHistogram {
	struct Entry {
		uint64_t count = 0, ns = 0;
	};
	std::mutex                          mutex;
	std::map<std::string, Entry>        entries;
	void Add(uint64_t vaddr, bool gpu_thread, const char* path, uint64_t ns) {
		char key[96];
		std::snprintf(key, sizeof(key), "0x%09llx00000 %s%s",
		              static_cast<unsigned long long>(vaddr >> 20u), gpu_thread ? "[gpu thread] " : "",
		              path);
		std::scoped_lock lock {mutex};
		auto&            entry = entries[key];
		entry.count++;
		entry.ns += ns;
	}
	void Print() {
		std::vector<std::pair<std::string, Entry>> sorted;
		{
			std::scoped_lock lock {mutex};
			sorted.assign(entries.begin(), entries.end());
			entries.clear();
		}
		std::sort(sorted.begin(), sorted.end(),
		          [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
		for (size_t i = 0; i < sorted.size() && i < 8; i++) {
			std::printf("[faults] %7llu, %5llu ms: %s\n",
			            static_cast<unsigned long long>(sorted[i].second.count),
			            static_cast<unsigned long long>(sorted[i].second.ns / 1000000u),
			            sorted[i].first.c_str());
		}
	}
} g_fault_histogram;
} // namespace

bool RenderContext::HandleFault(PageFaultAccess access, uint64_t fault_vaddr) noexcept {
	// Faults taken on the command thread stall command processing itself; count them apart.
	std::optional<Common::DebugCounters::ScopedTimer> gpu_thread_timer;
	const bool                                         on_gpu_thread = GuestGpu::IsGpuThread();
	if (on_gpu_thread) {
		gpu_thread_timer.emplace(Common::DebugCounters::Counter::GpuThreadFaults,
		                         Common::DebugCounters::Counter::GpuThreadFaultNs);
	}
	const auto fault_start = std::chrono::steady_clock::now();
	const char* fault_path = "unmapped";
	// Slow command-thread faults are CPU/GPU stalls; say which access caused each one.
	struct SlowFaultReport {
		bool                                  active;
		std::chrono::steady_clock::time_point start;
		PageFaultAccess                       access;
		uint64_t                              vaddr;
		const char**                          path;
		~SlowFaultReport() {
			if (!active) {
				return;
			}
			const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
			                    std::chrono::steady_clock::now() - start)
			                    .count();
			// Reads and writes have their own budgets: early write faults must not use up the
			// reports for the read-backs that matter.
			static std::atomic<int> reported_reads {0};
			static std::atomic<int> reported_writes {0};
			const bool              is_read  = access == PageFaultAccess::Read;
			auto&                   reported = is_read ? reported_reads : reported_writes;
			const int               limit    = is_read ? 40 : 6;
			if (us >= 300 && is_read && Common::DebugCounters::Enabled()) {
				// Trace who marks this page as written by the GPU (see BufferCache::ObtainBuffer).
				Common::DebugCounters::g_watch_page.store(vaddr & ~uint64_t {0xfff},
				                                          std::memory_order_relaxed);
			}
			if (us >= 300 && reported.load(std::memory_order_relaxed) < limit &&
			    Common::DebugCounters::Enabled()) {
				reported.fetch_add(1, std::memory_order_relaxed);
				const auto& dbg             = g_draw_debug_context;
				char        stack_text[512] = "";
#ifdef _WIN32
				{
					// Addresses relative to the exe, to be resolved against the linker map.
					void*      frames[20] {};
					const auto count = CaptureStackBackTrace(0, 20, frames, nullptr);
					const auto base  = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
					size_t     used  = 0;
					for (USHORT i = 0; i < count && used + 12 < sizeof(stack_text); i++) {
						const auto addr = reinterpret_cast<uintptr_t>(frames[i]);
						if (addr >= base && addr < base + 0x4000000) {
							used += static_cast<size_t>(
							    std::snprintf(stack_text + used, sizeof(stack_text) - used, " %llx",
							                  static_cast<unsigned long long>(addr - base)));
						}
					}
				}
				char rip_text[96] = "";
				char scan_text[256] = "";
				Common::DebugDescribeAddress(rip_text, sizeof(rip_text),
				                             Common::DebugCounters::g_fault_rip);
				Common::DebugStackScan(scan_text, sizeof(scan_text),
				                       Common::DebugCounters::g_fault_rsp);
				if (is_read) {
					std::printf("[slow fault rip] faulting instruction %s, stack scan:%s\n", rip_text,
					            scan_text);
				}
#endif
				std::printf("[slow fault] %lld us, %s, addr=0x%016llx, via=%s, pm4 op=0x%02x, "
				            "in index buffer=%s [0x%llx..0x%llx], prim=%u, reset_ctl=0x%x, stack:%s\n",
				            static_cast<long long>(us),
				            access == PageFaultAccess::Write ? "write" : "read",
				            static_cast<unsigned long long>(vaddr), *path,
				            GuestGpu::CurrentPm4Opcode(),
				            (vaddr >= dbg.index_begin && vaddr < dbg.index_end) ? "YES" : "no",
				            static_cast<unsigned long long>(dbg.index_begin),
				            static_cast<unsigned long long>(dbg.index_end), dbg.prim_type,
				            dbg.reset_ctl, stack_text);
			}
		}
	} slow_report {on_gpu_thread || (access == PageFaultAccess::Read &&
	                            Common::DebugCounters::Enabled()),
	               fault_start, access, fault_vaddr, &fault_path};
	struct HistogramEntry {
		std::chrono::steady_clock::time_point start;
		uint64_t                              vaddr;
		bool                                  gpu_thread;
		const char**                          path;
		~HistogramEntry() {
			if (Common::DebugCounters::Enabled()) {
				g_fault_histogram.Add(
				    vaddr, gpu_thread, *path,
				    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
				                              std::chrono::steady_clock::now() - start)
				                              .count()));
			}
		}
	} histogram_entry {fault_start, fault_vaddr, on_gpu_thread, &fault_path};
	// The host reports the faulting byte, not the instruction's access width. Both caches
	// resolve its page; guessing a width can cross the end of a valid guest mapping.
	constexpr uint64_t fault_size = 1;
	if (!IsMapped(fault_vaddr, fault_size)) {
		return false;
	}
	// The access retries once the image write-backs over its page land.
	if (WaitWriteBacks(fault_vaddr, fault_size)) {
		fault_path = "image write-back wait";
		return true;
	}
	if (access == PageFaultAccess::Write) {
		fault_path = "write invalidate";
		m_buffer_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_texture_cache.InvalidateMemory(fault_vaddr, fault_size);
		m_image_heap.InvalidateMemory(fault_vaddr, fault_size);
	} else {
		fault_path = "read GPU data back";
		m_buffer_cache.ReadMemory(fault_vaddr, fault_size);
	}
	return true;
}

bool RenderContext::TryReadCleanDword(uint64_t vaddr, uint32_t* value) {
	constexpr uint64_t size = sizeof(uint32_t);
	if (!GuestGpu::IsGpuThread() || !IsMapped(vaddr, size) ||
	    m_texture_cache.WriteBackTick(vaddr, size) != 0 ||
	    m_buffer_cache.HasGpuDirtyBytes(vaddr, size)) {
		return false;
	}
	return LibKernel::Memory::TryReadBacking(vaddr, value, size);
}

bool RenderContext::WaitWriteBacks(uint64_t vaddr, uint64_t size) {
	const auto tick = m_texture_cache.WriteBackTick(vaddr, size);
	if (tick == 0) {
		return false;
	}
	Common::DebugCounters::ScopedTimer write_back_timer(
	    Common::DebugCounters::Counter::WriteBackWaits,
	    Common::DebugCounters::Counter::WriteBackWaitNs);
	// The GPU thread can fault inside a draw, so it submits without running deferred operations.
	if (GuestGpu::IsGpuThread()) {
		m_command_scheduler.Wait(tick);
	} else if (tick == m_command_scheduler.CurrentTick()) {
		GetGpu().SendCommandSync([this, tick] { m_command_scheduler.Wait(tick); });
	}
	m_command_scheduler.WaitPriorityOperations(tick);
	return true;
}

bool RenderContext::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	// Host I/O into the range, even one running past its mapping, would fail on its inaccessible
	// pages, and a later write-back would overwrite it.
	(void)WaitWriteBacks(vaddr, size);
	if (!IsMapped(vaddr, size)) {
		return false;
	}
	m_buffer_cache.InvalidateMemory(vaddr, size);
	m_texture_cache.InvalidateMemory(vaddr, size);
	m_image_heap.InvalidateMemory(vaddr, size);
	return true;
}

bool RenderContext::IsMapped(uint64_t vaddr, uint64_t size) const noexcept {
	if (!GuestRange {vaddr, size}.Valid()) {
		return false;
	}
	std::shared_lock lock(m_mapped_ranges_mutex);
	return m_mapped_ranges.Contains(vaddr, size);
}

void RenderContext::MapMemory(uint64_t vaddr, uint64_t size) {
	std::lock_guard lock(m_mapped_ranges_mutex);
	m_mapped_ranges.Add(vaddr, size);
	m_buffer_cache.QueueBdaSync(vaddr, size);
}

void RenderContext::UnmapMemory(uint64_t vaddr, uint64_t size) {
	if (CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported memory unmap from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	const auto unmap = [this, vaddr, size] {
		// Check cache ownership on the GPU thread. Guest-memory callbacks can still
		// access a range with no cached data, so they must finish before it is unmapped.
		if (m_command_scheduler.Active() &&
		    (m_buffer_cache.IsRegionRegistered(vaddr, size) ||
		     m_texture_cache.IsRegionRegistered(vaddr, size) ||
		     m_command_scheduler.HasPendingPriorityOperations())) {
			const auto tick = m_command_scheduler.CurrentTick();
			m_command_scheduler.Finish();
			m_command_scheduler.WaitPriorityOperations(tick);
		}
		m_buffer_cache.InvalidateMemory(vaddr, size);
		m_texture_cache.UnmapMemory(vaddr, size);
		m_image_heap.UnmapMemory(vaddr, size);
		std::lock_guard lock(m_mapped_ranges_mutex);
		m_mapped_ranges.Subtract(vaddr, size);
	};
	// Shutdown still owns the GPU while queued rendering drains, but its command lane no
	// longer accepts external work. Use the guest GPU's state for the teardown route.
	if (m_gpu == nullptr || m_gpu->IsStopping()) {
		unmap();
		return;
	}
	m_gpu->SendCommandSync(unmap);
}

void RenderContext::PrepareBda() {
	std::shared_lock lock(m_mapped_ranges_mutex);
	m_buffer_cache.TakeBdaSync().ForEach([this](uint64_t begin, uint64_t end) {
		m_mapped_ranges.ForEachInRange(begin, end - begin, [this](uint64_t start, uint64_t finish) {
			m_buffer_cache.SynchronizeBuffersInRange(start, finish - start);
		});
	});
	m_fault_process_pending = true;
}

void RenderContext::RunGarbageCollector() {
	if (m_fault_process_pending) {
		m_fault_process_pending = false;
		m_buffer_cache.ProcessFaultBuffer();
	}
	m_image_heap.RunGarbageCollector();
	m_texture_cache.ProcessDownloadImages();
	m_texture_cache.RunGarbageCollector();
	m_buffer_cache.RunGarbageCollector();
	LogDebugStats();
}

void RenderContext::LogDebugStats() {
	using Common::DebugCounters::Counter;
	if (!Common::DebugCounters::Enabled()) {
		return;
	}
	constexpr int   COUNT = static_cast<int>(Counter::Count);
	static uint64_t last[COUNT] {};
	static auto     window      = std::chrono::steady_clock::now();
	static uint64_t frames      = 0;
	static uint64_t pending_max = 0;
	static uint64_t pending_sum = 0;

	// Submissions the host GPU has not finished yet: how far the command thread runs ahead.
	auto& master = m_command_scheduler.GetMasterSemaphore();
	master.Refresh();
	const auto submitted = master.CurrentTick() - 1;
	const auto done      = master.KnownGpuTick();
	const auto pending   = submitted > done ? submitted - done : uint64_t {0};
	pending_max          = std::max(pending_max, pending);
	pending_sum += pending;
	frames++;

	const auto now     = std::chrono::steady_clock::now();
	const auto elapsed = std::chrono::duration<double>(now - window).count();
	if (elapsed < 10.0) {
		return;
	}
	uint64_t delta[COUNT] {};
	for (int i = 0; i < COUNT; i++) {
		const auto value = Common::DebugCounters::Get(static_cast<Counter>(i));
		delta[i]         = value - last[i];
		last[i]          = value;
	}
	const auto n = [&](Counter c) {
		return static_cast<unsigned long long>(delta[static_cast<int>(c)]);
	};
	const auto ms = [&](Counter c) { return static_cast<double>(delta[static_cast<int>(c)]) / 1e6; };
	const auto mb = [](uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); };

	std::printf("[stats] %.1f s: frames=%llu (%.1f fps) flips=%llu rejected=%llu late_vblanks=%llu "
	            "dropped=%llu | draws=%llu dispatches=%llu submits=%llu\n",
	            elapsed, static_cast<unsigned long long>(frames),
	            static_cast<double>(frames) / elapsed, n(Counter::FlipsRequested),
	            n(Counter::FlipsRejected), n(Counter::LateVblanks), n(Counter::DroppedPresents),
	            n(Counter::Draws), n(Counter::Dispatches), n(Counter::QueueSubmits));
	std::printf("[stats] cpu<-gpu: waits=%llu (%.0f ms) | gpu thread busy=%.0f ms | pending "
	            "submits avg=%.1f max=%llu | faults=%llu (%.0f ms), on gpu thread=%llu (%.0f ms) | "
	            "present=%.0f ms\n",
	            n(Counter::GpuWaits), ms(Counter::GpuWaitNs), ms(Counter::GpuThreadBusyNs),
	            static_cast<double>(pending_sum) / static_cast<double>(frames),
	            static_cast<unsigned long long>(pending_max), n(Counter::GpuFaults),
	            ms(Counter::GpuFaultNs), n(Counter::GpuThreadFaults), ms(Counter::GpuThreadFaultNs),
	            ms(Counter::PresentNs));
	std::printf("[stats] command thread waits=%llu (%.0f ms) | guest read-backs: recorded=%llu "
	            "reused=%llu -> served=%llu stalled=%llu | command-thread read-backs=%llu | "
	            "textures created=%llu (%.0f MB)\n",
	            n(Counter::GpuThreadWaits), ms(Counter::GpuThreadWaitNs),
	            n(Counter::ReadbackRecorded), n(Counter::ReadbackReused),
	            n(Counter::ReadbackServed), n(Counter::ReadbackStalled),
	            n(Counter::ReadbackGpuThread), n(Counter::TexturesCreated),
	            mb(delta[static_cast<int>(Counter::TextureCreateBytes)]));
	std::printf("[stats] guest waits: buffer read-back %.0f ms (+%.0f ms reaching the command "
	            "thread) | image write-back %llu, %.0f ms | completion thread idle-wait %.0f ms\n",
	            ms(Counter::GuestReadbackWaitNs), ms(Counter::GuestReadbackTripNs),
	            n(Counter::WriteBackWaits), ms(Counter::WriteBackWaitNs),
	            ms(Counter::CompletionWaitNs));
	g_fault_histogram.Print();
	std::printf("[stats] texture re-uploads=%llu (%.0f MB)\n", n(Counter::TextureReuploads),
	            mb(delta[static_cast<int>(Counter::TextureReuploadBytes)]));
	const auto max_extent = Common::DebugCounters::g_max_render_extent.exchange(0);
	std::printf("[stats] render passes=%llu, %.2f MP each on average, largest %llux%llu\n",
	            n(Counter::RenderPasses),
	            delta[static_cast<int>(Counter::RenderPasses)] != 0
	                ? static_cast<double>(delta[static_cast<int>(Counter::RenderPassPixels)]) /
	                      static_cast<double>(delta[static_cast<int>(Counter::RenderPasses)]) / 1e6
	                : 0.0,
	            static_cast<unsigned long long>(max_extent >> 32u),
	            static_cast<unsigned long long>(max_extent & 0xffffffffu));
	const auto textures = m_texture_cache.GetDebugStats();
	const auto buffers  = m_buffer_cache.GetDebugStats();
	std::printf("[stats] textures: %llu images %.0f MB, evicted=%llu, up=%.1f MB, down=%.1f MB | "
	            "buffers: %llu, %.0f MB, evicted=%llu, up=%.1f MB, down=%.1f MB | compiled: "
	            "shaders=%llu (%.0f ms) pipelines=%llu (%.0f ms)\n",
	            static_cast<unsigned long long>(textures.registered_images),
	            mb(textures.registered_bytes), n(Counter::TextureEvictions),
	            mb(delta[static_cast<int>(Counter::TextureUploadBytes)]),
	            mb(delta[static_cast<int>(Counter::TextureDownloadBytes)]),
	            static_cast<unsigned long long>(buffers.registered_buffers),
	            mb(buffers.registered_bytes), n(Counter::BufferEvictions),
	            mb(delta[static_cast<int>(Counter::BufferUploadBytes)]),
	            mb(delta[static_cast<int>(Counter::BufferDownloadBytes)]),
	            n(Counter::ShadersCompiled), ms(Counter::ShaderCompileNs),
	            n(Counter::PipelinesCreated), ms(Counter::PipelineCreateNs));
	std::array<GraphicContext::HeapUsage, VK_MAX_MEMORY_HEAPS> heaps {};
	const auto heap_count = m_graphics.GetHeapUsage(heaps);
	std::printf("[stats] vram:");
	for (uint32_t heap = 0; heap < heap_count; heap++) {
		std::printf(" heap%u%s %.0f/%.0f MB (%u allocs, %.0f MB in %.0f MB of blocks)", heap,
		            heaps[heap].device_local ? "(local)" : "", mb(heaps[heap].usage),
		            mb(heaps[heap].budget), heaps[heap].allocations,
		            mb(heaps[heap].allocation_bytes), mb(heaps[heap].block_bytes));
	}
	std::printf(" | idle images pinned by heaps: %llu, %.0f MB",
	            static_cast<unsigned long long>(textures.pinned_images),
	            mb(textures.pinned_bytes));
	std::printf(" | gc sees %.0f MB; texture gc trigger/pressure/critical=%.0f/%.0f/%.0f MB; "
	            "buffer gc trigger/critical=%.0f/%.0f MB; image pool=%.0f MB | vram capacity=%.0f MB, "
	            "budget=%.0f MB\n",
	            mb(textures.used), mb(textures.trigger), mb(textures.pressure),
	            mb(textures.critical), mb(buffers.trigger), mb(buffers.critical),
	            mb(m_graphics.ImagePoolBytes()),
	            mb(m_graphics.GetDeviceLocalMemoryCapacity()),
	            mb(m_graphics.GetTotalMemoryBudget()));
	std::fflush(stdout);
	window      = now;
	frames      = 0;
	pending_max = 0;
	pending_sum = 0;
}

void RenderContext::AddInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it != m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.push_back({eq, event_id});
}

void RenderContext::DeleteInterruptEq(LibKernel::EventQueue::KernelEqueue eq, int event_id) {
	Common::LockGuard lock(m_interrupt_mutex);

	auto it = std::find_if(
	    m_interrupt_eqs.begin(), m_interrupt_eqs.end(),
	    [eq, event_id](const auto& entry) { return entry.eq == eq && entry.event_id == event_id; });
	if (it == m_interrupt_eqs.end()) {
		return;
	}

	m_interrupt_eqs.erase(it);
}

void RenderContext::TriggerInterrupt(int event_id, uint32_t context_id) {
	std::vector<InterruptEqRegistration> registrations;
	{
		Common::LockGuard lock(m_interrupt_mutex);
		for (const auto& registration: m_interrupt_eqs) {
			if (registration.event_id == event_id) {
				registrations.push_back(registration);
			}
		}
	}

	for (const auto& registration: registrations) {
		const auto result = LibKernel::EventQueue::KernelTriggerEvent(
		    registration.eq, static_cast<uintptr_t>(registration.event_id),
		    LibKernel::EventQueue::KERNEL_EVFILT_GRAPHICS,
		    reinterpret_cast<void*>(static_cast<uintptr_t>(context_id)));
		if (result == LibKernel::KERNEL_ERROR_EBADF || result == LibKernel::KERNEL_ERROR_ENOENT) {
			DeleteInterruptEq(registration.eq, registration.event_id);
			continue;
		}
		EXIT_NOT_IMPLEMENTED(result != OK);
	}
}

} // namespace Libs::Graphics
