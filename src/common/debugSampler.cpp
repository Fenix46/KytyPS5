#include "common/debugSampler.h"

#include "common/debugCounters.h"
#include "common/hostException.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#endif

namespace Common::DebugSampler {

#ifdef _WIN32
namespace {

std::atomic<bool> g_started {false};

void PrintTop(const char* title, std::unordered_map<uint64_t, uint32_t>& map, size_t count,
              uint32_t total) {
	std::vector<std::pair<uint64_t, uint32_t>> items(map.begin(), map.end());
	std::sort(items.begin(), items.end(), [](auto& a, auto& b) { return a.second > b.second; });
	std::printf("[sample] %s:", title);
	for (size_t i = 0; i < items.size() && i < count; i++) {
		std::printf(" %llx=%u%%", static_cast<unsigned long long>(items[i].first),
		            total != 0 ? items[i].second * 100u / total : 0u);
	}
	std::printf("\n");
}

// A thread suspended inside generated code or a prologue can hand the unwinder registers that
// lead it through unmapped memory: the walk is guarded and simply ends there.
int Unwind(CONTEXT& context, uint64_t* functions, int max_depth) {
	int depth = 0;
	Common::HostException::t_guarded_probe = true;
	__try {
		for (; depth < max_depth && context.Rip != 0; depth++) {
			DWORD64     image = 0;
			const auto* entry = RtlLookupFunctionEntry(context.Rip, &image, nullptr);
			functions[depth]  = entry != nullptr ? image + entry->BeginAddress : context.Rip;
			if (entry == nullptr) {
				break;
			}
			void*   handler_data = nullptr;
			DWORD64 frame        = 0;
			RtlVirtualUnwind(UNW_FLAG_NHANDLER, image, context.Rip,
			                 const_cast<PRUNTIME_FUNCTION>(entry), &context, &handler_data, &frame,
			                 nullptr);
		}
	} __except (EXCEPTION_EXECUTE_HANDLER) {
	}
	Common::HostException::t_guarded_probe = false;
	return depth;
}

void Run(HANDLE thread, std::string label) {
	const auto base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
	std::unordered_map<uint64_t, uint32_t> self, inclusive;
	uint32_t total = 0;
	auto     window = std::chrono::steady_clock::now();
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		if (!Common::DebugCounters::Enabled()) {
			continue;
		}
		CONTEXT context {};
		context.ContextFlags = CONTEXT_FULL;
		if (SuspendThread(thread) == static_cast<DWORD>(-1)) {
			return;
		}
		uint64_t functions[48] {};
		int      depth = 0;
		if (GetThreadContext(thread, &context)) {
			depth = Unwind(context, functions, 48);
		}
		ResumeThread(thread);
		if (depth == 0) {
			continue;
		}
		total++;
		self[functions[0] - base]++;
		for (int i = 0; i < depth; i++) {
			bool seen = false;
			for (int j = 0; j < i; j++) {
				seen = seen || functions[j] == functions[i];
			}
			if (!seen) {
				inclusive[functions[i] - base]++;
			}
		}
		if (std::chrono::steady_clock::now() - window >= std::chrono::seconds(10)) {
			std::printf("[sample] thread=%s samples=%u\n", label.c_str(), total);
			PrintTop("self", self, 14, total);
			PrintTop("inclusive", inclusive, 40, total);
			std::fflush(stdout);
			self.clear();
			inclusive.clear();
			total  = 0;
			window = std::chrono::steady_clock::now();
		}
	}
}

} // namespace
#endif

void StartForCurrentThread(const char* label) {
#ifdef _WIN32
	if (!Common::DebugCounters::Enabled() || g_started.exchange(true)) {
		return;
	}
	HANDLE handle = nullptr;
	if (!DuplicateHandle(GetCurrentProcess(), GetCurrentThread(), GetCurrentProcess(), &handle,
	                     THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION, FALSE,
	                     0)) {
		return;
	}
	std::thread(Run, handle, std::string(label)).detach();
#else
	(void)label;
#endif
}

} // namespace Common::DebugSampler
