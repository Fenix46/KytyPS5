#include "graphics/host_gpu/renderer/gpuProfiler.h"

#include "common/debugCounters.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <deque>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::GpuProfiler {

namespace {

constexpr uint32_t QUERY_COUNT = 8192; // two per slot

struct Pending {
	uint32_t first;
	uint64_t tick;
	uint64_t key;
	uint64_t a, b;
	uint32_t kind, work;
};

struct Total {
	uint64_t a = 0, b = 0;
	uint32_t kind = 0;
	uint64_t ns = 0, count = 0, work = 0;
};

vk::Device                              g_device;
vk::QueryPool                           g_pool;
float                                   g_period = 1.0f;
bool                                    g_ready  = false;
uint32_t                                g_next   = 0;
uint32_t                                g_inuse  = 0;
std::deque<Pending>                     g_pending;
std::unordered_map<uint64_t, Total>     g_totals;
uint64_t                                g_total_ns = 0;
std::chrono::steady_clock::time_point   g_window   = std::chrono::steady_clock::now();
std::vector<uint64_t>                   g_scratch;
CommandScheduler*                       g_scheduler = nullptr;

} // namespace

void Init(vk::Device device, float period_ns) {
	if (g_ready || !Common::DebugCounters::Enabled()) {
		return;
	}
	g_device = device;
	g_period = period_ns;
	const vk::QueryPoolCreateInfo info {.queryType  = vk::QueryType::eTimestamp,
	                                    .queryCount = QUERY_COUNT};
	if (device.createQueryPool(&info, nullptr, &g_pool) != vk::Result::eSuccess) {
		return;
	}
	device.resetQueryPool(g_pool, 0, QUERY_COUNT);
	g_scratch.resize(2);
	g_ready = true;
}

bool Active() {
	return g_ready;
}

int Begin(vk::CommandBuffer command, CommandScheduler& scheduler, uint64_t key, uint64_t a,
          uint64_t b, uint32_t kind, uint32_t work) {
	if (!g_ready || g_inuse + 2 > QUERY_COUNT) {
		return -1;
	}
	if (g_next + 2 > QUERY_COUNT) {
		g_next = 0;
	}
	g_scheduler      = &scheduler;
	const auto first = g_next;
	g_next += 2;
	g_inuse += 2;
	command.writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, g_pool, first);
	g_pending.push_back({first, scheduler.CurrentTick(), key, a, b, kind, work});
	return static_cast<int>(first);
}

void End(vk::CommandBuffer command, int slot) {
	if (slot >= 0) {
		command.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, g_pool,
		                       static_cast<uint32_t>(slot) + 1u);
	}
}

void Poll() {
	if (!g_ready || g_scheduler == nullptr) {
		return;
	}
	while (!g_pending.empty() && g_scheduler->IsFree(g_pending.front().tick)) {
		const auto p = g_pending.front();
		g_pending.pop_front();
		uint64_t values[2] {};
		const auto result =
		    g_device.getQueryPoolResults(g_pool, p.first, 2, sizeof(values), values,
		                                 sizeof(uint64_t), vk::QueryResultFlagBits::e64);
		g_device.resetQueryPool(g_pool, p.first, 2);
		g_inuse -= 2;
		if (result != vk::Result::eSuccess || values[1] < values[0]) {
			continue;
		}
		const auto ns = static_cast<uint64_t>(static_cast<double>(values[1] - values[0]) * g_period);
		auto&      t  = g_totals[p.key];
		t.a = p.a;
		t.b = p.b;
		t.kind = p.kind;
		t.ns += ns;
		t.count++;
		t.work += p.work;
		g_total_ns += ns;
	}
	if (std::chrono::steady_clock::now() - g_window < std::chrono::seconds(10)) {
		return;
	}
	std::vector<std::pair<uint64_t, Total>> items(g_totals.begin(), g_totals.end());
	std::sort(items.begin(), items.end(),
	          [](const auto& x, const auto& y) { return x.second.ns > y.second.ns; });
	std::printf("[gpu] 10 s window: %.0f ms of draw/dispatch GPU time\n",
	            static_cast<double>(g_total_ns) / 1e6);
	for (size_t i = 0; i < items.size() && i < 12; i++) {
		const auto& t = items[i].second;
		std::printf("[gpu] %5.1f ms  %6llu calls  %s  a=%016llx b=%016llx  avg work=%llu\n",
		            static_cast<double>(t.ns) / 1e6, static_cast<unsigned long long>(t.count),
		            t.kind == 0 ? "draw" : t.kind == 1 ? "mesh" : "cs  ",
		            static_cast<unsigned long long>(t.a), static_cast<unsigned long long>(t.b),
		            static_cast<unsigned long long>(t.count != 0 ? t.work / t.count : 0));
	}
	std::fflush(stdout);
	g_totals.clear();
	g_total_ns = 0;
	g_window   = std::chrono::steady_clock::now();
}

} // namespace Libs::Graphics::GpuProfiler
