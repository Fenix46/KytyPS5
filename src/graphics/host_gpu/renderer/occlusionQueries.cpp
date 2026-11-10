#include "graphics/host_gpu/renderer/occlusionQueries.h"

#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"

#include <algorithm>

namespace Libs::Graphics {

// Queries no dump has read yet; past them a draw counts one sample.
constexpr uint32_t QUERY_COUNT = 16384;

OcclusionQueries::OcclusionQueries(GraphicContext& graphics, CommandScheduler& scheduler)
    : m_scheduler(scheduler) {
	if (!graphics.occlusion_queries_enabled) {
		return;
	}
	m_device = graphics.device;
	const vk::QueryPoolCreateInfo info {.queryType  = vk::QueryType::eOcclusion,
	                                    .queryCount = QUERY_COUNT};
	RequireVulkanSuccess(m_device.createQueryPool(&info, nullptr, &m_pool), "vkCreateQueryPool");
	m_device.resetQueryPool(m_pool, 0, QUERY_COUNT);
	m_results.resize(QUERY_COUNT);
	if (graphics.occlusion_query_precise_enabled) {
		m_control = vk::QueryControlFlagBits::ePrecise;
	}
}

OcclusionQueries::~OcclusionQueries() {
	if (m_pool) {
		m_device.destroyQueryPool(m_pool, nullptr);
	}
}

int32_t OcclusionQueries::BeginQuery(vk::CommandBuffer command) {
	if (m_begun - m_read == QUERY_COUNT) {
		m_uncounted++;
		return -1;
	}
	const auto query = static_cast<uint32_t>(m_begun++ % QUERY_COUNT);
	command.beginQuery(m_pool, query, m_control);
	return static_cast<int32_t>(query);
}

void OcclusionQueries::Dump(uint64_t address) {
	// Begins dump to the results, ends 8 bytes into them.
	if ((address & 8u) == 0) {
		m_open++;
	} else if (m_open != 0) {
		m_open--;
	}
	m_dumps.push_back({address, m_scheduler.CurrentTick(), m_begun, m_uncounted});
}

void OcclusionQueries::FinalizeDumps(size_t count) {
	if (count == 0) {
		return;
	}
	if (const auto last = m_dumps[count - 1].queries; last != m_read) {
		// The unread queries wrap around the pool at most once.
		for (uint64_t read = m_read; read < last;) {
			const auto first = static_cast<uint32_t>(read % QUERY_COUNT);
			const auto n =
			    static_cast<uint32_t>(std::min<uint64_t>(last - read, QUERY_COUNT - first));
			RequireVulkanSuccess(
			    m_device.getQueryPoolResults(m_pool, first, n, n * sizeof(uint64_t),
				                             m_results.data() + first, sizeof(uint64_t),
				                             vk::QueryResultFlagBits::e64),
			    "vkGetQueryPoolResults");
			m_device.resetQueryPool(m_pool, first, n);
			read += n;
		}
	}
	for (size_t i = 0; i < count; i++) {
		const auto& dump = m_dumps[i];
		for (; m_read < dump.queries; m_read++) {
			m_samples += m_results[m_read % QUERY_COUNT];
		}
		WriteZPassCounters(reinterpret_cast<volatile uint64_t*>(dump.address),
		                   m_samples + dump.uncounted, (dump.address & 8u) == 0);
	}
	m_dumps.erase(m_dumps.begin(), m_dumps.begin() + static_cast<std::ptrdiff_t>(count));
}

void OcclusionQueries::PublishDumps(bool wait) {
	// A bounded backlog: past it, stop deferring.
	constexpr size_t MAX_DEFERRED_DUMPS = 512;
	if (wait || m_dumps.size() > MAX_DEFERRED_DUMPS) {
		if (m_dumps.back().queries != m_read) {
			m_scheduler.Wait(m_dumps.back().tick);
		}
		FinalizeDumps(m_dumps.size());
		return;
	}
	// Dumps run in order, so the ones the GPU has finished form a prefix. A dump that needs no new
	// query results is exact already.
	size_t ready = 0;
	while (ready < m_dumps.size() &&
	       (m_dumps[ready].queries <= m_read || m_scheduler.IsFree(m_dumps[ready].tick))) {
		ready++;
	}
	FinalizeDumps(ready);
	// The rest: a provisional counter pair that reads as visible, replaced when the GPU is done.
	// Rewritten on every call: finalizing an earlier begin dump clears the end counters.
	for (const auto& dump: m_dumps) {
		const bool begin = (dump.address & 8u) == 0;
		const auto base  = m_samples + dump.uncounted;
		WriteZPassCounters(reinterpret_cast<volatile uint64_t*>(dump.address),
		                   begin ? base : base + 1u, begin);
	}
}

} // namespace Libs::Graphics
