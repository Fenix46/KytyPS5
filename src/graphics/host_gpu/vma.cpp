#include "graphics/host_gpu/vulkanCommon.h"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#pragma clang diagnostic ignored "-Wunused-private-field"
#pragma clang diagnostic ignored "-Wunused-variable"
#endif

#define VMA_IMPLEMENTATION
#include <vk_mem_alloc.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"

#include <algorithm>
#include <cinttypes>

namespace Libs::Graphics {

bool GraphicContext::CreateAllocator() {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(instance == nullptr || physical_device == nullptr || device == nullptr ||
	        allocator != nullptr);

	VmaVulkanFunctions functions {};
	functions.vkGetInstanceProcAddr = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr;
	functions.vkGetDeviceProcAddr   = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;

	VmaAllocatorCreateInfo info {};
	info.instance         = instance;
	info.physicalDevice   = physical_device;
	info.device           = device;
	info.pVulkanFunctions = &functions;
	info.vulkanApiVersion = VULKAN_TARGET_API_VERSION;
	info.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
	if (memory_budget_ext_enabled) {
		info.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
	}

	const auto result = static_cast<vk::Result>(vmaCreateAllocator(&info, &allocator));
	if (result != vk::Result::eSuccess) {
		LOGF("vmaCreateAllocator failed: %s\n", vk::to_string(result).c_str());
		return false;
	}
	return true;
}

void GraphicContext::DestroyAllocator() {
	if (allocator == nullptr) {
		return;
	}
	FlushImagePool();
	vmaDestroyAllocator(allocator);
	allocator = nullptr;
}

void GraphicContext::LogMemoryBudget() const {
	if (allocator == nullptr || physical_device == nullptr) {
		return;
	}

	const auto& properties = physical_device_memory_properties;
	VmaBudget   budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	for (uint32_t i = 0; i < properties.memoryHeapCount; i++) {
		LOGF("VMA heap %u: usage=%" PRIu64 ", budget=%" PRIu64 ", allocation=%" PRIu64
		     ", blocks=%" PRIu64 "\n",
		     i, static_cast<uint64_t>(budgets[i].usage), static_cast<uint64_t>(budgets[i].budget),
		     static_cast<uint64_t>(budgets[i].statistics.allocationBytes),
		     static_cast<uint64_t>(budgets[i].statistics.blockBytes));
	}
}

uint64_t GraphicContext::GetDeviceMemoryUsage() const {
	if (!CanReportMemoryUsage() || allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t usage = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const bool device_local =
		    static_cast<bool>(physical_device_memory_properties.memoryHeaps[heap].flags &
		                      vk::MemoryHeapFlagBits::eDeviceLocal);
		if (!discrete || device_local) {
			usage += budgets[heap].usage;
		}
	}
	return usage;
}

uint32_t GraphicContext::GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const auto count = physical_device_memory_properties.memoryHeapCount;
	for (uint32_t heap = 0; heap < count; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		heaps[heap] = {.usage        = budgets[heap].usage,
		               .budget       = budgets[heap].budget,
		               .allocations  = budgets[heap].statistics.allocationCount,
		               .allocation_bytes = budgets[heap].statistics.allocationBytes,
		               .block_bytes      = budgets[heap].statistics.blockBytes,
		               .device_local = static_cast<bool>(properties.flags &
		                                                 vk::MemoryHeapFlagBits::eDeviceLocal)};
	}
	return count;
}

uint64_t GraphicContext::GetTotalMemoryBudget() const {
	if (allocator == nullptr) {
		return 0;
	}
	VmaBudget budgets[VK_MAX_MEMORY_HEAPS] {};
	vmaGetHeapBudgets(allocator, budgets);
	const bool discrete =
	    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
	uint64_t budget = 0;
	uint64_t local  = 0;
	uint64_t usage  = 0;
	for (uint32_t heap = 0; heap < physical_device_memory_properties.memoryHeapCount; heap++) {
		const auto& properties = physical_device_memory_properties.memoryHeaps[heap];
		const bool  device_local =
		    static_cast<bool>(properties.flags & vk::MemoryHeapFlagBits::eDeviceLocal);
		if (device_local) {
			local += properties.size;
		}
		if (!discrete || device_local) {
			budget += CanReportMemoryUsage() ? budgets[heap].budget : properties.size;
			usage += CanReportMemoryUsage() ? budgets[heap].usage : 0;
		}
	}
	if (discrete) {
		return budget - std::min<uint64_t>(budget / 8, 1024ull * 1024 * 1024);
	}
	constexpr uint64_t system_reserve = 8ull * 1024 * 1024 * 1024;
	const auto         available      = budget > usage ? budget - usage : uint64_t {0};
	return std::max(local, available > system_reserve ? available - system_reserve : uint64_t {0});
}

GraphicContext::GcThresholds GraphicContext::ComputeGcThresholds(uint64_t budget) {
	constexpr int64_t GiB = 1024ll * 1024 * 1024;
	const auto       b    = static_cast<int64_t>(std::min<uint64_t>(budget, INT64_MAX));
	return {
	    static_cast<uint64_t>(std::max<int64_t>(b * 75 / 100, b - 4 * GiB)),
	    static_cast<uint64_t>(std::max<int64_t>(b * 88 / 100, b - 2 * GiB)),
	};
}

// Images freed by the texture cache are kept and handed back to identical requests: creating and
// destroying device memory dominates the time spent loading levels.
static constexpr size_t   IMAGE_POOL_MAX_ENTRIES = 64;
static constexpr uint64_t IMAGE_POOL_MAX_BYTES   = 256ull * 1024 * 1024;

void GraphicContext::FlushImagePool() {
	std::vector<PooledImage> pool;
	{
		std::lock_guard lock(m_image_pool_mutex);
		pool.swap(m_image_pool);
		m_image_pool_bytes = 0;
	}
	for (const auto& pooled: pool) {
		vmaDestroyImage(allocator, pooled.image, pooled.allocation);
	}
}

bool GraphicContext::CreateImage(const vk::ImageCreateInfo& image_info, VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image != nullptr || image.allocation != nullptr);

	const bool poolable = image_info.pNext == nullptr &&
	                      image_info.tiling == vk::ImageTiling::eOptimal &&
	                      image_info.initialLayout == vk::ImageLayout::eUndefined &&
	                      image_info.sharingMode == vk::SharingMode::eExclusive;

	if (poolable) {
		std::lock_guard lock(m_image_pool_mutex);
		for (size_t i = m_image_pool.size(); i-- > 0;) {
			const auto& p = m_image_pool[i];
			if (p.format == image_info.format && p.image_type == image_info.imageType &&
			    p.extent.width == image_info.extent.width &&
			    p.extent.height == image_info.extent.height &&
			    p.extent.depth == image_info.extent.depth && p.layers == image_info.arrayLayers &&
			    p.mip_levels == image_info.mipLevels &&
			    p.samples == static_cast<uint32_t>(image_info.samples) &&
			    p.usage == image_info.usage && p.flags == image_info.flags) {
				image.image      = p.image;
				image.allocation = p.allocation;
				m_image_pool_bytes -= p.size;
				m_image_pool.erase(m_image_pool.begin() + static_cast<std::ptrdiff_t>(i));
				image.format     = image_info.format;
				image.image_type = image_info.imageType;
				image.extent     = image_info.extent;
				image.layers     = image_info.arrayLayers;
				image.mip_levels = image_info.mipLevels;
				image.samples    = static_cast<uint32_t>(image_info.samples);
				image.usage      = image_info.usage;
				image.flags      = image_info.flags;
				image.state      = {.layout = image_info.initialLayout};
				image.subresource_states.clear();
				image.poolable = true;
				return true;
			}
		}
	}

	VmaAllocationCreateInfo alloc_info {};
	alloc_info.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
	alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

	vk::Image::CType native_image = VK_NULL_HANDLE;
	auto             result       = static_cast<vk::Result>(vmaCreateImage(
        allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info), &alloc_info,
        &native_image, &image.allocation, nullptr));
	if (result != vk::Result::eSuccess) {
		// Cached images may be what is taking the memory: release them and retry once.
		FlushImagePool();
		native_image = VK_NULL_HANDLE;
		result       = static_cast<vk::Result>(vmaCreateImage(
            allocator, static_cast<const vk::ImageCreateInfo::NativeType*>(image_info),
            &alloc_info, &native_image, &image.allocation, nullptr));
	}
	image.image = native_image;
	if (result != vk::Result::eSuccess) {
		LogMemoryBudget();
		return false;
	}
	image.poolable = poolable;

	image.format     = image_info.format;
	image.image_type = image_info.imageType;
	image.extent     = image_info.extent;
	image.layers     = image_info.arrayLayers;
	image.mip_levels = image_info.mipLevels;
	image.samples    = static_cast<uint32_t>(image_info.samples);
	image.usage      = image_info.usage;
	image.flags      = image_info.flags;
	image.state      = {.layout = image_info.initialLayout};
	image.subresource_states.clear();

	return true;
}

void GraphicContext::DeleteImage(VulkanImage& image) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(allocator == nullptr || image.image == nullptr || image.allocation == nullptr);

	bool pooled = false;
	std::vector<PooledImage> evicted;
	if (image.poolable) {
		VmaAllocationInfo allocation_info {};
		vmaGetAllocationInfo(allocator, image.allocation, &allocation_info);
		const uint64_t size = allocation_info.size;

		std::lock_guard lock(m_image_pool_mutex);
		if (size <= IMAGE_POOL_MAX_BYTES) {
			while (!m_image_pool.empty() && (m_image_pool.size() >= IMAGE_POOL_MAX_ENTRIES ||
			                                 m_image_pool_bytes + size > IMAGE_POOL_MAX_BYTES)) {
				m_image_pool_bytes -= m_image_pool.front().size;
				evicted.push_back(m_image_pool.front());
				m_image_pool.erase(m_image_pool.begin());
			}
			PooledImage entry {};
			entry.format     = image.format;
			entry.image_type = image.image_type;
			entry.extent     = image.extent;
			entry.layers     = image.layers;
			entry.mip_levels = image.mip_levels;
			entry.samples    = image.samples;
			entry.usage      = image.usage;
			entry.flags      = image.flags;
			entry.image      = image.image;
			entry.allocation = image.allocation;
			entry.size       = size;
			m_image_pool.push_back(entry);
			m_image_pool_bytes += size;
			pooled = true;
		}
	}
	for (const auto& e: evicted) {
		vmaDestroyImage(allocator, e.image, e.allocation);
	}
	if (!pooled) {
		vmaDestroyImage(allocator, image.image, image.allocation);
	}
	image.image      = nullptr;
	image.allocation = nullptr;
	image.poolable   = false;
}

} // namespace Libs::Graphics
