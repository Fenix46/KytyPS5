#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/threads.h"
#include "graphics/host_gpu/vulkanCommon.h" // IWYU pragma: export

#include <chrono>
#include <map>
#include <mutex>
#include <span>
#include <tuple>
#include <vector>
#include <vk_mem_alloc.h>

namespace Libs::Graphics {

struct VulkanImage;

inline constexpr uint32_t VULKAN_TARGET_API_VERSION = VK_API_VERSION_1_3;

struct GraphicContext {
	vk::Instance                       instance                              = nullptr;
	vk::DebugUtilsMessengerEXT         debug_messenger                       = nullptr;
	vk::PhysicalDevice                 physical_device                       = nullptr;
	vk::PhysicalDeviceProperties       physical_device_properties            = {};
	vk::PhysicalDeviceMemoryProperties physical_device_memory_properties     = {};
	vk::Device                         device                                = nullptr;
	VmaAllocator                       allocator                             = nullptr;
	bool                               memory_budget_ext_enabled             = false;
	bool                               compute_subgroup_size_control_enabled = false;
	bool                               shader_image_int64_atomics_enabled    = false;
	bool                               attachment_feedback_loop_enabled      = false;
	bool                               provoking_vertex_last_enabled         = false;
	bool                               maintenance8_enabled                  = false;
	bool                               robust_buffer_access_enabled          = false;
	bool                               r8_srgb_views                         = false;
	bool                               r8g8_srgb_views                       = false;
	bool                               conditional_rendering_enabled         = false;
	bool                               sampler_filter_minmax_enabled         = false;
	bool                               occlusion_queries_enabled             = false;
	bool                               occlusion_query_precise_enabled       = false;
	bool                               supports_block_texel_view              = false;
	bool                                      mesh_shader_enabled                   = false;
	vk::PhysicalDeviceMeshShaderPropertiesEXT mesh_shader_properties                = {};
	uint32_t                           subgroup_size                         = 0;
	uint32_t                           min_subgroup_size                     = 0;
	uint32_t                           max_subgroup_size                     = 0;
	uint32_t                           max_push_descriptors                  = 0;
	uint32_t                           device_clock_period_ps                = 0;
	vk::DescriptorSetLayout            image_heap_layout                     = nullptr;
	Common::Mutex                      queue_mutex;
	uint32_t                           queue_family = static_cast<uint32_t>(-1);
	vk::Queue                          queue        = nullptr;

	[[nodiscard]] const vk::PhysicalDeviceProperties& GetPhysicalDeviceProperties() const {
		return physical_device_properties;
	}

	[[nodiscard]] vk::FormatProperties GetFormatProperties(vk::Format format) const {
		std::scoped_lock lock(m_format_properties_mutex);
		auto [it, inserted] = m_format_properties.try_emplace(format);
		if (inserted) {
			physical_device.getFormatProperties(format, &it->second);
		}
		return it->second;
	}

	// Sampled views of format filter linearly, as guest samplers may.
	[[nodiscard]] bool SamplesFiltered(vk::Format format) const {
		constexpr vk::FormatFeatureFlags features =
		    vk::FormatFeatureFlagBits::eSampledImage |
		    vk::FormatFeatureFlagBits::eSampledImageFilterLinear;
		return (GetFormatProperties(format).optimalTilingFeatures & features) == features;
	}

	[[nodiscard]] vk::Result GetImageFormatProperties(vk::Format format, vk::ImageType type,
	                                                  vk::ImageTiling            tiling,
	                                                  vk::ImageUsageFlags        usage,
	                                                  vk::ImageCreateFlags       flags,
	                                                  vk::ImageFormatProperties* properties) const {
		using Key = std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
		                       vk::ImageCreateFlags>;
		std::scoped_lock lock(m_image_format_properties_mutex);
		auto [it, inserted] =
		    m_image_format_properties.try_emplace(Key {format, type, tiling, usage, flags});
		if (inserted) {
			it->second.first = physical_device.getImageFormatProperties(format, type, tiling, usage,
			                                                            flags, &it->second.second);
		}
		if (properties != nullptr) {
			*properties = it->second.second;
		}
		return it->second.first;
	}

	[[nodiscard]] bool SupportsComputeWave64() const noexcept {
		return subgroup_size == 64u || compute_subgroup_size_control_enabled;
	}

	[[nodiscard]] vk::DeviceSize StorageMinAlignment() const {
		const auto alignment = physical_device_properties.limits.minStorageBufferOffsetAlignment;
		return alignment != 0 ? alignment : 1;
	}

	[[nodiscard]] bool CreateAllocator();
	void               DestroyAllocator();
	void               LogMemoryBudget() const;
	[[nodiscard]] bool CanReportMemoryUsage() const noexcept { return memory_budget_ext_enabled; }
	// The caches age their contents on this clock: sixtieths of a second since start. Counting
	// frames instead made everything look recently used as soon as the frame rate dropped, which
	// is exactly when memory has to be given back.
	[[nodiscard]] static uint64_t CacheTick() noexcept {
		static const auto start = std::chrono::steady_clock::now();
		return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
		                                 std::chrono::steady_clock::now() - start)
		                                 .count()) *
		       60u / 1000u;
	}
	[[nodiscard]] uint64_t GetDeviceMemoryUsage() const;
	[[nodiscard]] uint64_t GetTotalMemoryBudget() const;
	// Pressure/critical GC thresholds shared by the buffer and texture caches. Pure function of
	// the budget, no VMA calls; both caches recompute these on a smoothed budget so they track a
	// shrinking driver budget instead of being pinned to the startup snapshot.
	struct GcThresholds {
		uint64_t pressure = 0; // 90% of the budget.
		uint64_t critical = 0; // 97% of the budget.
	};
	[[nodiscard]] static GcThresholds ComputeGcThresholds(uint64_t budget);
	// Smooths a budget sample into `smoothed` and returns whether `applied`, the budget the levels
	// are computed from, changed: only when the smoothed budget (never above `initial`) has moved
	// more than 5% away from it.
	[[nodiscard]] static bool AdaptGcBudget(uint64_t sample, uint64_t initial, uint64_t& smoothed,
	                                        uint64_t& applied);
	// Fixed physical capacity of the device-local (or, on integrated, all) heaps. Unlike the live
	// VMA budget this never changes, so the overlay can show a stable reference beside it.
	[[nodiscard]] uint64_t GetDeviceLocalMemoryCapacity() const {
		const bool discrete =
		    physical_device_properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
		uint64_t cap = 0;
		for (uint32_t i = 0; i < physical_device_memory_properties.memoryHeapCount; i++) {
			const bool device_local = static_cast<bool>(
			    physical_device_memory_properties.memoryHeaps[i].flags &
			    vk::MemoryHeapFlagBits::eDeviceLocal);
			if (!discrete || device_local) {
				cap += physical_device_memory_properties.memoryHeaps[i].size;
			}
		}
		return cap;
	}

	struct HeapUsage {
		uint64_t usage        = 0; // Whole process, as reported by VK_EXT_memory_budget.
		uint64_t budget       = 0;
		uint32_t allocations  = 0;
		uint64_t allocation_bytes = 0; // Handed out by the allocator.
		uint64_t block_bytes      = 0; // Taken from the driver by the allocator.
		bool     device_local = false;
	};
	[[nodiscard]] uint32_t GetHeapUsage(std::span<HeapUsage, VK_MAX_MEMORY_HEAPS> heaps) const;
	[[nodiscard]] bool     CreateImage(const vk::ImageCreateInfo& info, VulkanImage& image);
	void                   DeleteImage(VulkanImage& image);
	// Destroys the cached images kept for reuse by CreateImage.
	void FlushImagePool();
	// Diagnostics: device memory held by the reuse pool.
	[[nodiscard]] uint64_t ImagePoolBytes() {
		std::lock_guard lock(m_image_pool_mutex);
		return m_image_pool_bytes;
	}

	uint32_t screen_width  = 0;
	uint32_t screen_height = 0;

private:
	struct PooledImage {
		vk::Format           format     = vk::Format::eUndefined;
		vk::ImageType        image_type = vk::ImageType::e2D;
		vk::Extent3D         extent     = {1, 1, 1};
		uint32_t             layers     = 1;
		uint32_t             mip_levels = 1;
		uint32_t             samples    = 1;
		vk::ImageUsageFlags  usage      = {};
		vk::ImageCreateFlags flags      = {};
		vk::Image            image      = nullptr;
		VmaAllocation        allocation = nullptr;
		uint64_t             size       = 0;
	};
	std::mutex               m_image_pool_mutex;
	std::vector<PooledImage> m_image_pool;
	uint64_t                 m_image_pool_bytes = 0;

	mutable std::mutex                                 m_format_properties_mutex;
	mutable std::map<vk::Format, vk::FormatProperties> m_format_properties;
	mutable std::mutex                                 m_image_format_properties_mutex;
	mutable std::map<std::tuple<vk::Format, vk::ImageType, vk::ImageTiling, vk::ImageUsageFlags,
	                            vk::ImageCreateFlags>,
	                 std::pair<vk::Result, vk::ImageFormatProperties>>
	    m_image_format_properties;
};

struct VulkanImageState {
	vk::PipelineStageFlags2 pl_stage    = vk::PipelineStageFlagBits2::eAllCommands;
	vk::AccessFlags2        access_mask = vk::AccessFlagBits2::eNone;
	vk::ImageLayout         layout      = vk::ImageLayout::eUndefined;
};

struct VulkanImage {
	VulkanImage() = default;
	KYTY_CLASS_NO_COPY(VulkanImage);

	vk::Format                    format      = vk::Format::eUndefined;
	vk::ImageType                 image_type  = vk::ImageType::e2D;
	vk::Extent3D                  extent      = {1, 1, 1};
	uint32_t                      layers      = 1;
	uint32_t                      mip_levels  = 1;
	uint32_t                      samples     = 1;
	vk::ImageUsageFlags           usage       = {};
	vk::ImageCreateFlags          flags       = {};
	vk::Image                     image       = nullptr;
	VulkanImageState              state;
	std::vector<VulkanImageState> subresource_states;
	VmaAllocation                allocation = nullptr;
	bool                         poolable   = false;
};



} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICCONTEXT_H_ */
