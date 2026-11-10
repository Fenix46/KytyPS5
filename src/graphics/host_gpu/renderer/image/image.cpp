#include "graphics/host_gpu/renderer/image/image.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/tiler.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "kernel/memory.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <memory>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

[[nodiscard]] vk::ImageType HostImageType(Prospero::ImageType type) {
	switch (type) {
		case Prospero::ImageType::kColor3D: return vk::ImageType::e3D;
		case Prospero::ImageType::kColor2D: return vk::ImageType::e2D;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(type));
	}
}

[[nodiscard]] vk::ImageCreateFlags
ImageCreateFlags(const GraphicContext& graphics, const ImageInfo& info, vk::ImageUsageFlags usage) {
	vk::ImageCreateFlags flags {};
	if (DepthAspectTransferFormat(info.pixel_format) == vk::Format::eUndefined) {
		flags |= vk::ImageCreateFlagBits::eMutableFormat;
		if (info.IsBlock() && graphics.supports_block_texel_view) {
			flags |= vk::ImageCreateFlagBits::eBlockTexelViewCompatible;
		}
	}
	if (usage & vk::ImageUsageFlagBits::eStorage) {
		flags |= vk::ImageCreateFlagBits::eExtendedUsage;
	}
	if (info.IsVolume()) {
		flags |= vk::ImageCreateFlagBits::e2DArrayCompatible;
	}
	return flags;
}

[[nodiscard]] bool HasFormatFeature(vk::FormatProperties      properties,
                                    vk::FormatFeatureFlagBits feature) {
	return static_cast<bool>(properties.optimalTilingFeatures & feature);
}

[[nodiscard]] vk::ImageUsageFlags ImageUsageFlags(GraphicContext& graphics, const ImageInfo& info) {
	auto usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst;
	if (info.IsBlock()) {
		usage |= vk::ImageUsageFlagBits::eSampled;
		if (graphics.supports_block_texel_view) {
			const auto storage = usage | vk::ImageUsageFlagBits::eStorage;
			if (graphics.GetImageFormatProperties(info.pixel_format, HostImageType(info.type),
			                                      vk::ImageTiling::eOptimal, storage,
			                                      ImageCreateFlags(graphics, info, storage),
			                                      nullptr) == vk::Result::eSuccess) {
				usage = storage;
			} else {
				static std::atomic_flag warned = ATOMIC_FLAG_INIT;
				if (!warned.test_and_set(std::memory_order_relaxed)) {
					Log::WriteToConsoleAndLog(fmt::format(
					    "Warning: format {} does not support storage access; block-compressed "
					    "textures written by the guest will not render.\n",
					    vk::to_string(info.pixel_format)));
				}
			}
		}
		return usage;
	}
	const auto properties = graphics.GetFormatProperties(info.pixel_format);
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eSampledImage)) {
		usage |= vk::ImageUsageFlagBits::eSampled;
	}
	if (DepthAspectTransferFormat(info.pixel_format) != vk::Format::eUndefined) {
		usage |= vk::ImageUsageFlagBits::eDepthStencilAttachment;
		if (graphics.attachment_feedback_loop_enabled && (usage & vk::ImageUsageFlagBits::eSampled)) {
			usage |= vk::ImageUsageFlagBits::eAttachmentFeedbackLoopEXT;
		}
		return usage;
	}
	if (HasFormatFeature(properties, vk::FormatFeatureFlagBits::eColorAttachment)) {
		usage |= vk::ImageUsageFlagBits::eColorAttachment;
	}
	if (graphics.GetPhysicalDeviceProperties().limits.storageImageSampleCounts &
	    static_cast<vk::SampleCountFlagBits>(info.samples)) {
		usage |= vk::ImageUsageFlagBits::eStorage;
	}
	return usage;
}

void ValidateOptionalRange(GuestRange range, const char* name) {
	if (!range.ValidOrEmpty()) {
		EXIT("invalid %s image range: address=0x%016llx size=0x%016llx\n", name,
		     static_cast<unsigned long long>(range.address),
		     static_cast<unsigned long long>(range.size));
	}
}

} // namespace

vk::ImageAspectFlags Image::FullAspectMask(vk::Format format) noexcept {
	switch (format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eX8D24UnormPack32:
		case vk::Format::eD32Sfloat: return vk::ImageAspectFlagBits::eDepth;
		case vk::Format::eS8Uint: return vk::ImageAspectFlagBits::eStencil;
		case vk::Format::eD16UnormS8Uint:
		case vk::Format::eD24UnormS8Uint:
		case vk::Format::eD32SfloatS8Uint:
			return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
		default: return vk::ImageAspectFlagBits::eColor;
	}
}

Image::Barriers Image::GetBarriers(vk::ImageLayout                      destination_layout,
                                   vk::AccessFlags2                     destination_access,
                                   vk::PipelineStageFlags2              destination_stage,
                                   std::optional<ImageSubresourceRange> range) {
	auto& state              = backing.state;
	auto& subresource_states = backing.subresource_states;
	if (range && info.IsVolume()) {
		range->base_layer  = 0;
		range->layer_count = 1;
	}

	const bool partial =
	    range && (range->base_level != 0 || range->level_count != info.resources.levels ||
	              range->base_layer != 0 || range->layer_count != info.resources.layers);
	const bool has_subresource_states = !subresource_states.empty();

	Barriers barriers;
	if (partial || has_subresource_states) {
		if (!has_subresource_states) {
			subresource_states.resize(info.resources.levels * info.resources.layers, state);
		}

		const uint32_t base_level  = partial ? range->base_level : 0;
		const uint32_t level_count = partial ? range->level_count : info.resources.levels;
		const uint32_t base_layer  = partial ? range->base_layer : 0;
		const uint32_t layer_count = partial ? range->layer_count : info.resources.layers;
		for (uint32_t level = base_level; level < base_level + level_count; level++) {
			for (uint32_t layer = base_layer; layer < base_layer + layer_count; layer++) {
				const auto index = level * info.resources.layers + layer;
				EXIT_IF(index >= subresource_states.size());
				auto& subresource_state = subresource_states[index];

				constexpr auto write_access = vk::AccessFlagBits2::eTransferWrite |
				                              vk::AccessFlagBits2::eShaderWrite |
				                              vk::AccessFlagBits2::eMemoryWrite;
				const bool     repeated_write =
				    static_cast<bool>(subresource_state.access_mask & write_access);
				if (subresource_state.layout != destination_layout ||
				    subresource_state.access_mask != destination_access || repeated_write) {
					vk::ImageMemoryBarrier2 barrier {};
					barrier.srcStageMask                    = subresource_state.pl_stage;
					barrier.srcAccessMask                   = subresource_state.access_mask;
					barrier.dstStageMask                    = destination_stage;
					barrier.dstAccessMask                   = destination_access;
					barrier.oldLayout                       = subresource_state.layout;
					barrier.newLayout                       = destination_layout;
					barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
					barrier.image                           = backing.image;
					barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
					barrier.subresourceRange.baseMipLevel   = level;
					barrier.subresourceRange.levelCount     = 1;
					barrier.subresourceRange.baseArrayLayer = layer;
					barrier.subresourceRange.layerCount     = 1;
					barriers.push_back(barrier);
					subresource_state = {destination_stage, destination_access, destination_layout};
				}
			}
		}

		if (!partial) {
			subresource_states.clear();
		}
	} else {
		constexpr auto write_access   = vk::AccessFlagBits2::eTransferWrite |
		                                vk::AccessFlagBits2::eShaderWrite |
		                                vk::AccessFlagBits2::eMemoryWrite;
		const bool     repeated_write = static_cast<bool>(state.access_mask & write_access);
		if (state.layout == destination_layout && state.access_mask == destination_access &&
		    !repeated_write) {
			return {};
		}

		vk::ImageMemoryBarrier2 barrier {};
		barrier.srcStageMask                    = state.pl_stage;
		barrier.srcAccessMask                   = state.access_mask;
		barrier.dstStageMask                    = destination_stage;
		barrier.dstAccessMask                   = destination_access;
		barrier.oldLayout                       = state.layout;
		barrier.newLayout                       = destination_layout;
		barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
		barrier.image                           = backing.image;
		barrier.subresourceRange.aspectMask     = FullAspectMask(backing.format);
		barrier.subresourceRange.baseMipLevel   = 0;
		barrier.subresourceRange.levelCount     = VK_REMAINING_MIP_LEVELS;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount     = VK_REMAINING_ARRAY_LAYERS;
		barriers.push_back(barrier);
	}

	// Resident heaps sample their images in the read state; only leaving it invalidates them.
	if (std::ranges::any_of(barriers, [&](const vk::ImageMemoryBarrier2& barrier) {
		    return barrier.oldLayout == ReadLayout() &&
		           barrier.srcAccessMask == vk::AccessFlagBits2::eShaderRead;
	    })) {
		HeapChanged();
	}
	state = {destination_stage, destination_access, destination_layout};
	return barriers;
}

void Image::Transit(vk::ImageLayout destination_layout, vk::AccessFlags2 destination_access,
                    std::optional<ImageSubresourceRange> range, vk::CommandBuffer command_buffer) {
	const auto transfer_access =
	    vk::AccessFlagBits2::eTransferRead | vk::AccessFlagBits2::eTransferWrite;
	vk::PipelineStageFlags2 destination_stage {};
	if (static_cast<bool>(destination_access & transfer_access)) {
		destination_stage |= vk::PipelineStageFlagBits2::eTransfer;
	}
	if (!destination_access ||
	    static_cast<bool>(destination_access & ~vk::AccessFlags2 {transfer_access})) {
		destination_stage |=
		    vk::PipelineStageFlagBits2::eAllGraphics | vk::PipelineStageFlagBits2::eComputeShader;
	}
	const auto barriers =
	    GetBarriers(destination_layout, destination_access, destination_stage, range);
	if (barriers.empty()) {
		return;
	}
	m_scheduler.EndRendering();
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
	dependency.pImageMemoryBarriers    = barriers.data();
	command_buffer.pipelineBarrier2(dependency);
}

void Image::Upload(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer, uint64_t offset,
                   uint64_t size) {
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	if (IsScaled()) {
		UploadScaled(copies, buffer, offset, size);
		return;
	}
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask        = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask       = vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyBufferToImage(buffer, backing.image, vk::ImageLayout::eTransferDstOptimal,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Download(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer,
                     uint64_t offset, uint64_t size) {
	EXIT_IF(copies.empty() || buffer == nullptr || size == 0);
	if (IsScaled()) {
		DownloadScaled(copies, buffer, offset, size);
		return;
	}
	m_scheduler.EndRendering();
	vk::BufferMemoryBarrier2 buffer_barrier {};
	buffer_barrier.srcStageMask = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.srcAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	buffer_barrier.dstStageMask        = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.dstAccessMask       = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	buffer_barrier.buffer              = buffer;
	buffer_barrier.offset              = offset;
	buffer_barrier.size                = size;
	const auto image_barriers =
	    GetBarriers(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	                vk::PipelineStageFlagBits2::eCopy, {});
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &buffer_barrier;
	dependency.imageMemoryBarrierCount  = static_cast<uint32_t>(image_barriers.size());
	dependency.pImageMemoryBarriers     = image_barriers.data();
	auto command                        = m_scheduler.Current().Handle();
	command.pipelineBarrier2(dependency);
	command.copyImageToBuffer(backing.image, vk::ImageLayout::eTransferSrcOptimal, buffer,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	buffer_barrier.srcStageMask  = vk::PipelineStageFlagBits2::eCopy;
	buffer_barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
	buffer_barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
	buffer_barrier.dstAccessMask =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	dependency.imageMemoryBarrierCount = 0;
	dependency.pImageMemoryBarriers    = nullptr;
	command.pipelineBarrier2(dependency);
}

std::pair<uint32_t, uint32_t> Image::SanitizeCopyLayers(const Image& source,
                                                        const Image& destination, uint32_t depth) {
	const auto source_type        = source.backing.image_type;
	const auto destination_type   = destination.backing.image_type;
	uint32_t   source_layers      = source.backing.layers;
	uint32_t   destination_layers = destination.backing.layers;
	if (source_type == vk::ImageType::e3D) {
		source_layers = 1;
	}
	if (destination_type == vk::ImageType::e3D) {
		destination_layers = 1;
	}
	if (source_type == destination_type) {
		source_layers = destination_layers = std::min(source_layers, destination_layers);
	} else if (source_type == vk::ImageType::e2D) {
		source_layers = depth;
	} else {
		destination_layers = depth;
	}
	return {source_layers, destination_layers};
}

void Image::CopyImage(Image& source) {
	EXIT_IF(source.backing.samples != backing.samples);
	m_scheduler.EndRendering();
	const uint32_t levels     = std::min(source.backing.mip_levels, backing.mip_levels);
	if ((IsScaled() || source.IsScaled()) && CanBlitFrom(source)) {
		// The same guest area has a different size in each image, so it is scaled across.
		const auto aspects = FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
		const auto layers  = std::min(source.backing.layers, backing.layers);
		std::array<vk::ImageBlit, std::tuple_size_v<decltype(ImageInfo::mip_layout)>> blits {};
		for (uint32_t level = 0; level < levels; level++) {
			const vk::Extent3D guest {
			    std::min(std::max(source.info.extent.width >> level, 1u),
			             std::max(info.extent.width >> level, 1u)),
			    std::min(std::max(source.info.extent.height >> level, 1u),
			             std::max(info.extent.height >> level, 1u)),
			    1};
			auto& blit          = blits[level];
			blit.srcSubresource = vk::ImageSubresourceLayers {aspects, level, 0, layers};
			blit.dstSubresource = vk::ImageSubresourceLayers {aspects, level, 0, layers};
			blit.srcOffsets     = source.HostRect(level, {}, guest);
			blit.dstOffsets     = HostRect(level, {}, guest);
		}
		auto command = m_scheduler.Current().Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
		        command);
		command.blitImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  backing.image, vk::ImageLayout::eTransferDstOptimal, levels, blits.data(),
		                  BlitFilter(aspects));
		Transit(vk::ImageLayout::eGeneral,
		        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
		return;
	}
	uint32_t       base_depth = source.backing.extent.depth;
	if (source.backing.image_type == backing.image_type) {
		base_depth = std::min(base_depth, backing.extent.depth);
	} else if (backing.image_type == vk::ImageType::e3D) {
		base_depth = backing.extent.depth;
	}
	const auto source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	std::array<vk::ImageCopy, std::tuple_size_v<decltype(ImageInfo::mip_layout)>> copies {};
	for (uint32_t level = 0; level < levels; level++) {
		// Overlapping cache images can differ in size; copying the source extent into a smaller
		// destination writes out of bounds and loses the device.
		const auto width  = std::min(std::max(source.backing.extent.width >> level, 1u),
		                             std::max(backing.extent.width >> level, 1u));
		const auto height = std::min(std::max(source.backing.extent.height >> level, 1u),
		                             std::max(backing.extent.height >> level, 1u));
		const auto depth  = std::max(base_depth >> level, 1u);
		const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);

		auto& copy          = copies[level];
		copy.srcSubresource = {source_aspect, level, 0, 1};
		copy.dstSubresource = {destination_aspect, level, 0, 1};
		if (source.backing.image_type == backing.image_type) {
			if (source.backing.image_type == vk::ImageType::e3D) {
				copy.extent = {width, height, depth};
			} else {
				copy.srcSubresource.layerCount = std::min(source_layers, destination_layers);
				copy.dstSubresource.layerCount = copy.srcSubresource.layerCount;
				copy.extent                    = {width, height, 1};
			}
		} else if (source.backing.image_type == vk::ImageType::e2D) {
			copy.srcSubresource.layerCount = source_layers;
			copy.extent                    = {width, height, source_layers};
		} else {
			copy.dstSubresource.layerCount = destination_layers;
			copy.extent                    = {width, height, destination_layers};
		}
	}
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, levels, copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::Resolve(Image& source, const ImageSubresourceRange& source_range,
                    const ImageSubresourceRange& destination_range) {
	EXIT_IF(backing.samples != 1 || source.backing.image_type != vk::ImageType::e2D ||
	        backing.image_type != vk::ImageType::e2D || source_range.level_count != 1 ||
	        destination_range.level_count != 1 ||
	        source_range.base_level >= source.backing.mip_levels ||
	        destination_range.base_level >= backing.mip_levels ||
	        source_range.base_layer >= source.backing.layers ||
	        destination_range.base_layer >= backing.layers);
	const auto layers       = std::min({source_range.layer_count, destination_range.layer_count,
	                                    source.backing.layers - source_range.base_layer,
	                                    backing.layers - destination_range.base_layer});
	const auto source_width = std::max(source.backing.extent.width >> source_range.base_level, 1u);
	const auto source_height =
	    std::max(source.backing.extent.height >> source_range.base_level, 1u);
	const auto destination_width =
	    std::max(backing.extent.width >> destination_range.base_level, 1u);
	const auto destination_height =
	    std::max(backing.extent.height >> destination_range.base_level, 1u);
	const bool copy = source.backing.samples == 1;
	EXIT_IF(layers == 0 ||
	        (copy ? !ImageViewOps::FormatsCompatible(source.backing.format, backing.format)
	              : source.backing.format != backing.format));
	auto resolved_source_range             = source_range;
	auto resolved_destination_range        = destination_range;
	resolved_source_range.layer_count      = layers;
	resolved_destination_range.layer_count = layers;
	const vk::Extent3D resolve_extent {std::min(source_width, destination_width),
	                                   std::min(source_height, destination_height), 1};

	m_scheduler.EndRendering();
	auto command = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
	               resolved_source_range, command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite,
	        resolved_destination_range, command);
	if (copy) {
		vk::ImageCopy region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, region);
	} else {
		vk::ImageResolve region {};
		region.srcSubresource = {vk::ImageAspectFlagBits::eColor, resolved_source_range.base_level,
		                         resolved_source_range.base_layer, layers};
		region.dstSubresource = {vk::ImageAspectFlagBits::eColor,
		                         resolved_destination_range.base_level,
		                         resolved_destination_range.base_layer, layers};
		region.extent         = resolve_extent;
		command.resolveImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                     backing.image, vk::ImageLayout::eTransferDstOptimal, region);
	}
}

uint32_t Image::CopyRows(uint64_t row_size, uint32_t rows, uint64_t capacity) noexcept {
	if (row_size == 0 || rows == 0 || row_size > capacity) {
		return 0;
	}
	return static_cast<uint32_t>(std::min<uint64_t>(rows, capacity / row_size));
}

void Image::CopyImageWithBuffer(Image& source, Buffer& buffer, TileManager& tiler) {
	EXIT_IF(buffer.Handle() == nullptr || source.backing.samples != 1 || backing.samples != 1);
	if ((IsScaled() || source.IsScaled()) &&
	    (source.backing.extent.width > backing.extent.width ||
	     source.backing.extent.height > backing.extent.height)) {
		// Bytes cannot be reinterpreted across two host sizes; the destination keeps its contents.
		static std::atomic_flag warned = ATOMIC_FLAG_INIT;
		if (!warned.test_and_set(std::memory_order_relaxed)) {
			std::printf("[scale] skipped a cross-format copy from %ux%u (host %ux%u) into %ux%u "
			            "(host %ux%u)\n",
			            source.info.extent.width, source.info.extent.height,
			            source.backing.extent.width, source.backing.extent.height, info.extent.width,
			            info.extent.height, backing.extent.width, backing.extent.height);
		}
		return;
	}
	m_scheduler.EndRendering();
	const uint32_t levels = std::min(source.backing.mip_levels, backing.mip_levels);
	const auto     source_aspect =
	    FullAspectMask(source.backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto destination_aspect =
	    FullAspectMask(backing.format) & ~vk::ImageAspectFlagBits::eStencil;
	const auto     source_bytes      = DepthAspectTransferBytes(source.backing.format) != 0
	                                       ? DepthAspectTransferBytes(source.backing.format)
	                                       : source.info.bytes_per_block;
	const auto     destination_bytes = DepthAspectTransferBytes(backing.format) != 0
	                                       ? DepthAspectTransferBytes(backing.format)
	                                       : info.bytes_per_block;
	const uint32_t source_block      = source.info.IsBlock() ? 4u : 1u;
	const uint32_t destination_block = info.IsBlock() ? 4u : 1u;
	EXIT_IF(levels == 0 || source_bytes == 0 || source_bytes != destination_bytes ||
	        source_block != destination_block);
	const auto source_transform = source.info.GetColorTransform();
	const auto target_transform = info.GetColorTransform();

	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.dstStageMask        = vk::PipelineStageFlagBits2::eTransfer;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer.Handle();
	barrier.offset              = 0;
	vk::DependencyInfo dependency {};
	dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	auto command                        = m_scheduler.Current().Handle();
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	for (uint32_t level = 0; level < levels; level++) {
		const auto width             = std::max(source.backing.extent.width >> level, 1u);
		const auto height            = std::max(source.backing.extent.height >> level, 1u);
		const auto source_depth      = source.backing.image_type == vk::ImageType::e3D
		                                   ? std::max(source.backing.extent.depth >> level, 1u)
		                                   : source.backing.layers;
		const auto destination_depth = backing.image_type == vk::ImageType::e3D
		                                   ? std::max(backing.extent.depth >> level, 1u)
		                                   : backing.layers;
		const auto slices            = std::min(source_depth, destination_depth);
		const auto block_rows        = (height + source_block - 1) / source_block;
		const auto row_size =
		    static_cast<uint64_t>((width + source_block - 1) / source_block) * source_bytes;
		const auto rows_per_copy = CopyRows(row_size, block_rows, buffer.Size());
		EXIT_IF(slices == 0 || rows_per_copy == 0);
		for (uint32_t slice = 0; slice < slices; slice++) {
			for (uint32_t block_row = 0; block_row < block_rows; block_row += rows_per_copy) {
				const auto          copy_rows   = std::min(rows_per_copy, block_rows - block_row);
				const auto          y           = block_row * source_block;
				const auto          copy_height = std::min(copy_rows * source_block, height - y);
				const auto          copy_size   = row_size * copy_rows;
				vk::BufferImageCopy source_copy {};
				source_copy.imageSubresource = {
				    source_aspect, level,
				    source.backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				source_copy.imageOffset           = {0, static_cast<int32_t>(y),
				                                     source.backing.image_type == vk::ImageType::e3D
				                                         ? static_cast<int32_t>(slice)
				                                         : 0};
				source_copy.imageExtent           = {width, copy_height, 1};
				auto destination_copy             = source_copy;
				destination_copy.imageSubresource = {
				    destination_aspect, level,
				    backing.image_type == vk::ImageType::e3D ? 0u : slice, 1};
				destination_copy.imageOffset.z =
				    backing.image_type == vk::ImageType::e3D ? static_cast<int32_t>(slice) : 0;
				barrier.size          = copy_size;
				barrier.srcAccessMask = vk::AccessFlagBits2::eTransferRead;
				barrier.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
				command.pipelineBarrier2(dependency);
				command.copyImageToBuffer(source.backing.image,
				                          vk::ImageLayout::eTransferSrcOptimal, buffer.Handle(),
				                          source_copy);
				if (source_transform != target_transform) {
					const TileManager::Result bytes {buffer.Handle(), 0, copy_size};
					if (source_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, source_transform, false);
					}
					if (target_transform != ColorTransform::None) {
						tiler.TransformColor(bytes, bytes, target_transform, true);
					}
				} else {
					barrier.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
					barrier.dstAccessMask = vk::AccessFlagBits2::eTransferRead;
					command.pipelineBarrier2(dependency);
				}
				command.copyBufferToImage(buffer.Handle(), backing.image,
				                          vk::ImageLayout::eTransferDstOptimal, destination_copy);
			}
		}
	}
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

void Image::CopyMip(Image& source, uint32_t mip, uint32_t layer) {
	EXIT_IF(source.backing.samples != backing.samples || mip >= backing.mip_levels ||
	        layer >= backing.layers);
	m_scheduler.EndRendering();
	const auto width  = std::max(backing.extent.width >> mip, 1u);
	const auto height = std::max(backing.extent.height >> mip, 1u);
	const auto depth  = std::max(backing.extent.depth >> mip, 1u);
	const auto [source_layers, destination_layers] = SanitizeCopyLayers(source, *this, depth);
	const auto aspects                             = FullAspectMask(source.backing.format);
	EXIT_IF(aspects != FullAspectMask(backing.format));
	if (width != source.backing.extent.width || height != source.backing.extent.height) {
		// Only a scaled image differs in size from the level it stands for.
		EXIT_IF((!IsScaled() && !source.IsScaled()) || !CanBlitFrom(source) ||
		        source_layers != destination_layers);
		vk::ImageBlit blit {};
		blit.srcSubresource = vk::ImageSubresourceLayers {aspects, 0, 0, source_layers};
		blit.dstSubresource = vk::ImageSubresourceLayers {aspects, mip, layer, destination_layers};
		blit.srcOffsets[1]  = vk::Offset3D {static_cast<int32_t>(source.backing.extent.width),
		                                    static_cast<int32_t>(source.backing.extent.height), 1};
		blit.dstOffsets[1]  =
		    vk::Offset3D {static_cast<int32_t>(width), static_cast<int32_t>(height), 1};
		auto command = m_scheduler.Current().Handle();
		Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {},
		        command);
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		command.blitImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  backing.image, vk::ImageLayout::eTransferDstOptimal, blit,
		                  BlitFilter(aspects));
		Transit(vk::ImageLayout::eGeneral,
		        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
		return;
	}
	std::array<vk::ImageCopy, 2> copies {};
	uint32_t                     copy_count = 0;
	for (const auto aspect: {vk::ImageAspectFlagBits::eColor, vk::ImageAspectFlagBits::eDepth,
	                         vk::ImageAspectFlagBits::eStencil}) {
		if (!static_cast<bool>(aspects & aspect)) {
			continue;
		}
		auto& copy          = copies[copy_count++];
		copy.srcSubresource = {aspect, 0, 0, source_layers};
		copy.dstSubresource = {aspect, mip, layer, destination_layers};
		copy.extent         = {width, height, depth};
	}
	auto command = m_scheduler.Current().Handle();
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
	               command);
	command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
	                  vk::ImageLayout::eTransferDstOptimal, copy_count, copies.data());
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
}

namespace {

[[nodiscard]] bool CanBlit(const GraphicContext& graphics, vk::Format format) {
	constexpr auto both = vk::FormatFeatureFlagBits::eBlitSrc | vk::FormatFeatureFlagBits::eBlitDst;
	return (graphics.GetFormatProperties(format).optimalTilingFeatures & both) == both;
}

// Color and depth targets have to shrink together, so every depth format must take a blit.
[[nodiscard]] bool ScalingSupported(const GraphicContext& graphics) {
	static const bool supported = [&] {
		const bool depth = CanBlit(graphics, vk::Format::eD16Unorm) &&
		                   CanBlit(graphics, vk::Format::eD32Sfloat) &&
		                   CanBlit(graphics, vk::Format::eD32SfloatS8Uint);
		if (!depth) {
			std::printf("[scale] --internal-resolution is ignored: this GPU cannot scale depth "
			            "images\n");
		} else {
			std::printf("[scale] internal resolution capped at %u lines\n",
			            Config::GetInternalResolution());
		}
		return depth;
	}();
	return supported;
}

[[nodiscard]] bool IsNumericFormat(vk::Format format, const char* kind) {
	// Format names end in their numeric kind, and packed ones in the pack size after it.
	return vk::to_string(format).find(kind) != std::string::npos;
}

// A guest-sized stand-in for one transfer of a scaled image.
[[nodiscard]] std::shared_ptr<VulkanImage> CreateTransferImage(GraphicContext& graphics,
                                                               const Image&    image) {
	vk::ImageCreateInfo create {};
	create.usage         = image.backing.usage;
	create.flags         = image.backing.flags;
	create.imageType     = image.backing.image_type;
	create.extent        = image.info.extent;
	create.mipLevels     = image.backing.mip_levels;
	create.arrayLayers   = image.backing.layers;
	create.format        = image.backing.format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.samples       = vk::SampleCountFlagBits::e1;
	auto transfer        = std::make_shared<VulkanImage>();
	if (!graphics.CreateImage(create, *transfer)) {
		EXIT("failed to create a transfer image: extent=%ux%u format=%d\n", create.extent.width,
		     create.extent.height, static_cast<int>(create.format));
	}
	return transfer;
}

void TransferImageBarrier(vk::CommandBuffer command, const VulkanImage& image,
                          vk::ImageAspectFlags aspects, vk::ImageLayout from, vk::ImageLayout to) {
	const bool              fresh = from == vk::ImageLayout::eUndefined;
	vk::ImageMemoryBarrier2 barrier {};
	barrier.srcStageMask =
	    fresh ? vk::PipelineStageFlagBits2::eNone : vk::PipelineStageFlagBits2::eAllTransfer;
	barrier.srcAccessMask =
	    fresh ? vk::AccessFlagBits2::eNone : vk::AccessFlagBits2::eTransferWrite;
	barrier.dstStageMask  = vk::PipelineStageFlagBits2::eAllTransfer;
	barrier.dstAccessMask = to == vk::ImageLayout::eTransferSrcOptimal
	                            ? vk::AccessFlagBits2::eTransferRead
	                            : vk::AccessFlagBits2::eTransferWrite;
	barrier.oldLayout           = from;
	barrier.newLayout           = to;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.image               = image.image;
	barrier.subresourceRange    =
	    vk::ImageSubresourceRange {aspects, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS};
	vk::DependencyInfo dependency {};
	dependency.imageMemoryBarrierCount = 1;
	dependency.pImageMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

// Hands a buffer range to the transfer stage (begin) or back to everything else.
void TransferBufferBarrier(vk::CommandBuffer command, vk::Buffer buffer, uint64_t offset,
                           uint64_t size, bool begin, bool write) {
	const vk::AccessFlags2 host =
	    vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
	const vk::AccessFlags2 access =
	    write ? vk::AccessFlagBits2::eTransferWrite : vk::AccessFlagBits2::eTransferRead;
	vk::BufferMemoryBarrier2 barrier {};
	barrier.srcStageMask        = begin ? vk::PipelineStageFlagBits2::eAllCommands
	                                    : vk::PipelineStageFlagBits2::eAllTransfer;
	barrier.srcAccessMask       = begin ? host : access;
	barrier.dstStageMask        = begin ? vk::PipelineStageFlagBits2::eAllTransfer
	                                    : vk::PipelineStageFlagBits2::eAllCommands;
	barrier.dstAccessMask       = begin ? access : host;
	barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	barrier.buffer              = buffer;
	barrier.offset              = offset;
	barrier.size                = size;
	vk::DependencyInfo dependency {};
	dependency.bufferMemoryBarrierCount = 1;
	dependency.pBufferMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

} // namespace

bool Image::CanBlitFrom(const Image& source) const {
	if (source.backing.samples != 1 || backing.samples != 1 ||
	    source.backing.image_type != vk::ImageType::e2D ||
	    backing.image_type != vk::ImageType::e2D ||
	    !CanBlit(m_graphics, source.backing.format) || !CanBlit(m_graphics, backing.format)) {
		return false;
	}
	if (FullAspectMask(source.backing.format) != vk::ImageAspectFlagBits::eColor ||
	    FullAspectMask(backing.format) != vk::ImageAspectFlagBits::eColor) {
		return source.backing.format == backing.format;
	}
	return IsNumericFormat(source.backing.format, "Uint") ==
	           IsNumericFormat(backing.format, "Uint") &&
	       IsNumericFormat(source.backing.format, "Sint") == IsNumericFormat(backing.format, "Sint");
}

vk::Filter Image::BlitFilter(vk::ImageAspectFlags aspects) const {
	const bool linear = static_cast<bool>(
	    m_graphics.GetFormatProperties(backing.format).optimalTilingFeatures &
	    vk::FormatFeatureFlagBits::eSampledImageFilterLinear);
	return aspects == vk::ImageAspectFlagBits::eColor && linear ? vk::Filter::eLinear
	                                                           : vk::Filter::eNearest;
}

std::array<vk::Offset3D, 2> Image::HostRect(uint32_t level, vk::Offset3D begin,
                                            vk::Extent3D extent) const {
	const auto map = [level](int32_t first, uint32_t count, uint32_t guest_size,
	                         uint32_t host_size) -> std::pair<int32_t, int32_t> {
		const uint64_t guest = std::max(guest_size >> level, 1u);
		const uint64_t host  = std::max(host_size >> level, 1u);
		const uint64_t end   = static_cast<uint64_t>(first) + count;
		auto           low   = static_cast<uint64_t>(first) * host / guest;
		auto           high  = std::min(host, (end * host + guest - 1) / guest);
		low                  = std::min(low, host - 1);
		high                 = std::max(high, low + 1);
		return {static_cast<int32_t>(low), static_cast<int32_t>(high)};
	};
	const auto [left, right] = map(begin.x, extent.width, info.extent.width, backing.extent.width);
	const auto [top, bottom] =
	    map(begin.y, extent.height, info.extent.height, backing.extent.height);
	return {vk::Offset3D {left, top, 0}, vk::Offset3D {right, bottom, 1}};
}

void Image::UploadScaled(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer,
                         uint64_t offset, uint64_t size) {
	EXIT_IF(backing.samples != 1);
	m_scheduler.EndRendering();
	const auto aspects  = FullAspectMask(backing.format);
	const auto transfer = CreateTransferImage(m_graphics, *this);
	auto       command  = m_scheduler.Current().Handle();
	TransferBufferBarrier(command, buffer, offset, size, true, false);
	TransferImageBarrier(command, *transfer, aspects, vk::ImageLayout::eUndefined,
	                     vk::ImageLayout::eTransferDstOptimal);
	command.copyBufferToImage(buffer, transfer->image, vk::ImageLayout::eTransferDstOptimal,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	TransferImageBarrier(command, *transfer, aspects, vk::ImageLayout::eTransferDstOptimal,
	                     vk::ImageLayout::eTransferSrcOptimal);
	Transit(vk::ImageLayout::eTransferDstOptimal, vk::AccessFlagBits2::eTransferWrite, {}, command);
	for (const auto& copy: copies) {
		vk::ImageBlit blit {};
		blit.srcSubresource = copy.imageSubresource;
		blit.dstSubresource = copy.imageSubresource;
		blit.srcOffsets[0]  = copy.imageOffset;
		blit.srcOffsets[1]  =
		    vk::Offset3D {copy.imageOffset.x + static_cast<int32_t>(copy.imageExtent.width),
		                  copy.imageOffset.y + static_cast<int32_t>(copy.imageExtent.height), 1};
		blit.dstOffsets =
		    HostRect(copy.imageSubresource.mipLevel, copy.imageOffset, copy.imageExtent);
		command.blitImage(transfer->image, vk::ImageLayout::eTransferSrcOptimal, backing.image,
		                  vk::ImageLayout::eTransferDstOptimal, blit,
		                  BlitFilter(copy.imageSubresource.aspectMask));
	}
	TransferBufferBarrier(command, buffer, offset, size, false, false);
	Transit(vk::ImageLayout::eGeneral,
	        vk::AccessFlagBits2::eShaderRead | vk::AccessFlagBits2::eTransferRead, {}, command);
	m_scheduler.DeferOperation(
	    [graphics = &m_graphics, transfer] { graphics->DeleteImage(*transfer); });
}

void Image::DownloadScaled(std::span<const vk::BufferImageCopy> copies, vk::Buffer buffer,
                           uint64_t offset, uint64_t size) {
	EXIT_IF(backing.samples != 1);
	m_scheduler.EndRendering();
	const auto aspects  = FullAspectMask(backing.format);
	const auto transfer = CreateTransferImage(m_graphics, *this);
	auto       command  = m_scheduler.Current().Handle();
	TransferBufferBarrier(command, buffer, offset, size, true, true);
	TransferImageBarrier(command, *transfer, aspects, vk::ImageLayout::eUndefined,
	                     vk::ImageLayout::eTransferDstOptimal);
	Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {}, command);
	for (const auto& copy: copies) {
		vk::ImageBlit blit {};
		blit.srcSubresource = copy.imageSubresource;
		blit.dstSubresource = copy.imageSubresource;
		blit.srcOffsets =
		    HostRect(copy.imageSubresource.mipLevel, copy.imageOffset, copy.imageExtent);
		blit.dstOffsets[0] = copy.imageOffset;
		blit.dstOffsets[1] =
		    vk::Offset3D {copy.imageOffset.x + static_cast<int32_t>(copy.imageExtent.width),
		                  copy.imageOffset.y + static_cast<int32_t>(copy.imageExtent.height), 1};
		command.blitImage(backing.image, vk::ImageLayout::eTransferSrcOptimal, transfer->image,
		                  vk::ImageLayout::eTransferDstOptimal, blit,
		                  BlitFilter(copy.imageSubresource.aspectMask));
	}
	TransferImageBarrier(command, *transfer, aspects, vk::ImageLayout::eTransferDstOptimal,
	                     vk::ImageLayout::eTransferSrcOptimal);
	command.copyImageToBuffer(transfer->image, vk::ImageLayout::eTransferSrcOptimal, buffer,
	                          static_cast<uint32_t>(copies.size()), copies.data());
	TransferBufferBarrier(command, buffer, offset, size, false, true);
	m_scheduler.DeferOperation(
	    [graphics = &m_graphics, transfer] { graphics->DeleteImage(*transfer); });
}

namespace ImageOps {

vk::Extent3D HostExtent(const GraphicContext& graphics, const ImageInfo& info) {
	const uint64_t cap_height = Config::GetInternalResolution();
	if (cap_height == 0 || info.pixel_format == vk::Format::eUndefined || info.IsVolume() ||
	    info.extent.depth != 1 || info.IsBlock() || info.resources.levels != 1) {
		return info.extent;
	}
	const uint64_t width     = info.extent.width;
	const uint64_t height    = info.extent.height;
	const uint64_t cap_width = (cap_height * 16 + 8) / 9;
	if (width <= cap_width && height <= cap_height) {
		return info.extent;
	}
	// Only screen-shaped images shrink: shadow maps and atlases keep their size.
	const auto wide = width * 9;
	const auto tall = height * 16;
	if ((wide > tall ? wide - tall : tall - wide) * 50 > wide) {
		return info.extent;
	}
	if (!ScalingSupported(graphics) || !CanBlit(graphics, info.pixel_format)) {
		return info.extent;
	}
	auto host_height = std::min(height, cap_height);
	auto host_width  = std::max<uint64_t>(1, (width * host_height + height / 2) / height);
	if (host_width > cap_width) {
		host_width  = cap_width;
		host_height = std::max<uint64_t>(1, (height * host_width + width / 2) / width);
	}
	return {static_cast<uint32_t>(host_width), static_cast<uint32_t>(host_height), 1};
}

void Validate(const ImageInfo& info) {
	ValidateOptionalRange(info.data, "data");
	ValidateOptionalRange(info.stencil, "stencil");

	if (info.pixel_format == vk::Format::eUndefined) {
		const bool metadata_empty =
		    info.metadata.range.Empty() && info.metadata.kind == ImageMetadataKind::None &&
		    info.metadata.compression == VideoOutCompression::Uncompressed &&
		    !info.metadata.stencil_compressed;
		if (info.data.Empty() || info.HasStencil() || !metadata_empty || info.extent.width == 0 ||
		    info.extent.height == 0 || info.extent.depth == 0 || info.resources.levels != 1 ||
		    info.resources.layers != 1 || info.samples != 1 || info.pitch != 0 ||
		    info.bytes_per_block != 0) {
			EXIT("invalid stencil association image\n");
		}
		return;
	}

	if (info.extent.width == 0 || info.extent.height == 0 || info.extent.depth == 0 ||
	    info.resources.levels == 0 || info.resources.levels > info.mip_layout.size() ||
	    info.resources.layers == 0 || info.samples == 0 ||
	    vulkan_sample_count(info.samples) == vk::SampleCountFlagBits {} ||
	    info.bytes_per_block == 0 || (info.data.address != 0 && info.pitch == 0)) {
		EXIT("invalid image geometry or format\n");
	}

	switch (info.type) {
		case Prospero::ImageType::kColor3D:
			if (info.resources.layers != 1) {
				EXIT("3D images cannot have array layers\n");
			}
			break;
		case Prospero::ImageType::kColor2D:
			if (info.extent.depth != 1) {
				EXIT("invalid 2D image shape\n");
			}
			break;
		default: EXIT("non-base image type: %u\n", static_cast<uint32_t>(info.type));
	}
	if (info.samples > 1 && info.resources.levels != 1) {
		EXIT("multisampled images cannot have mip levels\n");
	}

	if (info.metadata.stencil_compressed && !info.HasStencil()) {
		EXIT("compressed stencil metadata requires a stencil plane\n");
	}
	switch (info.metadata.kind) {
		case ImageMetadataKind::None:
			if (!info.metadata.range.Empty() ||
			    info.metadata.compression != VideoOutCompression::Uncompressed ||
			    info.metadata.stencil_compressed) {
				EXIT("metadata-free image has metadata state\n");
			}
			break;
		case ImageMetadataKind::Htile:
			if (!info.metadata.range.Valid() ||
			    info.metadata.compression != VideoOutCompression::Uncompressed) {
				EXIT("invalid HTILE metadata\n");
			}
			break;
		case ImageMetadataKind::Dcc:
		case ImageMetadataKind::Cmask:
			if (!GuestRange {info.metadata.range.address,
			                 std::max<uint64_t>(info.metadata.range.size, 1)}.Valid() ||
			    info.metadata.compression == VideoOutCompression::Unsupported) {
				EXIT("invalid color metadata\n");
			}
			break;
	}
}

Prospero::BufferFormat RenderTargetTransferFormat(uint32_t bytes_per_element) {
	switch (bytes_per_element) {
		case 1: return Prospero::BufferFormat::k8UNorm;
		case 2: return Prospero::BufferFormat::k16UNorm;
		case 4: return Prospero::BufferFormat::k32Float;
		case 8: return Prospero::BufferFormat::k16_16_16_16Float;
		case 16: return Prospero::BufferFormat::k32_32_32_32Float;
		default: EXIT("unsupported render-target element size: %u\n", bytes_per_element);
	}
}

} // namespace ImageOps

Image::Image(GraphicContext& graphics, CommandScheduler& scheduler, const ImageInfo& image_info)
    : info(image_info),
      stencil_subresources {0, image_info.resources.levels, 0, image_info.resources.layers},
      m_graphics(graphics), m_scheduler(scheduler) {
	KYTY_PROFILER_FUNCTION();
	ImageOps::Validate(info);
	m_cpu_dirty =
	    !info.data.Empty() && info.metadata.compression == VideoOutCompression::Uncompressed;
	if (info.pixel_format == vk::Format::eUndefined) {
		return;
	}

	vk::ImageCreateInfo create {};
	create.usage         = ImageUsageFlags(graphics, info);
	create.flags         = ImageCreateFlags(graphics, info, create.usage);
	create.imageType     = HostImageType(info.type);
	create.extent        = ImageOps::HostExtent(graphics, info);
	create.mipLevels     = info.resources.levels;
	create.arrayLayers   = info.IsVolume() ? 1u : info.resources.layers;
	create.format        = info.pixel_format;
	create.tiling        = vk::ImageTiling::eOptimal;
	create.initialLayout = vk::ImageLayout::eUndefined;
	create.samples       = vulkan_sample_count(info.samples);

	vk::ImageFormatProperties properties {};
	if (graphics.GetImageFormatProperties(create.format, create.imageType, create.tiling,
	                                      create.usage, create.flags,
	                                      &properties) != vk::Result::eSuccess ||
	    !static_cast<bool>(properties.sampleCounts & create.samples)) {
		EXIT("image format does not support required usage: format=%d type=%d usage=0x%x "
		     "flags=0x%x samples=%u\n",
		     static_cast<int>(create.format), static_cast<int>(create.imageType),
		     static_cast<vk::ImageUsageFlags::MaskType>(create.usage),
		     static_cast<vk::ImageCreateFlags::MaskType>(create.flags), info.samples);
	}

	if (!graphics.CreateImage(create, backing)) {
		EXIT("failed to create image: extent=%ux%ux%u format=%d layers=%u levels=%u\n",
		     create.extent.width, create.extent.height, create.extent.depth,
		     static_cast<int>(create.format), create.arrayLayers, create.mipLevels);
	}
	SetVulkanObjectNameF(
	    graphics.device, backing.image,
	    "Kyty.Image[guest=0x{:016x} size=0x{:x} extent={}x{}x{} format={} mips={} layers={} samples={}]",
	    info.data.address, info.data.size, info.extent.width, info.extent.height, info.extent.depth,
	    static_cast<uint32_t>(info.pixel_format), info.resources.levels, info.resources.layers,
	    info.samples);
}

uint64_t Image::HashGuestEdges() const {
	std::array<uint8_t, TRACKER_PAGE_SIZE * 2> bytes {};
	const auto                                 range = info.data;
	const uint64_t head_end =
	    std::min(range.End(), Common::AlignUp(range.address, TRACKER_PAGE_SIZE));
	const uint64_t tail_begin =
	    std::max(range.address, Common::AlignDown(range.End(), TRACKER_PAGE_SIZE));
	const uint64_t head_size    = head_end - range.address;
	const uint64_t tail_address = tail_begin < head_end ? head_end : tail_begin;
	const uint64_t tail_size    = range.End() - tail_address;
	if ((head_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(range.address, bytes.data(), head_size)) ||
	    (tail_size != 0 &&
	     !LibKernel::Memory::TryReadBacking(tail_address, bytes.data() + head_size, tail_size))) {
		EXIT("Image: failed to hash guest backing\n");
	}
	return XXH3_64bits(bytes.data(), static_cast<size_t>(head_size + tail_size));
}

Image::~Image() {
	KYTY_PROFILER_FUNCTION();
	for (const auto& cached: views) {
		m_graphics.device.destroyImageView(cached.view, nullptr);
	}
	if (backing.image != nullptr) {
		m_graphics.DeleteImage(backing);
	}
}

} // namespace Libs::Graphics
