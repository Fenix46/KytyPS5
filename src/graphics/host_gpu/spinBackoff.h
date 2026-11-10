#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_SPINBACKOFF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_SPINBACKOFF_H_

#include <chrono>
#include <cstdint>
#include <thread>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace Libs::Graphics {

// The wait of a spin lock's contender: spins briefly, then yields, then sleeps 1 ms per round.
// The holders of the tracking locks can be inside a page-protection call or a whole collection;
// with many guest threads spinning in the fault handler at once the holder got no core, and the
// frame stalled until the scheduler boosted it. Short waits never leave the spin phase.
// Based on Erlou93/KytyPS5-Yotei cc05e4c2 (pageManager.cpp, SpinGuard), GPL-2.0.
class SpinBackoff final {
public:
	void Wait() noexcept {
		if (++m_spins < SPINS_PER_ROUND) {
#if defined(__x86_64__) || defined(_M_X64)
			_mm_pause();
#elif defined(_M_ARM64)
			__yield();
#elif defined(__aarch64__)
			asm volatile("yield");
#endif
			return;
		}
		m_spins = 0;
		if (m_rounds < YIELD_ROUNDS) {
			m_rounds++;
			std::this_thread::yield();
		} else {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
	}

private:
	// About 25 us of pauses per round; about 1 ms of yielding rounds before the first sleep.
	static constexpr uint32_t SPINS_PER_ROUND = 2048;
	static constexpr uint32_t YIELD_ROUNDS    = 40;

	uint32_t m_spins  = 0;
	uint32_t m_rounds = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_SPINBACKOFF_H_
