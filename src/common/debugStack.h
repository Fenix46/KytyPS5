#ifndef EMULATOR_SRC_COMMON_DEBUGSTACK_H_
#define EMULATOR_SRC_COMMON_DEBUGSTACK_H_

// Diagnostics only: caller addresses relative to the exe, to be resolved against the linker map.

#include <cstddef>
#include <cstdint>
#include <cstdio>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Common {

inline void DebugStackRvas(char* out, size_t out_size) {
	if (out_size == 0) {
		return;
	}
	out[0] = '\0';
#ifdef _WIN32
	void*      frames[20] {};
	const auto count = CaptureStackBackTrace(0, 20, frames, nullptr);
	const auto base  = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	size_t     used  = 0;
	for (USHORT i = 0; i < count && used + 12 < out_size; i++) {
		const auto addr = reinterpret_cast<uintptr_t>(frames[i]);
		if (addr >= base && addr < base + 0x4000000) {
			used += static_cast<size_t>(std::snprintf(out + used, out_size - used, " %llx",
			                                          static_cast<unsigned long long>(addr - base)));
		}
	}
#endif
}

// Raw scan of the thread stack for words that point into the exe, for frames the unwinder cannot
// see (leaf library code, JIT-less guest frames). Candidates only: not every hit is a return address.
inline void DebugStackScan(char* out, size_t out_size, uint64_t rsp) {
	if (out_size == 0) {
		return;
	}
	out[0] = '\0';
#ifdef _WIN32
	if (rsp == 0) {
		return;
	}
	const auto* tib   = reinterpret_cast<const NT_TIB*>(NtCurrentTeb());
	const auto  top   = reinterpret_cast<uintptr_t>(tib->StackBase);
	const auto  base  = reinterpret_cast<uintptr_t>(GetModuleHandleW(nullptr));
	size_t      used  = 0;
	int         found = 0;
	for (uintptr_t p = static_cast<uintptr_t>(rsp); p + 8 <= top && p < rsp + 0x600 &&
	                                                 used + 12 < out_size && found < 14;
	     p += 8) {
		const auto value = *reinterpret_cast<const uintptr_t*>(p);
		if (value >= base + 0x1000 && value < base + 0x2000000) {
			used += static_cast<size_t>(std::snprintf(out + used, out_size - used, " %llx",
			                                          static_cast<unsigned long long>(value - base)));
			found++;
		}
	}
#endif
}

// Module-relative form of an instruction address ("exe+rva" or "<other module>+rva").
inline void DebugDescribeAddress(char* out, size_t out_size, uint64_t address) {
	if (out_size == 0) {
		return;
	}
	out[0] = '\0';
#ifdef _WIN32
	HMODULE module = nullptr;
	if (address != 0 &&
	    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
	                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
	                       reinterpret_cast<LPCWSTR>(address), &module) != 0) {
		wchar_t path[260] {};
		GetModuleFileNameW(module, path, 260);
		const wchar_t* name = path;
		for (const wchar_t* c = path; *c != 0; c++) {
			if (*c == L'\\' || *c == L'/') {
				name = c + 1;
			}
		}
		char narrow[64] {};
		size_t i = 0;
		for (; name[i] != 0 && i + 1 < sizeof(narrow); i++) {
			narrow[i] = static_cast<char>(name[i] < 128 ? name[i] : '?');
		}
		std::snprintf(out, out_size, "%s+0x%llx", narrow,
		              static_cast<unsigned long long>(address - reinterpret_cast<uintptr_t>(module)));
	} else {
		std::snprintf(out, out_size, "0x%llx", static_cast<unsigned long long>(address));
	}
#endif
}

} // namespace Common

#endif // EMULATOR_SRC_COMMON_DEBUGSTACK_H_
