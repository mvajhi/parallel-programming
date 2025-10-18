#include <iostream>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <string>

using namespace std;

// --- Compiler Compatibility (MSVC vs GCC/Clang) ---

#if defined(_MSC_VER)
    // For Microsoft Visual Studio
    #include <intrin.h>
    inline void cpuid(int leaf, int subleaf, int* eax, int* ebx, int* ecx, int* edx) {
        int regs[4];
        __cpuidex(regs, leaf, subleaf);
        *eax = regs[0];
        *ebx = regs[1];
        *ecx = regs[2];
        *edx = regs[3];
    }
#else
    // For GCC or Clang (Linux, macOS, MinGW)
    #include <cpuid.h>
    inline void cpuid(int leaf, int subleaf, int* eax, int* ebx, int* ecx, int* edx) {
        // __cpuid_count is defined in <cpuid.h>
        __cpuid_count(leaf, subleaf, *eax, *ebx, *ecx, *edx);
    }
#endif

// --------------------------------------------------------

// Helper function to print a string from registers
void print_string(int eax, int ebx, int ecx, int edx) {
    char str[17];
    str[16] = '\0';
    memcpy(str + 0, &eax, 4);
    memcpy(str + 4, &ebx, 4);
    memcpy(str + 8, &ecx, 4);
    memcpy(str + 12, &edx, 4);
    cout << str;
}

int main() {
    int eax, ebx, ecx, edx;
    int max_leaf, max_ext_leaf;

    // Get highest supported basic leaf
    cpuid(0x0, 0, &max_leaf, &ebx, &ecx, &edx);
    
    // Get highest supported extended leaf
    cpuid(0x80000000, 0, &max_ext_leaf, &ebx, &ecx, &edx);

    cout << "===============================================\n";
    cout << "        CPU Information Report (CPUID)\n";
    cout << "===============================================\n\n";

    // --- (A) Processor Name ---
    cout << "--- (A) Processor Name ---\n";
    if (max_ext_leaf >= 0x80000004) {
        cout << "Full Name: ";
        cpuid(0x80000002, 0, &eax, &ebx, &ecx, &edx);
        print_string(eax, ebx, ecx, edx);
        cpuid(0x80000003, 0, &eax, &ebx, &ecx, &edx);
        print_string(eax, ebx, ecx, edx);
        cpuid(0x80000004, 0, &eax, &ebx, &ecx, &edx);
        print_string(eax, ebx, ecx, edx);
        cout << "\n";
    } else {
        cout << "Fetching processor name string not supported.\n";
    }

    // --- (C) Operating Frequency ---
    cout << "\n--- (C) Operating Frequency ---\n";
    if (max_leaf >= 0x16) {
        cpuid(0x16, 0, &eax, &ebx, &ecx, &edx);
        if (eax > 0) cout << "Base Frequency:      " << eax << " MHz\n";
        if (ebx > 0) cout << "Max Turbo Frequency: " << ebx << " MHz\n";
        if (ecx > 0) cout << "Bus Frequency:       " << ecx << " MHz\n";
    } else {
        cout << "Leaf 0x16 (Frequency) not supported.\n";
        cout << "Note: Check manufacturer (e.g., Intel Ark) for full specs.\n";
    }

    // --- (B) Core Count and Hyperthreading ---
    cout << "\n--- (B) Core Count and Hyperthreading ---\n";
    int logical_cores = 0;
    int physical_cores = 0;
    int htt_supported = 0;

    if (max_leaf >= 1) {
        cpuid(0x1, 0, &eax, &ebx, &ecx, &edx);
        // Logical core count (Threads)
        logical_cores = (ebx >> 16) & 0xFF;
        // Check Hyperthreading support bit
        htt_supported = (edx >> 28) & 1;
    }

    // Modern method (Leaf 0xB) for core count
    if (max_leaf >= 0xB) {
        int level_type;
        cpuid(0xB, 1, &eax, &ebx, &ecx, &edx); // Core level
        level_type = (ecx >> 8) & 0xFF;
        if (level_type == 2) { // 2 = Core level
            logical_cores = ebx & 0xFFFF;
        }

        cpuid(0xB, 0, &eax, &ebx, &ecx, &edx); // Thread level
        level_type = (ecx >> 8) & 0xFF;
        if (level_type == 1) { // 1 = SMT (Thread) level
            int threads_per_core = ebx & 0xFFFF;
            if (threads_per_core > 0) {
                physical_cores = logical_cores / threads_per_core;
            }
        }
    } 
    
    // Fallback (legacy) method if modern one failed
    if (physical_cores == 0 && max_leaf >= 4) {
         cpuid(0x4, 0, &eax, &ebx, &ecx, &edx);
         physical_cores = ((eax >> 26) & 0x3F) + 1;
    }

    // Final fallback for very old CPUs
    if (physical_cores == 0) physical_cores = logical_cores;

    cout << "Physical Cores: " << physical_cores << "\n";
    cout << "Logical Threads: " << logical_cores << "\n";

    cout << "Hyperthreading Status:\n";
    if (htt_supported == 0) {
        cout << " -> Not supported (or AMD SMT).\n";
    } else if (logical_cores > physical_cores) {
        cout << " -> Enabled.\n";
    } else {
        cout << " -> Supported but Disabled in BIOS.\n";
    }
    cout << "Note: To enable/disable Hyperthreading, enter BIOS/UEFI setup.\n";

    // --- (D) Cache Sizes ---
    cout << "\n--- (D) Cache Sizes ---\n";
    if (max_leaf >= 4) {
        int cache_index = 0;
        while (true) {
            cpuid(0x4, cache_index, &eax, &ebx, &ecx, &edx);
            int cache_type = eax & 0x1F;
            
            if (cache_type == 0) // Type 0 means end of cache list
                break;

            int cache_level = (eax >> 5) & 0x7;
            const char* type_str;
            if (cache_type == 1) type_str = "Data Cache";
            else if (cache_type == 2) type_str = "Instruction Cache";
            else if (cache_type == 3) type_str = "Unified Cache";
            else type_str = "Unknown Cache";

            int ways = ((ebx >> 22) & 0x3FF) + 1;
            int partitions = ((ebx >> 12) & 0x3FF) + 1;
            int line_size = (ebx & 0xFFF) + 1;
            int sets = ecx + 1;

            long cache_size_kb = static_cast<long>(ways * partitions * line_size * sets) / 1024;

            cout << "L" << cache_level << " " << type_str << ": " 
                 << cache_size_kb << " KB\n";
            cache_index++;
        }
    } else {
        cout << "Modern cache info (Leaf 0x4) not supported.\n";
    }

    // --- (E) Supported SIMD Architectures ---
    cout << "\n--- (E) Supported SIMD Architectures ---\n";
    if (max_leaf >= 1) {
        cpuid(0x1, 0, &eax, &ebx, &ecx, &edx);
        cout << "EDX Flags:\n";
        if (edx & (1 << 23)) cout << "  [+] MMX\n";
        if (edx & (1 << 25)) cout << "  [+] SSE\n";
        if (edx & (1 << 26)) cout << "  [+] SSE2\n";
        
        cout << "ECX Flags:\n";
        if (ecx & (1 << 0))  cout << "  [+] SSE3\n";
        if (ecx & (1 << 9))  cout << "  [+] SSSE3\n";
        if (ecx & (1 << 19)) cout << "  [+] SSE4.1\n";
        if (ecx & (1 << 20)) cout << "  [+] SSE4.2\n";
        if (ecx & (1 << 28)) cout << "  [+] AVX\n";
    }

    if (max_leaf >= 7) {
        cpuid(0x7, 0, &eax, &ebx, &ecx, &edx);
        cout << "Leaf 7 EBX Flags:\n";
        if (ebx & (1 << 5))  cout << "  [+] AVX2\n";
        if (ebx & (1 << 16)) cout << "  [+] AVX-512 F\n";
        if (ebx & (1 << 17)) cout << "  [+] AVX-512 DQ\n";
        if (ebx & (1 << 21)) cout << "  [+] AVX-512 IFMA\n";
        if (ebx & (1 << 28)) cout << "  [+] AVX-512 CD\n";
        if (ebx & (1 << 30)) cout << "  [+] AVX-512 BW\n";
        if (ebx & (1 << 31)) cout << "  [+] AVX-512 VL\n";
    }
    cout << "\n===============================================\n";

    return 0;
}