#include <iostream>
#include <array>
#include <cstring>
#include <string>
#include <cpuid.h>

using namespace std;

void get_cpuid(unsigned int leaf, unsigned int subleaf, unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d) {
    __cpuid_count(leaf, subleaf, a, b, c, d);
}

string get_cpu_name() {
    unsigned int data[4];
    char name[49] = {0};
    for (int i = 0; i < 3; i++) {
        __cpuid(0x80000002 + i, data[0], data[1], data[2], data[3]);
        memcpy(name + i * 16, data, 16);
    }
    return string(name);
}

void get_core_info() {
    unsigned int a, b, c, d;

    __cpuid(1, a, b, c, d);
    int logical = (b >> 16) & 0xFF;
    bool ht_supported = d & (1 << 28);

    __cpuid_count(4, 0, a, b, c, d);
    int physical = ((a >> 26) & 0x3F) + 1;

    cout << "Physical cores: " << physical << endl;
    cout << "Logical processors: " << logical << endl;
    cout << "Hyperthreading supported: " << (ht_supported ? "Yes" : "No") << endl;
}

void get_clock_info() {
    unsigned int a, b, c, d;
    if (__get_cpuid_max(0x16, nullptr) >= 0x16) {
        __cpuid(0x16, a, b, c, d);
        cout << "Base Frequency: " << a << " MHz" << endl;
        cout << "Max Frequency: " << b << " MHz" << endl;
    } 
    else {
        cout << "Clock frequency info not supported on this CPU." << endl;
    }
}

void get_cache_info() {
    cout << "Cache Hierarchy:" << endl;

    unsigned int a, b, c, d;
    for (int i = 0; ; i++) {
        __cpuid_count(4, i, a, b, c, d);
        unsigned int cache_type = a & 0x1F;
        if (cache_type == 0) break;

        unsigned int cache_level = (a >> 5) & 0x7;
        unsigned int line_size = (b & 0xFFF) + 1;
        unsigned int partitions = ((b >> 12) & 0x3FF) + 1;
        unsigned int ways = ((b >> 22) & 0x3FF) + 1;
        unsigned int sets = c + 1;

        unsigned int cache_size = ways * partitions * line_size * sets;

        string type_str;
        switch (cache_type) {
            case 1: type_str = "Data Cache"; break;
            case 2: type_str = "Instruction Cache"; break;
            case 3: type_str = "Unified Cache"; break;
            default: type_str = "Unknown"; break;
        }

        cout << "L" << cache_level << " " << type_str
             << " - Size: " << cache_size / 1024 << " KB, "
             << "Line Size: " << line_size << " bytes, "
             << "Associativity: " << ways << "-way" << endl;
    }
}

int get_simd_info() {
	bool MMX   = false;
	bool SSE   = false;
	bool SSE2  = false;
	bool SSE3  = false;
	bool AES   = false;
	bool SSE41 = false;
	bool SSE42 = false;
	bool AVX   = false;
	bool AVX2  = false;
	bool SHA   = false;

    int a, b, c, d;
	__cpuid(0x00000001, a, b, c, d);

	MMX   = (d & ((int)1 << 23)) != 0;
	SSE   = (d & ((int)1 << 25)) != 0;
	SSE2  = (d & ((int)1 << 26)) != 0;
	SSE3  = (c & ((int)1 << 0))  != 0;
	AES   = (c & ((int)1 << 25)) != 0;
	SSE41 = (c & ((int)1 << 19)) != 0;
	SSE42 = (c & ((int)1 << 20)) != 0;

	AVX   = (c & ((int)1 << 28)) != 0;

	__cpuid(0x00000001, a, b, c, d);
	if (a >= 0x00000007){
		__cpuid(0x00000001, a, b, c, d);
		AVX2   = (b & ((int)1 <<  5)) != 0;
		SHA    = (b & ((int)1 << 29)) != 0;
	}

    cout << "SIMD Info:" << endl;
	printf("%s\n", MMX   ? "MMX   Supported" : "MMX   NOT Supported");
	printf("%s\n", SSE   ? "SSE   Supported" : "SSE   NOT Supported");
	printf("%s\n", SSE2  ? "SSE2  Supported" : "SSE2  NOT Supported");
	printf("%s\n", SSE3  ? "SSE3  Supported" : "SSE3  NOT Supported");
	printf("%s\n", SSE41 ? "SSE41 Supported" : "SSE41 NOT Supported");
	printf("%s\n", SSE42 ? "SSE42 Supported" : "SSE42 NOT Supported");
	printf("%s\n", AES   ? "AES   Supported" : "AES   NOT Supported");
	printf("%s\n", SHA   ? "SHA   Supported" : "SHA   NOT Supported");
	printf("%s\n", AVX   ? "AVX   Supported" : "AVX   NOT Supported");
	printf("%s\n", AVX2  ? "AVX2  Supported" : "AVX2  NOT Supported");

	return 0;
}


int main() {
    cout << "Processor Name: " << get_cpu_name() << "\n\n";
    get_core_info();
    cout << endl;
    get_clock_info();
    cout << endl;
    get_cache_info();
    cout << endl;
    get_simd_info();
    return 0;
}


