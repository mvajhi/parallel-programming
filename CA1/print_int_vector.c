#include <stdio.h>
#include <x86intrin.h>
#include <string.h>

typedef union {
    __m128i             int128;

    unsigned char       m128_u8[16];
    signed char         m128_i8[16];

    unsigned short      m128_u16[8];
    signed short        m128_i16[8];

    unsigned int        m128_u32[4];
    signed int          m128_i32[4];

    unsigned long long  m128_u64[2];
    signed long long    m128_i64[2];
} intVec;

void print_int_vector(__m128i a, const char *type)
{
    intVec tmp;
    tmp.int128 = a;

    printf("[");

    if (strcmp(type, "u8") == 0) {
        for (int i = 15; i > 0; i--)
            printf("0x%02X, ", tmp.m128_u8[i]);
        printf("0x%02X", tmp.m128_u8[0]);

    } 
    else if (strcmp(type, "i8") == 0) {
        for (int i = 15; i > 0; i--)
            printf("%4d, ", tmp.m128_i8[i]);
        printf("%4d", tmp.m128_i8[0]);

    } 
    else if (strcmp(type, "u16") == 0) {
        for (int i = 7; i > 0; i--)
            printf("0x%04X, ", tmp.m128_u16[i]);
        printf("0x%04X", tmp.m128_u16[0]);

    } 
    else if (strcmp(type, "i16") == 0) {
        for (int i = 7; i > 0; i--)
            printf("%6d, ", tmp.m128_i16[i]);
        printf("%6d", tmp.m128_i16[0]);

    } 
    else if (strcmp(type, "u32") == 0) {
        for (int i = 3; i > 0; i--)
            printf("0x%08X, ", tmp.m128_u32[i]);
        printf("0x%08X", tmp.m128_u32[0]);

    } 
    else if (strcmp(type, "i32") == 0) {
        for (int i = 3; i > 0; i--)
            printf("%11d, ", tmp.m128_i32[i]);
        printf("%11d", tmp.m128_i32[0]);

    } 
    else if (strcmp(type, "u64") == 0) {
        printf("0x%016llX, 0x%016llX", tmp.m128_u64[1], tmp.m128_u64[0]);

    } 
    else if (strcmp(type, "i64") == 0) {
        printf("%20lld, %20lld", tmp.m128_i64[1], tmp.m128_i64[0]);

    } 
    else {
        printf("Invalid type");
    }

    printf("]\n");
}

int main(void)
{
    unsigned char intArray[16] = {
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
        0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF
    };

    __m128i a = _mm_loadu_si128((const __m128i*)intArray);

    printf("u8:  ");  print_int_vector(a, "u8");
    printf("i8:  ");  print_int_vector(a, "i8");
    printf("u16: ");  print_int_vector(a, "u16");
    printf("i16: ");  print_int_vector(a, "i16");
    printf("u32: ");  print_int_vector(a, "u32");
    printf("i32: ");  print_int_vector(a, "i32");
    printf("u64: ");  print_int_vector(a, "u64");
    printf("i64: ");  print_int_vector(a, "i64");

    return 0;
}
