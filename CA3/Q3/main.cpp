#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include "stdio.h"
#include "ipp.h"
#include "x86intrin.h"
#include <omp.h>

#define NUM_THREADS 6

class Blur
{
protected:
    const int16_t kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}};

    const int normalizationFactor = 16;

public:
    Blur() {}
    ~Blur() {}

    virtual cv::Mat apply(const cv::Mat &in_img) = 0;
};

class BlurSerial : public Blur
{
protected:
    unsigned char apply_for_one_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col)
    {
        int pixel = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                pixel += kernel[i][j] * in_image[(row + i - 1) * NCOLS + col + j - 1];
        return cv::saturate_cast<unsigned char>(pixel / normalizationFactor);
    }

public:
    cv::Mat apply(const cv::Mat &in_img)
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img = cv::Mat::zeros(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col++)
                out_image[row * NCOLS + col] = apply_for_one_pixel(in_image, row, NCOLS, col);
        return out_img;
    }
};

class BlurOMP : public BlurSerial
{
public:
    cv::Mat apply(const cv::Mat &in_img)
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

#pragma omp parallel for
        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col++)
                out_image[row * NCOLS + col] = apply_for_one_pixel(in_image, row, NCOLS, col);
        return out_img;
    }
};

class BlurSIMD : public Blur
{
protected:
    void apply_for_32_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col, unsigned char *out_image)
    {
        __m256i p_8, p_16_low, p_16_high, k_value;
        __m256i multi_p_16_low, multi_p_16_high;
        __m256i acc_p_16_low, acc_p_16_high;
        __m256i packed, reordered;
        __m256i max255 = _mm256_set1_epi16(255);
        __m256i zero = _mm256_setzero_si256();

        acc_p_16_low = zero;
        acc_p_16_high = zero;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                p_8 = _mm256_loadu_si256((__m256i const *)&in_image[(row + i - 1) * NCOLS + col + j - 1]);
                p_16_low = _mm256_unpacklo_epi8(p_8, zero);
                p_16_high = _mm256_unpackhi_epi8(p_8, zero);

                k_value = _mm256_set1_epi16((short int)kernel[i][j]);

                multi_p_16_low = _mm256_mullo_epi16(k_value, p_16_low);
                acc_p_16_low = _mm256_adds_epi16(multi_p_16_low, acc_p_16_low);

                multi_p_16_high = _mm256_mullo_epi16(k_value, p_16_high);
                acc_p_16_high = _mm256_adds_epi16(multi_p_16_high, acc_p_16_high);
            }

        // div 16
        acc_p_16_low = _mm256_srli_epi16(acc_p_16_low, 4);
        acc_p_16_high = _mm256_srli_epi16(acc_p_16_high, 4);

        // fix range [255,0]
        acc_p_16_low = _mm256_min_epu16(acc_p_16_low, max255);
        acc_p_16_high = _mm256_min_epu16(acc_p_16_high, max255);

        // pack 16-bit -> 8-bit (within each 128-bit lane: [high0..7|low0..7], [high8..15|low8..15])
        packed = _mm256_packus_epi16(acc_p_16_high, acc_p_16_low);
        /* _mm256_permute4x64_epi64
        DEFINE SELECT4(src, control) {
            CASE(control[1:0]) OF
            0:	tmp[63:0] := src[63:0]
            1:	tmp[63:0] := src[127:64]
            2:	tmp[63:0] := src[191:128]
            3:	tmp[63:0] := src[255:192]
            ESAC
            RETURN tmp[63:0]
        }
        dst[63:0] := SELECT4(a[255:0], imm8[1:0])
        dst[127:64] := SELECT4(a[255:0], imm8[3:2])
        dst[191:128] := SELECT4(a[255:0], imm8[5:4])
        dst[255:192] := SELECT4(a[255:0], imm8[7:6])
        dst[MAX:256] := 0
        */
        reordered = _mm256_permute4x64_epi64(packed, (2 << 6) | (3 << 4) | (0 << 2) | 1);
        _mm256_storeu_si256((__m256i_u *)&out_image[row * NCOLS + col], reordered);
    }

public:
    cv::Mat apply(const cv::Mat &in_img)
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col += 32)
            {
                apply_for_32_pixel(in_image, row, NCOLS, col, out_image);
                if (col == 1)
                    out_image[(row * NCOLS) + 0] = 0;
                else if (!(col + 32 < NCOLS - 1))
                    out_image[(row * NCOLS) + (NCOLS - 1)] = 0;
            }

        out_image[((NROWS - 1) * (NCOLS)) + 0] = 0;
        return out_img;
    }
};

class BlurSIMD_OMP : public BlurSIMD
{
public:
    cv::Mat apply(const cv::Mat &in_img)
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

#pragma omp parallel for
        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col += 32)
            {
                apply_for_32_pixel(in_image, row, NCOLS, col, out_image);
                if (col == 1)
                    out_image[(row * NCOLS) + 0] = 0;
                else if (!(col + 32 < NCOLS - 1))
                {
                    out_image[(row * NCOLS) + (NCOLS - 1)] = 0;
                    out_image[((row + 1) * NCOLS) + 0] = 0;
                }
            }

        out_image[((NROWS - 1) * (NCOLS)) + 0] = 0;
        return out_img;
    }
};

class Sobel
{
protected:
    const int kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};

    const int kernel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

public:
    Sobel() {}
    ~Sobel() {}

    virtual cv::Mat apply(const cv::Mat &in_img) = 0;
};

class SobelSerial : public Sobel
{
protected:
    unsigned char apply_for_one_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col)
    {
        int gx = 0;
        int gy = 0;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                int pixel_val = in_image[(row + i - 1) * NCOLS + col + j - 1];
                if (kernel_x[i][j] != 0)
                    gx += kernel_x[i][j] * pixel_val;
                if (kernel_y[i][j] != 0)
                    gy += kernel_y[i][j] * pixel_val;
            }

        double magnitude = std::sqrt(static_cast<double>(gx * gx) + static_cast<double>(gy * gy));

        return cv::saturate_cast<unsigned char>(magnitude);
    }

public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col++)
                out_image[row * NCOLS + col] = apply_for_one_pixel(in_image, row, NCOLS, col);
        return out_img;
    }
};

class SobelOMP : public SobelSerial
{
public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

#pragma omp parallel for
        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col++)
                out_image[row * NCOLS + col] = apply_for_one_pixel(in_image, row, NCOLS, col);
        return out_img;
    }
};

class SobelSIMD : public Sobel
{
protected:
    inline void accumulate_kernel_product(int kernel_value,
                                          const __m256i &p_16_low, const __m256i &p_16_high,
                                          __m256i &acc_low, __m256i &acc_high)
    {
        // Only perform calculations if the kernel weight is not zero
        if (kernel_value != 0)
        {
            // Set a 256-bit register with 16 copies of the 16-bit kernel value
            __m256i k_value = _mm256_set1_epi16((short int)kernel_value);

            // Multiply kernel value by low pixel data and accumulate
            __m256i multi_low = _mm256_mullo_epi16(k_value, p_16_low);
            acc_low = _mm256_add_epi16(multi_low, acc_low);

            // Multiply kernel value by high pixel data and accumulate
            __m256i multi_high = _mm256_mullo_epi16(k_value, p_16_high);
            acc_high = _mm256_add_epi16(multi_high, acc_high);
        }
    }

    inline void apply_sobel_kernels(const unsigned char *in_image, int row, unsigned int NCOLS, int col,
                                    __m256i &gx_16_low, __m256i &gx_16_high,
                                    __m256i &gy_16_low, __m256i &gy_16_high)
    {
        __m256i p_8, p_16_low, p_16_high;
        __m256i zero = _mm256_setzero_si256();

        gx_16_low = zero;
        gx_16_high = zero;
        gy_16_low = zero;
        gy_16_high = zero;

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                // Load 32 pixels (8-bit)
                p_8 = _mm256_loadu_si256((__m256i const *)&in_image[(row + i - 1) * NCOLS + col + j - 1]);

                // Unpack 8-bit to 16-bit (with zero extension)
                p_16_low = _mm256_unpacklo_epi8(p_8, zero);
                p_16_high = _mm256_unpackhi_epi8(p_8, zero);

                // Apply Gx kernel
                accumulate_kernel_product(kernel_x[i][j],
                                          p_16_low, p_16_high,
                                          gx_16_low, gx_16_high);

                // Apply Gy kernel
                accumulate_kernel_product(kernel_y[i][j],
                                          p_16_low, p_16_high,
                                          gy_16_low, gy_16_high);
            }
        }
    }

    inline void calculate_magnitude(const __m256i &gx_16_low, const __m256i &gx_16_high,
                                    const __m256i &gy_16_low, const __m256i &gy_16_high,
                                    __m256i &result_a, __m256i &result_b,
                                    __m256i &result_c, __m256i &result_d)
    {
        __m256i gx_32_a, gx_32_b, gx_32_c, gx_32_d;
        __m256i gy_32_a, gy_32_b, gy_32_c, gy_32_d;
        __m256i gx_squared_a, gx_squared_b, gx_squared_c, gx_squared_d;
        __m256i gy_squared_a, gy_squared_b, gy_squared_c, gy_squared_d;
        __m256i sum_a, sum_b, sum_c, sum_d;
        __m256 sqrt_a, sqrt_b, sqrt_c, sqrt_d;

        convert_gradients_16bit_to_32(gx_32_a, gx_16_low, gx_32_b, gx_32_c, gx_16_high, gx_32_d,
                                      gy_32_a, gy_16_low, gy_32_b, gy_32_c, gy_16_high, gy_32_d);

        // gx^2 , gy^2
        square_gradients(gx_squared_a, gx_32_a, gx_squared_b, gx_32_b, gx_squared_c, gx_32_c, gx_squared_d, gx_32_d,
                         gy_squared_a, gy_32_a, gy_squared_b, gy_32_b, gy_squared_c, gy_32_c, gy_squared_d, gy_32_d);

        // Sum gx^2 + gy^2
        sum_gx_gy(sum_a, gx_squared_a, gy_squared_a, sum_b, gx_squared_b, gy_squared_b, sum_c,
                  gx_squared_c, gy_squared_c, sum_d, gx_squared_d, gy_squared_d);

        // Convert to float, compute sqrt, convert back to int
        sqrt_g_sum(sqrt_a, sum_a, sqrt_b, sum_b, sqrt_c, sum_c, sqrt_d, sum_d);

        convert_float_to_int(result_a, sqrt_a, result_b, sqrt_b, result_c, sqrt_c, result_d, sqrt_d);
    }

    inline void convert_float_to_int(__m256i &result_a, const __m256 &sqrt_a, __m256i &result_b, const __m256 &sqrt_b,
                                     __m256i &result_c, const __m256 &sqrt_c, __m256i &result_d, const __m256 &sqrt_d)
    {
        result_a = _mm256_cvtps_epi32(sqrt_a);
        result_b = _mm256_cvtps_epi32(sqrt_b);
        result_c = _mm256_cvtps_epi32(sqrt_c);
        result_d = _mm256_cvtps_epi32(sqrt_d);
    }

    inline void sqrt_g_sum(__m256 &sqrt_a, const __m256i &sum_a, __m256 &sqrt_b, const __m256i &sum_b,
                           __m256 &sqrt_c, const __m256i &sum_c, __m256 &sqrt_d, const __m256i &sum_d)
    {
        sqrt_a = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_a));
        sqrt_b = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_b));
        sqrt_c = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_c));
        sqrt_d = _mm256_sqrt_ps(_mm256_cvtepi32_ps(sum_d));
    }

    inline void sum_gx_gy(__m256i &sum_a, const __m256i &gx_squared_a, const __m256i &gy_squared_a, __m256i &sum_b,
                          const __m256i &gx_squared_b, const __m256i &gy_squared_b, __m256i &sum_c, const __m256i &gx_squared_c,
                          const __m256i &gy_squared_c, __m256i &sum_d, const __m256i &gx_squared_d, const __m256i &gy_squared_d)
    {
        sum_a = _mm256_add_epi32(gx_squared_a, gy_squared_a);
        sum_b = _mm256_add_epi32(gx_squared_b, gy_squared_b);
        sum_c = _mm256_add_epi32(gx_squared_c, gy_squared_c);
        sum_d = _mm256_add_epi32(gx_squared_d, gy_squared_d);
    }

    inline void square_gradients(__m256i &gx_squared_a, const __m256i &gx_32_a, __m256i &gx_squared_b,
                                 const __m256i &gx_32_b, __m256i &gx_squared_c, const __m256i &gx_32_c, __m256i &gx_squared_d,
                                 const __m256i &gx_32_d, __m256i &gy_squared_a, const __m256i &gy_32_a, __m256i &gy_squared_b,
                                 const __m256i &gy_32_b, __m256i &gy_squared_c, const __m256i &gy_32_c, __m256i &gy_squared_d, const __m256i &gy_32_d)
    {
        gx_squared_a = _mm256_mullo_epi32(gx_32_a, gx_32_a);
        gx_squared_b = _mm256_mullo_epi32(gx_32_b, gx_32_b);
        gx_squared_c = _mm256_mullo_epi32(gx_32_c, gx_32_c);
        gx_squared_d = _mm256_mullo_epi32(gx_32_d, gx_32_d);

        gy_squared_a = _mm256_mullo_epi32(gy_32_a, gy_32_a);
        gy_squared_b = _mm256_mullo_epi32(gy_32_b, gy_32_b);
        gy_squared_c = _mm256_mullo_epi32(gy_32_c, gy_32_c);
        gy_squared_d = _mm256_mullo_epi32(gy_32_d, gy_32_d);
    }

    inline void convert_gradients_16bit_to_32(__m256i &gx_32_a, const __m256i &gx_16_low, __m256i &gx_32_b, __m256i &gx_32_c,
                                              const __m256i &gx_16_high, __m256i &gx_32_d, __m256i &gy_32_a, const __m256i &gy_16_low,
                                              __m256i &gy_32_b, __m256i &gy_32_c, const __m256i &gy_16_high, __m256i &gy_32_d)
    {
        // Convert 16-bit gradients to 32-bit for squaring and sqrt
        // Extract 128-bit lanes and convert
        gx_32_a = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_low, 0));
        gx_32_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_low, 1));
        gx_32_c = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_high, 0));
        gx_32_d = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gx_16_high, 1));

        gy_32_a = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_low, 0));
        gy_32_b = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_low, 1));
        gy_32_c = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_high, 0));
        gy_32_d = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(gy_16_high, 1));
    }

    inline void pack_and_store_results(__m256i &result_a, __m256i &result_b,
                                       __m256i &result_c, __m256i &result_d,
                                       unsigned char *out_image, int row, unsigned int NCOLS, int col)
    {
        clamp_results(result_a, result_b, result_c, result_d);

        __m256i reordered;
        reduce_and_reorder_results(result_a, result_b, result_c, result_d, reordered);

        _mm256_storeu_si256((__m256i_u *)&out_image[row * NCOLS + col], reordered);
    }

    inline void reduce_and_reorder_results(__m256i &result_a, __m256i &result_b, __m256i &result_c, __m256i &result_d, __m256i &reordered)
    {
        __m256i packed_ab, packed_cd, final_packed;

        // Pack 32-bit -> 16-bit [b0-3|a0-3] [b4-7|a4-7]
        packed_ab = _mm256_packus_epi32(result_a, result_b);
        packed_cd = _mm256_packus_epi32(result_c, result_d);

        // Permute to fix lane crossing: [a0-7, b0-7, a8-15, b8-15] -> [a0-15, b0-15]
        packed_ab = _mm256_permute4x64_epi64(packed_ab, (3 << 6) | (1 << 4) | (2 << 2) | 0);
        packed_cd = _mm256_permute4x64_epi64(packed_cd, (3 << 6) | (1 << 4) | (2 << 2) | 0);

        // Pack 16-bit -> 8-bit
        // This pack correctly orders the pixels: [d | c | b | a]
        final_packed = _mm256_packus_epi16(packed_ab, packed_cd);

        // Permute to final storage order: [d | c | b | a] -> [a | b | c | d]
        reordered = _mm256_permute4x64_epi64(final_packed, (3 << 6) | (2 << 4) | (1 << 2) | 0);
    }

    inline void clamp_results(__m256i &result_a, __m256i &result_b, __m256i &result_c, __m256i &result_d)
    {
        __m256i zero = _mm256_setzero_si256();
        __m256i max255 = _mm256_set1_epi32(255);

        result_a = _mm256_min_epi32(result_a, max255);
        result_a = _mm256_max_epi32(result_a, zero);

        result_b = _mm256_min_epi32(result_b, max255);
        result_b = _mm256_max_epi32(result_b, zero);

        result_c = _mm256_min_epi32(result_c, max255);
        result_c = _mm256_max_epi32(result_c, zero);

        result_d = _mm256_min_epi32(result_d, max255);
        result_d = _mm256_max_epi32(result_d, zero);
    }

    inline void apply_for_32_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col, unsigned char *out_image)
    {
        // Step 1: Apply Sobel kernels to get 16-bit gradients
        __m256i gx_16_low, gx_16_high, gy_16_low, gy_16_high;
        apply_sobel_kernels(in_image, row, NCOLS, col,
                            gx_16_low, gx_16_high, gy_16_low, gy_16_high);

        // Step 2: Calculate magnitude (sqrt(gx^2 + gy^2)) as 32-bit integers
        __m256i result_a, result_b, result_c, result_d;
        calculate_magnitude(gx_16_low, gx_16_high, gy_16_low, gy_16_high,
                            result_a, result_b, result_c, result_d);

        // Step 3: Clamp to [0, 255], pack to 8-bit, reorder, and store
        pack_and_store_results(result_a, result_b, result_c, result_d,
                               out_image, row, NCOLS, col);
    }

public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col += 32)
            {
                apply_for_32_pixel(in_image, row, NCOLS, col, out_image);

                if (col == 1)
                    out_image[(row * NCOLS) + 0] = 0;
                else if (!(col + 32 < NCOLS - 1))
                    out_image[(row * NCOLS) + (NCOLS - 1)] = 0;
            }

        out_image[((NROWS - 1) * (NCOLS)) + 0] = 0;

        return out_img;
    }
};

class SobelSIMD_OMP : public SobelSIMD
{
public:
    cv::Mat apply(const cv::Mat &in_img) override
    {
        unsigned int NCOLS = in_img.cols;
        unsigned int NROWS = in_img.rows;

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

#pragma omp parallel for
        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col += 32)
            {
                apply_for_32_pixel(in_image, row, NCOLS, col, out_image);

                if (col == 1)
                    out_image[(row * NCOLS) + 0] = 0;
                else if (!(col + 32 < NCOLS - 1))
                {
                    out_image[(row * NCOLS) + (NCOLS - 1)] = 0;
                    out_image[((row + 1) * NCOLS) + 0] = 0;
                }
            }

        out_image[((NROWS - 1) * (NCOLS)) + 0] = 0;

        return out_img;
    }
};

void printSectionHeader(const char *title)
{
    printf("\n\n");
    printf("**********************************************************\n");
    printf(" %s\n", title);
    printf("**********************************************************\n");
}

void validateImage(const char *algorithm_name, const cv::Mat &img_serial, const cv::Mat &img_target)
{
    unsigned int NCOLS = img_serial.cols;
    unsigned int NROWS = img_serial.rows;
    long long diff_count = 0;

    for (int row = 0; row < NROWS; row++)
    {
        const unsigned char *p_serial = img_serial.ptr<unsigned char>(row);
        const unsigned char *p_target = img_target.ptr<unsigned char>(row);

        for (int col = 0; col < NCOLS; col++)
            if (p_serial[col] != p_target[col])
                diff_count++;
    }

    long long total_processed_pixels = (long long)(NROWS - 2) * (NCOLS - 2);
    double diff_percentage = 0.0;

    if (total_processed_pixels > 0)
        diff_percentage = (double)diff_count / total_processed_pixels * 100.0;

    const char *status = (diff_count == 0) ? "[ PASS ]" : "[ FAIL ]";

    // Print formatted row
    // Format: | Algorithm Name | Diff Pixels | Error % | Status |
    printf(" %-22s | %12lld | %8.4f%% | %s\n",
           algorithm_name, diff_count, diff_percentage, status);
}

void printPerformanceReport(const char *title, double time_serial, double time_simd, double time_omp, double time_simd_omp)
{
    printf("\n");
    printf("==========================================================\n");
    printf(" METRIC: %s \n", title);
    printf("==========================================================\n");

    printf(" [Raw Timings]\n");
    printf("  Serial            : %9.6f s\n", time_serial);
    printf("  SIMD              : %9.6f s\n", time_simd);
    printf("  OpenMP            : %9.6f s\n", time_omp);
    printf("  SIMD + OpenMP     : %9.6f s\n", time_simd_omp);

    printf("\n [Speedup vs. Serial] (Baseline)\n");
    printf("  Serial / SIMD       : %6.2fx\n", time_serial / time_simd);
    printf("  Serial / OpenMP     : %6.2fx\n", time_serial / time_omp);
    printf("  Serial / SIMD+OMP   : %6.2fx  <-- Total Speedup\n", time_serial / time_simd_omp);

    printf("\n [Relative Improvements]\n");
    printf("  SIMD   / SIMD+OMP   : %6.2fx  (Scalability Gain)\n", time_simd / time_simd_omp);
    printf("  OpenMP / SIMD+OMP   : %6.2fx  (Vectorization Gain)\n", time_omp / time_simd_omp);
    printf("  SIMD   / OpenMP     : %6.2fx\n", time_simd / time_omp);

    printf("==========================================================\n");
}

// https://gemini.google.com/share/c28dddc233ef
void runFullDiagnostics(
    const cv::Mat &b_img_serial, const cv::Mat &b_img_simd,
    const cv::Mat &b_img_OMP, const cv::Mat &b_img_simd_OMP,
    const cv::Mat &out_img_serial, const cv::Mat &out_img_simd,
    const cv::Mat &out_img_OMP, const cv::Mat &out_img_simd_OMP,
    double time1_serial, double time1_SIMD, double time1_OMP, double time1_SIMD_OMP,
    double time2_serial, double time2_SIMD, double time2_OMP, double time2_SIMD_OMP)
{
    // --- 1. VALIDATION DASHBOARD ---
    printSectionHeader("VALIDATION DASHBOARD (Reference: Serial)");
    printf(" %-22s | %12s | %9s | %s\n", "Comparison", "Diff Pixels", "Error %", "Status");
    printf("---------------------------------------------------------------------\n");

    // Blur Checks
    validateImage("Blur (Serial vs OMP)", b_img_serial, b_img_OMP);
    validateImage("Blur (Serial vs SIMD)", b_img_serial, b_img_simd);
    validateImage("Blur (Hybrid)", b_img_serial, b_img_simd_OMP);
    printf("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n");

    // Sobel Checks
    validateImage("Sobel (Serial vs OMP)", out_img_serial, out_img_OMP);
    validateImage("Sobel (Serial vs SIMD)", out_img_serial, out_img_simd);
    validateImage("Sobel (Hybrid)", out_img_serial, out_img_simd_OMP);
    printf("---------------------------------------------------------------------\n");

    // --- 2. PERFORMANCE DASHBOARD ---
    printSectionHeader("PERFORMANCE ANALYSIS");

    printPerformanceReport("Blur Kernel",
                           time1_serial, time1_SIMD, time1_OMP, time1_SIMD_OMP);

    printPerformanceReport("Sobel Kernel",
                           time2_serial, time2_SIMD, time2_OMP, time2_SIMD_OMP);

    printPerformanceReport("Total Pipeline (Blur + Sobel)",
                           time1_serial + time2_serial,
                           time1_SIMD + time2_SIMD,
                           time1_OMP + time2_OMP,
                           time1_SIMD_OMP + time2_SIMD_OMP);

    printf("\n[Diagnostics Completed]\n");
}

int main()
{
    omp_set_num_threads(NUM_THREADS);

    double time1_serial, time2_serial;
    double time1_OMP, time2_OMP;
    double time1_SIMD, time2_SIMD;
    double time1_SIMD_OMP, time2_SIMD_OMP;
    double start, end;
    cv::Mat in_img;
    in_img = cv::imread("base.jpg", cv::IMREAD_GRAYSCALE);

    start = omp_get_wtime();
    BlurSerial blur_operator_serial;
    cv::Mat b_img_serial = blur_operator_serial.apply(in_img);
    end = omp_get_wtime();
    time1_serial = end - start;

    start = omp_get_wtime();
    BlurOMP blur_operator_OMP;
    cv::Mat b_img_OMP = blur_operator_OMP.apply(in_img);
    end = omp_get_wtime();
    time1_OMP = end - start;

    start = omp_get_wtime();
    BlurSIMD blur_operator_SIMD;
    cv::Mat b_img_simd = blur_operator_SIMD.apply(in_img);
    end = omp_get_wtime();
    time1_SIMD = end - start;

    start = omp_get_wtime();
    BlurSIMD_OMP blur_operator_SIMD_OMP;
    cv::Mat b_img_simd_OMP = blur_operator_SIMD_OMP.apply(in_img);
    end = omp_get_wtime();
    time1_SIMD_OMP = end - start;

    start = omp_get_wtime();
    SobelSerial sobel_operator_serial;
    cv::Mat out_img_serial = sobel_operator_serial.apply(b_img_serial);
    end = omp_get_wtime();
    time2_serial = end - start;

    start = omp_get_wtime();
    SobelOMP sobel_operator_OMP;
    cv::Mat out_img_OMP = sobel_operator_OMP.apply(b_img_OMP);
    end = omp_get_wtime();
    time2_OMP = end - start;

    start = omp_get_wtime();
    SobelSIMD sobel_operator_simd;
    cv::Mat out_img_simd = sobel_operator_simd.apply(b_img_simd);
    end = omp_get_wtime();
    time2_SIMD = end - start;

    start = omp_get_wtime();
    SobelSIMD_OMP sobel_operator_simd_OMP;
    cv::Mat out_img_simd_OMP = sobel_operator_simd_OMP.apply(b_img_simd_OMP);
    end = omp_get_wtime();
    time2_SIMD_OMP = end - start;

    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::imshow("input image", in_img);

    cv::namedWindow("blur(serial)", cv::WINDOW_AUTOSIZE);
    cv::imshow("blur(serial)", b_img_serial);

    cv::namedWindow("blur(OMP)", cv::WINDOW_AUTOSIZE);
    cv::imshow("blur(OMP)", b_img_OMP);

    cv::namedWindow("blur(SIMD)", cv::WINDOW_AUTOSIZE);
    cv::imshow("blur(SIMD)", b_img_simd);

    cv::namedWindow("blur(SIMD + OMP)", cv::WINDOW_AUTOSIZE);
    cv::imshow("blur(SIMD + OMP)", b_img_simd_OMP);

    cv::namedWindow("sobel(serial)", cv::WINDOW_AUTOSIZE);
    cv::imshow("sobel(serial)", out_img_serial);

    cv::namedWindow("sobel(OMP)", cv::WINDOW_AUTOSIZE);
    cv::imshow("sobel(OMP)", out_img_OMP);

    cv::namedWindow("sobel(SIMD)", cv::WINDOW_AUTOSIZE);
    cv::imshow("sobel(SIMD)", out_img_simd);

    cv::namedWindow("sobel(SIMD + OMP)", cv::WINDOW_AUTOSIZE);
    cv::imshow("sobel(SIMD + OMP)", out_img_simd_OMP);

    runFullDiagnostics(
        b_img_serial, b_img_simd, b_img_OMP, b_img_simd_OMP,
        out_img_serial, out_img_simd, out_img_OMP, out_img_simd_OMP,
        time1_serial, time1_SIMD, time1_OMP, time1_SIMD_OMP,
        time2_serial, time2_SIMD, time2_OMP, time2_SIMD_OMP);

    cv::waitKey(0);
    return 0;
}