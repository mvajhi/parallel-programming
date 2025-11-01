#include <cstdlib> 
#include <iostream>
#include <sys/time.h>
#include "x86intrin.h"
#include "immintrin.h"
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

int main() {
    struct timeval start, end;
    long seconds, micros, time1, time2;

    cv::Mat base_img = cv::imread("./assets/base.jpg", IMREAD_COLOR);
    cv::Mat watermark_img = cv::imread("./assets/watermark.png", IMREAD_UNCHANGED);
    cv::resize(watermark_img, watermark_img, base_img.size());
    
    // SERIAL PART:

    gettimeofday(&start, NULL);

    unsigned char *base_image = (unsigned char *) base_img.data;
    unsigned char *watermark_image = (unsigned char *) watermark_img.data;

    unsigned int NCOLS = base_img.cols;
    unsigned int NROWS = base_img.rows;
    unsigned int BASE_CHANNELS = base_img.channels();
    unsigned int WATERMARK_CHANNELS = watermark_img.channels();

    cv::Mat out_img1(base_img.rows, base_img.cols, CV_8UC3);
    unsigned char *out_image1 = (unsigned char *) out_img1.data;

    for (int row = 0; row < NROWS; row++) {
        for (int col = 0; col < NCOLS; col++) {
            float gradient_alpha = (float)(col + row) / (float)(NCOLS + NROWS);
            int base_pixel_idx = (row * NCOLS + col) * BASE_CHANNELS;
            int watermark_pixel_idx = (row * NCOLS + col) * WATERMARK_CHANNELS;
            unsigned char watermark_alpha_value = *(watermark_image + watermark_pixel_idx + 3);
            float watermark_transparency = watermark_alpha_value / 255.0;
            float combined_alpha = gradient_alpha * watermark_transparency;
            
            for (int ch = 0; ch < BASE_CHANNELS; ch++) {
                unsigned char base_pixel_value = *(base_image + base_pixel_idx + ch);
                unsigned char watermark_pixel_value = *(watermark_image + watermark_pixel_idx + ch);
                
                float new_value = combined_alpha * watermark_pixel_value + (1.0 - combined_alpha) * base_pixel_value;
                *(out_image1 + base_pixel_idx + ch) = new_value;
            }
        }
    }

    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
	time1 = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    
    // PARALLEL PART:

    gettimeofday(&start, NULL);

    cv::Mat out_img2(base_img.rows, base_img.cols, CV_8UC3);
    unsigned char *out_image2 = (unsigned char *) out_img2.data;

    unsigned char *pBase = (unsigned char*) base_img.data;
    unsigned char *pWatermarkBGRA = (unsigned char*) watermark_img.data;
    unsigned char *pRes = (unsigned char*) out_img2.data;

    int stride_wmk = 8;
    int num_pixels = NROWS * NCOLS;
    int bytes_per_pixel = base_img.channels();
    int bytes_per_row = base_img.cols * bytes_per_pixel;    
    int total_dim = NROWS + NCOLS;
    int delta_alpha = 255/total_dim;

    const __m256i zeros = _mm256_setzero_si256();
    const __m256i full_alpha = _mm256_set1_epi16(255);
    
    const __m128i mask_12_pad =       _mm_setr_epi8(0, 1, 2, 3, 4, 5,  6,  7,  8,  9, 10, 11, -1, -1, -1, -1);
    const __m128i shuffle_bgr_128 =   _mm_setr_epi8(0, 1, 2, 4, 5, 6,  8,  9, 10, 12, 13, 14, -1, -1, -1, -1);
    const __m128i shuffle_alpha_128 = _mm_setr_epi8(3, 3, 3, 7, 7, 7, 11, 11, 11, 15, 15, 15, -1, -1, -1, -1);
    const __m256i shuffle_bgr   = _mm256_broadcastsi128_si256(shuffle_bgr_128);
    const __m256i shuffle_alpha = _mm256_broadcastsi128_si256(shuffle_alpha_128);
    const __m256i m_alpha_offset = _mm256_setr_epi16(
        0*delta_alpha, 0*delta_alpha, 0*delta_alpha, 1*delta_alpha, 1*delta_alpha, 1*delta_alpha,
        2*delta_alpha, 2*delta_alpha, 2*delta_alpha, 3*delta_alpha, 3*delta_alpha, 3*delta_alpha, 
        4*delta_alpha, 4*delta_alpha, 4*delta_alpha, 5*delta_alpha
    );

    for (int px_idx = 0; px_idx <= num_pixels - stride_wmk; px_idx += stride_wmk) {
        int wmk_byte_idx = px_idx * WATERMARK_CHANNELS;
        int base_byte_idx = px_idx * BASE_CHANNELS;
        int curr_row = px_idx / NCOLS;
        int curr_col = px_idx % NCOLS;
        int alpha_num = curr_row + curr_col;
        
        int alpha_for_img = (255 * alpha_num) / total_dim;
        __m256i m_grad_alpha_base = _mm256_set1_epi16(alpha_for_img);
        __m256i m_grad_alpha = _mm256_add_epi16(m_grad_alpha_base, m_alpha_offset);
        
        __m128i m_base_lo = _mm_loadu_si128((__m128i*)(pBase + base_byte_idx));
        __m128i m_base_lo_padded = _mm_shuffle_epi8(m_base_lo, mask_12_pad);
        __m128i m_base_hi = _mm_loadu_si128((__m128i*)(pBase + base_byte_idx + 12));
        __m128i m_base_hi_padded = _mm_shuffle_epi8(m_base_hi, mask_12_pad);
        __m256i m_base_bgr = _mm256_set_m128i(m_base_hi_padded, m_base_lo_padded);

        __m256i m_wmk_bgra = _mm256_loadu_si256((__m256i*)(pWatermarkBGRA + wmk_byte_idx));
        __m256i m_wmk_bgr = _mm256_shuffle_epi8(m_wmk_bgra, shuffle_bgr);
        __m256i m_wmk_alpha_8bit = _mm256_shuffle_epi8(m_wmk_bgra, shuffle_alpha);


        __m256i wmk_alpha_lo = _mm256_unpacklo_epi8(m_wmk_alpha_8bit, zeros);
        __m256i wmk_alpha_hi = _mm256_unpackhi_epi8(m_wmk_alpha_8bit, zeros);
        __m256i combined_alpha_lo = _mm256_srli_epi16(_mm256_mullo_epi16(m_grad_alpha, wmk_alpha_lo), 8);
        __m256i combined_alpha_hi = _mm256_srli_epi16(_mm256_mullo_epi16(m_grad_alpha, wmk_alpha_hi), 8);

        __m256i one_minus_alpha_lo = _mm256_sub_epi16(full_alpha, combined_alpha_lo);
        __m256i one_minus_alpha_hi = _mm256_sub_epi16(full_alpha, combined_alpha_hi);

        __m256i base_lo = _mm256_unpacklo_epi8(m_base_bgr, zeros);
        __m256i wmk_lo  = _mm256_unpacklo_epi8(m_wmk_bgr, zeros);
        __m256i base_hi = _mm256_unpackhi_epi8(m_base_bgr, zeros);
        __m256i wmk_hi  = _mm256_unpackhi_epi8(m_wmk_bgr, zeros);
        
        __m256i term1_lo = _mm256_mullo_epi16(combined_alpha_lo, wmk_lo);
        __m256i term2_lo = _mm256_mullo_epi16(one_minus_alpha_lo, base_lo);
        __m256i sum_lo   = _mm256_add_epi16(term1_lo, term2_lo);
        __m256i result_lo = _mm256_srli_epi16(sum_lo, 8);

        __m256i term1_hi = _mm256_mullo_epi16(combined_alpha_hi, wmk_hi);
        __m256i term2_hi = _mm256_mullo_epi16(one_minus_alpha_hi, base_hi);
        __m256i sum_hi   = _mm256_add_epi16(term1_hi, term2_hi);
        __m256i result_hi = _mm256_srli_epi16(sum_hi, 8);

        __m256i final_result = _mm256_packus_epi16(result_lo, result_hi);
        __m128i res_low128  = _mm256_castsi256_si128(final_result);
        __m128i res_high128 = _mm256_extracti128_si256(final_result, 1);
        _mm_storeu_si128((__m128i*)(pRes + base_byte_idx), res_low128);       
        _mm_storeu_si128((__m128i*)(pRes + base_byte_idx + 12), res_high128);
    }

    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
    time2 = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);

    cv::imwrite("output_serial.png", out_img1);
    cv::imwrite("output_parallel.png", out_img2);

    printf("Serial Run time = %ld \n", time1);
    printf("Parallel Run time = %ld \n", time2);
    printf("Speedup = %4.2f\n", (float) (time1)/(float) time2);
    
    return 0;
}



