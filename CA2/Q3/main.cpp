#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include "stdio.h"
#include "ipp.h"
#include "x86intrin.h"

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
private:
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

        cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);

        const unsigned char *in_image = (const unsigned char *)in_img.data;
        unsigned char *out_image = (unsigned char *)out_img.data;

        for (int row = 1; row < NROWS - 1; row++)
            for (int col = 1; col < NCOLS - 1; col++)
                out_image[row * NCOLS + col] = apply_for_one_pixel(in_image, row, NCOLS, col);
        return out_img;
    }
};

class BlurSIMD : public Blur
{
private:
    
    void apply_for_16_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col, unsigned char *out_image)
    {
        __m256i p_8, p_16_low, p_16_high, k_value, product;
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
                // pixel += kernel[i][j] * in_image[(row + i - 1) * NCOLS + col + j - 1];
                p_8 = _mm256_loadu_si256((__m256i const*)&in_image[(row + i - 1) * NCOLS + col + j - 1]);
                p_16_low = _mm256_unpacklo_epi8(p_8, zero);
                p_16_high = _mm256_unpackhi_epi8(p_8, zero);
                
                k_value = _mm256_set1_epi16((short int) kernel[i][j]);

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
        reordered = _mm256_permute4x64_epi64(packed, (2<<6)|(3<<4)|(0<<2)|1);
        _mm256_storeu_si256((__m256i_u*) &out_image[row * NCOLS + col], reordered);
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
            for (int col = 1; col < NCOLS - 1; col+=32)
            // for (int col = 1; col < 2; col+=32)
                apply_for_16_pixel(in_image, row, NCOLS, col, out_image);
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
private:
    unsigned char apply_for_one_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col)
    {
        int gx = 0;
        int gy = 0;

        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
            {
                int pixel_val = in_image[(row + i - 1) * NCOLS + col + j - 1];
                gx += kernel_x[i][j] * pixel_val;
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

int main()
{
    Ipp64u time1, time2;
    Ipp64u start, end;
    cv::Mat in_img;
    in_img = cv::imread("base.jpg", cv::IMREAD_GRAYSCALE);

    start = ippGetCpuClocks();
    BlurSerial blur_operator;
    cv::Mat b_img = blur_operator.apply(in_img);
    
    SobelSerial sobel_operator;
    cv::Mat out_img = sobel_operator.apply(b_img);
    end = ippGetCpuClocks();
    time1 = end - start;

    start = ippGetCpuClocks();
    BlurSIMD blur_operator2;
    cv::Mat b_img2 = blur_operator.apply(in_img);

    end = ippGetCpuClocks();
    time2 = end - start;
    
    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::imshow("input image", in_img);
    
    cv::Mat show_b_img(b_img.rows, b_img.cols, CV_8U);
    cv::resize(b_img, show_b_img, cv::Size(), 1, 1);
    cv::namedWindow("blur(serial)", cv::WINDOW_AUTOSIZE);
    cv::imshow("blur(serial)", show_b_img);
    
    cv::Mat show_b_img2(b_img2.rows, b_img2.cols, CV_8U);
    cv::resize(b_img2, show_b_img2, cv::Size(), 1, 1);
    cv::namedWindow("blur(SIMD)", cv::WINDOW_AUTOSIZE);
    cv::imshow("blur(SIMD)", show_b_img2);

    // cv::Mat show_out_img(out_img.rows, out_img.cols, CV_8U);
    // cv::resize(out_img, show_out_img, cv::Size(), 1, 1);
    // cv::namedWindow("output(serial)", cv::WINDOW_AUTOSIZE);
    // cv::imshow("output(serial)", show_out_img);

    printf("Serial Run time = %u \n", (Ipp32s)time1);
    printf("Parallel Run time = %u \n", (Ipp32s)time2);
    printf("Speedup = %f\n\n", (float)(time1) / (float)time2);

    cv::waitKey(0);
    return 0;
}