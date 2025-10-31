#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include "ipp.h"

class Blur
{
private:
    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}};

    const int normalizationFactor = 16;

    unsigned char apply_for_one_pixel(const unsigned char *in_image, int row, unsigned int NCOLS, int col)
    {
        int pixel = 0;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                pixel += kernel[i][j] * in_image[(row + i - 1) * NCOLS + col + j - 1];
        return cv::saturate_cast<unsigned char>(pixel / normalizationFactor);
    }

public:
    Blur() {}
    ~Blur() {}

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

class Sobel
{
private:
    const int kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};

    const int kernel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}};

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
    Sobel() {}
    ~Sobel() {}

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

int main()
{
    Ipp64u time1, time2;
    Ipp64u start, end;
    cv::Mat in_img;
    in_img = cv::imread("base.jpg", cv::IMREAD_GRAYSCALE);
    
	start = ippGetCpuClocks();
    Blur blur_operator;
    cv::Mat b_img = blur_operator.apply(in_img);
    
    Sobel sobel_operator;
    cv::Mat out_img = sobel_operator.apply(b_img);
	end = ippGetCpuClocks();
    time1 = end - start;

    start = ippGetCpuClocks();
    end = ippGetCpuClocks();
    time2 = end - start;

    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);
    cv::imshow("input image", in_img);

    cv::Mat show_b_img(b_img.rows, b_img.cols, CV_8U);
    cv::resize(b_img, show_b_img, cv::Size(), 1, 1);
    cv::namedWindow("blur(serial)", cv::WINDOW_AUTOSIZE);
    cv::imshow("blur(serial)", show_b_img);

    cv::Mat show_out_img(out_img.rows, out_img.cols, CV_8U);
    cv::resize(out_img, show_out_img, cv::Size(), 1, 1);
    cv::namedWindow("output(serial)", cv::WINDOW_AUTOSIZE);
    cv::imshow("output(serial)", show_out_img);

	printf ("Serial Run time = %u \n", (Ipp32s) time1);
	printf ("Parallel Run time = %u \n", (Ipp32s) time2);
	printf ("Speedup = %f\n\n", (float) (time1)/(float) time2);

    cv::waitKey(0);
    return 0;
}