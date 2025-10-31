#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

const int sobel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

const int sobel_y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1},
};

const int blur[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1},
};

int main()
{
    cv::Mat in_img;
    in_img = cv::imread("base.jpg", cv::IMREAD_GRAYSCALE);

    unsigned char *in_image = (unsigned char *)in_img.data;

    unsigned int NCOLS = in_img.cols;
    unsigned int NROWS = in_img.rows;

    cv::Mat out_img(in_img.rows, in_img.cols, CV_8U);
    unsigned char *out_image = (unsigned char *)out_img.data;

    cv::Mat b_img(in_img.rows, in_img.cols, CV_8U);
    unsigned char *b_image = (unsigned char *)b_img.data;

    for (int row = 1; row < NROWS - 1; row++)
        for (int col = 1; col < NCOLS - 1; col++)
        {
            int tmp = 0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    tmp += blur[i][j] * in_image[(row + i - 1) * NCOLS + col + j - 1];
                }
            b_image[row * NCOLS + col] = tmp / 16;
        }

    for (int row = 1; row < NROWS - 1; row++)
        for (int col = 1; col < NCOLS - 1; col++)
        {
            int tmp = 0;
            int tmp2 = 0;
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                {
                    tmp += sobel_x[i][j] * b_image[(row + i - 1) * NCOLS + col + j - 1];
                    tmp2 += sobel_y[i][j] * b_image[(row + i - 1) * NCOLS + col + j - 1];
                }
            out_image[row * NCOLS + col] = MIN(MAX(sqrt(tmp * tmp + tmp2 * tmp2), 0), 255);
        }

    cv::namedWindow("input image", cv::WINDOW_AUTOSIZE);

    cv::imshow("input image", in_img);

    cv::Mat show_b_img(b_img.rows, b_img.cols, CV_8U);
    cv::resize(b_img, show_b_img, cv::Size(), 1, 1);
    cv::namedWindow("b", cv::WINDOW_AUTOSIZE);
    cv::imshow("b", show_b_img);

    cv::Mat show_out_img(out_img.rows, out_img.cols, CV_8U);
    cv::resize(out_img, show_out_img, cv::Size(), 1, 1);
    cv::namedWindow("output(serial)", cv::WINDOW_AUTOSIZE);
    cv::imshow("output(serial)", show_out_img);

    cv::waitKey(0);
    return 0;
}