#include "omp.h"
#include <vector>
#include <cstdlib>
#include <complex>
#include <iostream>
#include <sys/time.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

cv::Mat generate_julia_set_serial(int n, float c_real, float c_imag, int width, int height, int max_iter) {
    cv::Mat image_base(height, width, CV_8UC1);
    unsigned char *image_base_data = (unsigned char *) image_base.data;
    
    float min_re = -1.5, max_re = 1.5;
    float min_im = -1.5, max_im = 1.5;
    float factor_re = (max_re-min_re) / (width-1);
    float factor_im = (max_im-min_im) / (height-1);

    complex<float> c(c_real, c_imag);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            complex<float> z(min_re + x*factor_re, max_im - y*factor_im);
            int iter = 0;
            bool flag = false;

            while (iter < max_iter) {            
                if (norm(z) > 4) {
                    flag = true;
                    break;
                }
                z = pow(z, n) + c;
                iter++;
            }

            int base_pixel_idx = y * width + x;

            if (flag == true) {
                float smooth_val = sqrt((float) iter / max_iter) * 255;
                *(image_base_data + base_pixel_idx) = smooth_val;
            } 
            else {
                *(image_base_data + base_pixel_idx) = 255;
            }
        }
    }

    cv::Mat image_out;
    applyColorMap(image_base, image_out, COLORMAP_JET);
    return image_out;
}


cv::Mat generate_julia_set_parallel(int n, float c_real, float c_imag, int width, int height, int max_iter) {
    cv::Mat image_base(height, width, CV_8UC1);
    unsigned char *image_base_data = (unsigned char *) image_base.data;
    
    float min_re = -1.5, max_re = 1.5;
    float min_im = -1.5, max_im = 1.5;
    float factor_re = (max_re-min_re) / (width-1);
    float factor_im = (max_im-min_im) / (height-1);

    complex<float> c(c_real, c_imag);

    #pragma omp parallel for schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            complex<float> z(min_re + x*factor_re, max_im - y*factor_im);
            int iter = 0;
            bool flag = false;

            while (iter < max_iter) {            
                if (norm(z) > 4) {
                    flag = true;
                    break;
                }
                z = pow(z, n) + c;
                iter++;
            }

            int base_pixel_idx = y * width + x;

            if (flag == true) {
                float smooth_val = sqrt((float) iter / max_iter) * 255;
                *(image_base_data + base_pixel_idx) = smooth_val;
            } 
            else {
                *(image_base_data + base_pixel_idx) = 255;
            }
        }
    }

    cv::Mat image_out;
    applyColorMap(image_base, image_out, COLORMAP_JET);
    return image_out;
}



int main() {
    struct timeval start, end;
    long seconds, micros, time_serial, time_omp;
    int width = 2000, height = 2000;
    int max_iterations = 1000;
    int n = 2;
    float c_re = -0.7, c_im = 0.27015; 

    // SERIAL PART:

    gettimeofday(&start, NULL);

    cv::Mat out_image_serial = generate_julia_set_serial(n, c_re, c_im, width, height, max_iterations);

    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
	time_serial = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);


    // OMP PARALLEL PART:

    gettimeofday(&start, NULL);

    cv::Mat out_image_omp = generate_julia_set_parallel(n, c_re, c_im, width, height, max_iterations);

    gettimeofday(&end, NULL);
    seconds = (end.tv_sec - start.tv_sec);
	time_omp = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    
    
    // Results:

    cv::imwrite("output_julia_serial.png", out_image_serial);
    cv::imwrite("output_julia_omp.png", out_image_omp);

    printf("Serial Run time         = %ld \n", time_serial);
	printf("OMP Parallel Run time   = %ld \n", time_omp);
	printf("Speedup (OMP vs Serial) = %f\n", (double) (time_serial)/(double) time_omp);
}