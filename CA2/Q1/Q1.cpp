#include "stdio.h"
#include "x86intrin.h"
#include "stdlib.h"
#include "time.h"
#include <sys/time.h>
#include "immintrin.h"

#define	VECTOR_SIZE	16000000

int main() {
    struct timeval start, end;
	long seconds, micros, time1, time2;

    float sum_temp[8];
    float value_temp;
	float mean_serial, var_serial;
	float mean_parallel, var_parallel;

    float *vector;
	vector = new float [VECTOR_SIZE];

    if (!vector) {
		printf("Memory allocation error!\n");
		return 1;
	}

	for (long i = 0; i < VECTOR_SIZE; i++) {
		vector[i] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX/100.0));
	}

    // SERIAL PART:

    gettimeofday(&start, NULL);
    sum_temp[0] = sum_temp[1] = sum_temp[2] = sum_temp[3] = 0.0;
    sum_temp[4] = sum_temp[5] = sum_temp[6] = sum_temp[7] = 0.0;

    for (long i = 0; i < VECTOR_SIZE; i+=8) {
		sum_temp[0] += vector[i+0];
        sum_temp[1] += vector[i+1];
        sum_temp[2] += vector[i+2];
        sum_temp[3] += vector[i+3];
        sum_temp[4] += vector[i+4];
        sum_temp[5] += vector[i+5];
        sum_temp[6] += vector[i+6];
        sum_temp[7] += vector[i+7];
    }

    sum_temp[0] = sum_temp[0] + sum_temp[4];
    sum_temp[1] = sum_temp[1] + sum_temp[5];
    sum_temp[2] = sum_temp[2] + sum_temp[6];
    sum_temp[3] = sum_temp[3] + sum_temp[7];
    mean_serial = ((sum_temp[0] + sum_temp[1]) + (sum_temp[2] + sum_temp[3])) / VECTOR_SIZE;


    sum_temp[0] = sum_temp[1] = sum_temp[2] = sum_temp[3] = 0.0;
    sum_temp[4] = sum_temp[5] = sum_temp[6] = sum_temp[7] = 0.0;

    for (long i = 0; i < VECTOR_SIZE; i+=8) {
        value_temp = vector[i+0] - mean_serial;
		sum_temp[0] += value_temp * value_temp;

        value_temp = vector[i+1] - mean_serial;
        sum_temp[1] += value_temp * value_temp;

        value_temp = vector[i+2] - mean_serial;
        sum_temp[2] += value_temp * value_temp;

        value_temp = vector[i+3] - mean_serial;
        sum_temp[3] += value_temp * value_temp;

        value_temp = vector[i+4] - mean_serial;
        sum_temp[4] += value_temp * value_temp;

        value_temp = vector[i+5] - mean_serial;
        sum_temp[5] += value_temp * value_temp;

        value_temp = vector[i+6] - mean_serial;
        sum_temp[6] += value_temp * value_temp;

        value_temp = vector[i+7] - mean_serial;
        sum_temp[7] += value_temp * value_temp;
    }

    sum_temp[0] = sum_temp[0] + sum_temp[4];
    sum_temp[1] = sum_temp[1] + sum_temp[5];
    sum_temp[2] = sum_temp[2] + sum_temp[6];
    sum_temp[3] = sum_temp[3] + sum_temp[7];
    var_serial = ((sum_temp[0] + sum_temp[1]) + (sum_temp[2] + sum_temp[3])) / VECTOR_SIZE;

    gettimeofday(&end, NULL);
	seconds = (end.tv_sec - start.tv_sec);
	time1 = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);


    // PARALLEL PART:

    gettimeofday(&start, NULL);
    __m256 vec, diff, squared;
    __m256 sum_vec = _mm256_set1_ps(0.0);
    for (long i = 0; i < VECTOR_SIZE; i += 8) {
        vec = _mm256_load_ps(&vector[i]);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    __m128 sum_vec_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_vec_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_vec_128 = _mm_add_ps(sum_vec_low, sum_vec_high);
    sum_vec_128 = _mm_hadd_ps(sum_vec_128, sum_vec_128);
    sum_vec_128 = _mm_hadd_ps(sum_vec_128, sum_vec_128);
    mean_parallel = _mm_cvtss_f32(sum_vec_128) / VECTOR_SIZE;

    __m256 mean_vec = _mm256_set1_ps(mean_parallel);
    __m256 var_vec = _mm256_set1_ps(0.0);
    for (long i = 0; i < VECTOR_SIZE; i += 8) {
        vec = _mm256_load_ps(&vector[i]);
        diff = _mm256_sub_ps(vec, mean_vec);
        squared = _mm256_mul_ps(diff, diff);
        var_vec = _mm256_add_ps(var_vec, squared);
    }

    __m128 var_vec_low = _mm256_castps256_ps128(var_vec);
    __m128 var_vec_high = _mm256_extractf128_ps(var_vec, 1);
    __m128 var_vec_128 = _mm_add_ps(var_vec_low, var_vec_high);
    var_vec_128 = _mm_hadd_ps(var_vec_128, var_vec_128);
    var_vec_128 = _mm_hadd_ps(var_vec_128, var_vec_128);
    var_parallel = _mm_cvtss_f32(var_vec_128) / VECTOR_SIZE;

    gettimeofday(&end, NULL);
	seconds = (end.tv_sec - start.tv_sec);
	time2 = micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);


    printf ("The serial result   = %.4f\n", var_serial);
	printf ("The parallel result = %.4f\n", var_parallel);
	printf ("Serial Run time   = %ld \n", time1);
	printf ("Parallel Run time = %ld \n", time2);
	printf ("Speedup = %f\n", (double) (time1)/(double) time2);

	return 0;
}




