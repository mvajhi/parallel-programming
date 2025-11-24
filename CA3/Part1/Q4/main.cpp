#include "stdio.h"
#include "x86intrin.h"
#include <iostream>
#include <iomanip>
#include <omp.h>

#define NUM_THREADS 8

using namespace std;

const int INPUT_SIZE = 8000;
const int HIDDEN_SIZE = 16000;
const int OUTPUT_SIZE = 8;

void print_list(float *list, int size)
{
	std::cout << std::fixed;
	std::cout << std::setprecision(3);
	cout << "[";
	for (int i = 0; i < size; i++)
		cout << list[i] << " ,";
	cout << "]\n";
}

void print_matrix(float **matrix, int row, int col)
{
	cout << "{\n";
	for (int i = 0; i < row; i++)
		print_list(matrix[i], col);
	cout << "}\n";
}

void free_mat(float **matrix, int row)
{
	for (int i = 0; i < row; i++)
		free(matrix[i]);
}

float *create_list(int size)
{
	float *list = (float *)_mm_malloc(sizeof(float) * size, sizeof(float) * 8);
	for (int i = 0; i < size; i++)
		list[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 200.0f - 100.0f;
	return list;
}

float **create_matrix(int row, int col)
{
	float **mat = (float **)_mm_malloc(sizeof(float *) * row, sizeof(float *) * 8);
	for (int i = 0; i < row; i++)
		mat[i] = create_list(col);
	return mat;
}

class Layer
{
protected:
	float **weights;
	float bias;
	int input_size;
	float *out;

	inline float ReLU(float x)
	{
		return (0.0 < x) ? x : 0.0;
	}

public:
	int output_size;

	Layer(int input_size, int output_size) : input_size(input_size), output_size(output_size), weights(nullptr), out(nullptr)
	{
		this->out = (float *)_mm_malloc(sizeof(float) * output_size, sizeof(float) * 8);
	}

	~Layer()
	{
		_mm_free(out);
	}

	void set_weights(float **weights, float bias)
	{
		this->weights = weights;
		this->bias = bias;
	}

	virtual float *forward(float *x) = 0;
};

class SerialLayer : public Layer
{
public:
	SerialLayer(int input_size, int output_size)
		: Layer(input_size, output_size) {}

	float *forward(float *x)
	{
		for (int i = 0; i < output_size; i++)
		{
			float sum = bias;

			for (int j = 0; j < input_size; j++)
				sum += x[j] * weights[i][j];

			out[i] = ReLU(sum);
		}

		return out;
	}
};

class OMPLayer : public Layer
{
public:
	OMPLayer(int input_size, int output_size)
		: Layer(input_size, output_size) {}

	float *forward(float *x)
	{
#pragma omp parallel for
		for (int i = 0; i < output_size; i++)
		{
			float sum = bias;

			for (int j = 0; j < input_size; j++)
				sum += x[j] * weights[i][j];

			out[i] = ReLU(sum);
		}

		return out;
	}
};

class SIMDLayer : public Layer
{
public:
	SIMDLayer(int input_size, int output_size)
		: Layer(input_size, output_size) {}

	float *forward(float *x) override
	{
		__m256 mac, x_tmp, w_tmp;
		__m128 sum, high, low, relu;
		for (int i = 0; i < output_size; i++)
		{
			mac = _mm256_set1_ps(0.0);
			for (int j = 0; j < input_size; j += 8)
			{
				x_tmp = _mm256_load_ps(&x[j]);
				w_tmp = _mm256_load_ps(&weights[i][j]);
				mac = _mm256_fmadd_ps(x_tmp, w_tmp, mac);
			}

			// mac = [a0, a1, a2, a3, a4, a5, a6, a7]
			low = _mm256_castps256_ps128(mac);
			high = _mm256_extractf128_ps(mac, 1);
			// sum = [a0+a4, a1+a5, a2+a6, a3+a7]
			sum = _mm_add_ps(low, high);

			// sum = [a0+a4+a1+a5, a2+a6+a3+a7, a0+a4+a1+a5, a2+a6+a3+a7]
			sum = _mm_hadd_ps(sum, sum);
			// sum = [sum(0,7), sum(0,7), sum(0,7), sum(0,7)]
			sum = _mm_hadd_ps(sum, sum);
			// sum = [sum(0,7)+bias, sum(0,7), sum(0,7), sum(0,7)]
			sum = _mm_add_ss(sum, _mm_set_ss(bias));

			relu = _mm_max_ss(sum, _mm_setzero_ps());

			out[i] = _mm_cvtss_f32(relu);
		}

		return out;
	}
};

class SIMD_OMPLayer : public Layer
{
public:
	SIMD_OMPLayer(int input_size, int output_size)
		: Layer(input_size, output_size) {}

	float *forward(float *x) override
	{
#pragma omp parallel for
		for (int i = 0; i < output_size; i++)
		{
			__m256 mac, x_tmp, w_tmp;
			__m128 sum, high, low, relu;
			mac = _mm256_set1_ps(0.0);
			for (int j = 0; j < input_size; j += 8)
			{
				x_tmp = _mm256_load_ps(&x[j]);
				w_tmp = _mm256_load_ps(&weights[i][j]);
				mac = _mm256_fmadd_ps(x_tmp, w_tmp, mac);
			}

			low = _mm256_castps256_ps128(mac);
			high = _mm256_extractf128_ps(mac, 1);
			sum = _mm_add_ps(low, high);
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_hadd_ps(sum, sum);
			sum = _mm_add_ss(sum, _mm_set_ss(bias));
			relu = _mm_max_ss(sum, _mm_setzero_ps());
			out[i] = _mm_cvtss_f32(relu);
		}

		return out;
	}
};

double execute_and_print(float *input, Layer &l1, Layer &l2, Layer &l3)
{
	double start, end;
	float *x = input;

	start = omp_get_wtime();

	x = l1.forward(x);
	x = l2.forward(x);
	x = l3.forward(x);

	end = omp_get_wtime();

	double time = end - start;
	print_list(x, l3.output_size);

	return time;
}

int main(void)
{
	omp_set_num_threads(NUM_THREADS);

	double time_serial, time_omp, time_simd, time_simd_omp;
	auto seed = static_cast<unsigned int>(std::time(nullptr));
	srand(seed);

	cout << "Seed: " << seed << "\n";
	cout << "Params: " << INPUT_SIZE << ", " << HIDDEN_SIZE << ", " << OUTPUT_SIZE << "\n";

	float *input = create_list(INPUT_SIZE);

	float **weights_l1 = create_matrix(HIDDEN_SIZE, INPUT_SIZE);
	float **weights_l2 = create_matrix(HIDDEN_SIZE, HIDDEN_SIZE);
	float **weights_l3 = create_matrix(OUTPUT_SIZE, HIDDEN_SIZE);

	float bias_l1 = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 200.0f - 100.0f;
	float bias_l2 = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 200.0f - 100.0f;
	float bias_l3 = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 200.0f - 100.0f;

	cout << "\n--- Serial Execution ---" << endl;
	{
		SerialLayer l1(INPUT_SIZE, HIDDEN_SIZE);
		l1.set_weights(weights_l1, bias_l1);
		SerialLayer l2(HIDDEN_SIZE, HIDDEN_SIZE);
		l2.set_weights(weights_l2, bias_l2);
		SerialLayer l3(HIDDEN_SIZE, OUTPUT_SIZE);
		l3.set_weights(weights_l3, bias_l3);

		time_serial = execute_and_print(input, l1, l2, l3);
	}

	cout << "\n--- OMP Execution ---" << endl;
	{
		OMPLayer l1(INPUT_SIZE, HIDDEN_SIZE);
		l1.set_weights(weights_l1, bias_l1);
		OMPLayer l2(HIDDEN_SIZE, HIDDEN_SIZE);
		l2.set_weights(weights_l2, bias_l2);
		OMPLayer l3(HIDDEN_SIZE, OUTPUT_SIZE);
		l3.set_weights(weights_l3, bias_l3);

		time_omp = execute_and_print(input, l1, l2, l3);
	}

	cout << "\n--- SIMD Execution ---" << endl;
	{
		SIMDLayer l1(INPUT_SIZE, HIDDEN_SIZE);
		l1.set_weights(weights_l1, bias_l1);
		SIMDLayer l2(HIDDEN_SIZE, HIDDEN_SIZE);
		l2.set_weights(weights_l2, bias_l2);
		SIMDLayer l3(HIDDEN_SIZE, OUTPUT_SIZE);
		l3.set_weights(weights_l3, bias_l3);

		time_simd = execute_and_print(input, l1, l2, l3);
	}

	cout << "\n--- SIMD + OMP Execution ---" << endl;
	{
		SIMD_OMPLayer l1(INPUT_SIZE, HIDDEN_SIZE);
		l1.set_weights(weights_l1, bias_l1);
		SIMD_OMPLayer l2(HIDDEN_SIZE, HIDDEN_SIZE);
		l2.set_weights(weights_l2, bias_l2);
		SIMD_OMPLayer l3(HIDDEN_SIZE, OUTPUT_SIZE);
		l3.set_weights(weights_l3, bias_l3);

		time_simd_omp = execute_and_print(input, l1, l2, l3);
	}

	cout << "\n------------------------------------------------\n";
	printf("Serial Time     = %.6f s\n", time_serial);
	printf("OMP Time        = %.6f s (Speedup: %.2fx)\n", time_omp, time_serial / time_omp);
	printf("SIMD Time       = %.6f s (Speedup: %.2fx)\n", time_simd, time_serial / time_simd);
	printf("SIMD + OMP Time = %.6f s (Speedup: %.2fx)\n", time_simd_omp, time_serial / time_simd_omp);
	cout << "------------------------------------------------\n";

	return 0;
}