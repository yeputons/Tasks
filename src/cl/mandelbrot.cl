#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define threshold 256.0f
#define threshold2 (threshold * threshold)

__kernel void mandelbrot(__global float *results, float fromX, float fromY, float sizeX, float sizeY, unsigned int iters, unsigned int smoothing)
{
	int i = get_global_id(0);
	int j = get_global_id(1);
	unsigned int width = get_global_size(0);
	unsigned int height = get_global_size(1);

	float x0 = fromX + (i + 0.5f) * sizeX / width;
	float y0 = fromY + (j + 0.5f) * sizeY / height;

	float x = x0;
	float y = y0;

	int iter = 0;
	for (; iter < iters; ++iter) {
		float xPrev = x;
		x = x * x - y * y + x0;
		y = 2.0f * xPrev * y + y0;
		if ((x * x + y * y) > threshold2) {
			break;
		}
	}
	float result = iter;
	if (smoothing && iter != iters) {
		result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
	}

	result = 1.0f * result / iters;
	results[j * width + i] = result;
    // TODO если хочется избавиться от зернистости и дрожжания при интерактивном погружении - добавьте anti-aliasing:
    // грубо говоря при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
