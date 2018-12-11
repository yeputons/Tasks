#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    const unsigned int M = 1024;
    const unsigned int K = 1024;

    std::vector<float> as(M*K, 0);
    std::vector<float> as_t(M*K, 0);

    FastRandom r(M + K);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << "!" << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M*K);
    as_t_gpu.resizeN(K*M);

    as_gpu.writeN(as.data(), M*K);

    const int GROUP_SIZE = 32;

    const auto measure = [&](ocl::Kernel &kernel)
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            const int WORK_SIZE_M = (M + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
            const int WORK_SIZE_K = (K + GROUP_SIZE - 1) / GROUP_SIZE * GROUP_SIZE;
            kernel.exec(gpu::WorkSize(GROUP_SIZE, GROUP_SIZE, WORK_SIZE_M, WORK_SIZE_K), as_gpu, as_t_gpu, M, K);
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << M * K / 1000.0 / 1000.0 / t.lapAvg() << " millions/s" << std::endl;
    };

    ocl::Kernel matrix_transpose_kernel_naive(matrix_transpose, matrix_transpose_length, "matrix_transpose", "-DNAIVE");
    matrix_transpose_kernel_naive.compile();
    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose", "-DGROUP_SIZE=" + to_string(GROUP_SIZE));
    matrix_transpose_kernel.compile();

    std::cout << "Naive:\n";
    measure(matrix_transpose_kernel_naive);

    std::cout << "Coalesced:\n";
    measure(matrix_transpose_kernel);
    as_t_gpu.readN(as_t.data(), M*K);

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
