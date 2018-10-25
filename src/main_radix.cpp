#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 5*1000*1000+37;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radixInit(radix_kernel, radix_kernel_length, "radixInit");
        ocl::Kernel radixSum(radix_kernel, radix_kernel_length, "radixSum");
        ocl::Kernel radixShuffle(radix_kernel, radix_kernel_length, "radixShuffle");
        radixInit.compile();
        radixSum.compile();
        radixShuffle.compile();

        //#define DEBUG
        #ifdef DEBUG
        std::vector<unsigned int> prefsums0(n), prefsums1(n), as_next(n);
        #endif

        gpu::gpu_mem_32u prefsums0_gpu, prefsums1_gpu, prefsums0_next_gpu, prefsums1_next_gpu, as_next_gpu;
        prefsums0_gpu.resizeN(n);
        prefsums1_gpu.resizeN(n);
        prefsums0_next_gpu.resizeN(n);
        prefsums1_next_gpu.resizeN(n);
        as_next_gpu.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            for (int bit = 0; bit < 32; bit++) {
                radixInit.exec(gpu::WorkSize(workGroupSize, global_work_size),
                               as_gpu, prefsums0_gpu, prefsums1_gpu, n, bit);
                for (int sumBlock = 1; sumBlock < n; sumBlock *= 2) {
                    radixSum.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                  prefsums0_gpu, prefsums1_gpu, prefsums0_next_gpu, prefsums1_next_gpu,
                                  n, bit, sumBlock);
                    prefsums0_gpu.swap(prefsums0_next_gpu);
                    prefsums1_gpu.swap(prefsums1_next_gpu);
                }
                radixShuffle.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                  as_gpu, as_next_gpu, prefsums0_gpu, prefsums1_gpu, n, bit);

                #ifdef DEBUG
                as_gpu.readN(as.data(), n);
                as_next_gpu.readN(as_next.data(), n);
                prefsums0_gpu.readN(prefsums0.data(), n);
                prefsums1_gpu.readN(prefsums1.data(), n);
                std::cout << "bits:";
                for (int i = 0; i < n; i++) {
                    std::cout << " " << !!(as[i] & (1 << bit));
                }
                std::cout << "\n";
                std::cout << "prefsums0:";
                for (int i = 0; i < n; i++) {
                    std::cout << " " << prefsums0[i];
                }
                std::cout << "\n";
                std::cout << "prefsums1:";
                for (int i = 0; i < n; i++) {
                    std::cout << " " << prefsums1[i];
                }
                std::cout << "\n";
                std::cout << "new bits:";
                for (int i = 0; i < n; i++) {
                    std::cout << " " << !!(as_next[i] & (1 << bit));
                }
                std::cout << "\n";
                return 0;
                #endif
                as_gpu.swap(as_next_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
