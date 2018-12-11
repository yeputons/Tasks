#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include "libgpu/context.h"
#include "cl/max_prefix_sum_cl.h"
#include "libgpu/shared_device_buffer.h"
#include <algorithm>

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
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
			gpu::Device device = gpu::chooseGPUDevice(argc, argv);
			gpu::Context context;
			context.init(device.device_id_opencl);
			context.activate();

			ocl::Kernel fill_zero(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "fill_zero");
			fill_zero.compile(/*printLog=*/ false);

			ocl::Kernel fill_zero_i(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "fill_zero_i");
			fill_zero_i.compile(/*printLog=*/ false);

			ocl::Kernel prefix_sum_fwd_local(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_sum_fwd_local");
			prefix_sum_fwd_local.compile(/*printLog=*/ false);

			ocl::Kernel prefix_sum_fwd(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_sum_fwd");
			prefix_sum_fwd.compile(/*printLog=*/ false);

			ocl::Kernel prefix_sum_bwd(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_sum_bwd");
			prefix_sum_bwd.compile(/*printLog=*/ false);

			ocl::Kernel prefix_sum_bwd_local(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "prefix_sum_bwd_local");
			prefix_sum_bwd_local.compile(/*printLog=*/ false);

			ocl::Kernel max_val(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_val");
			max_val.compile(/*printLog=*/ false);

			const int GROUP_SIZE = 1024;
			const int WORK_SIZE = (n + 2 * GROUP_SIZE - 1) / (2 * GROUP_SIZE) * (2 * GROUP_SIZE);

			gpu::shared_device_buffer_typed<cl_int> as_buffer, prefsum_buffer, max_id_buffer;
			as_buffer.resizeN(WORK_SIZE);
			fill_zero.exec(gpu::WorkSize(GROUP_SIZE,WORK_SIZE), as_buffer);
			as_buffer.writeN(as.data(), n);
			prefsum_buffer.resizeN(WORK_SIZE);
			max_id_buffer.resizeN(WORK_SIZE);

			timer t;
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				as_buffer.copyToN(prefsum_buffer, WORK_SIZE);
				prefix_sum_fwd_local.exec(gpu::WorkSize(GROUP_SIZE, WORK_SIZE / 2), prefsum_buffer);
				for (int step = 2 * GROUP_SIZE; step < WORK_SIZE; step *= 2) {
					prefix_sum_fwd.exec(gpu::WorkSize(GROUP_SIZE, std::max(GROUP_SIZE, WORK_SIZE / step / 2)), prefsum_buffer, WORK_SIZE, step);
				}
				cl_int total_sum;
				prefsum_buffer.readN(&total_sum, 1, WORK_SIZE - 1);
				fill_zero_i.exec(gpu::WorkSize(1, 1), prefsum_buffer, WORK_SIZE - 1);
				for (int step = WORK_SIZE / 2; step >= 2 * GROUP_SIZE; step /= 2) {
					prefix_sum_bwd.exec(gpu::WorkSize(GROUP_SIZE, std::max(GROUP_SIZE, WORK_SIZE / step / 2)), prefsum_buffer, WORK_SIZE, step);
				}
				prefix_sum_bwd_local.exec(gpu::WorkSize(GROUP_SIZE, WORK_SIZE / 2), prefsum_buffer);

				for (int step = 1; step < WORK_SIZE; step *= 2)
				{
					max_val.exec(gpu::WorkSize(GROUP_SIZE, std::max(GROUP_SIZE, WORK_SIZE / step / 2)), prefsum_buffer, max_id_buffer, WORK_SIZE, step);
				}

				int max_sum, result;
				prefsum_buffer.readN(&max_sum, 1);
				max_id_buffer.readN(&result, 1);
				if (max_sum < total_sum)
				{
					max_sum = total_sum;
					result =  WORK_SIZE;
				}
				EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
				EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
				t.nextLap();
			}
			std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}
    }
}
