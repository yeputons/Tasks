#define GROUP_SIZE 1024

__kernel void sum(__global unsigned int *global_as, __global unsigned int *total_sum)
{
	int global_i = get_global_id(0);
	int global_n = get_global_size(0);

	int local_i = get_local_id(0);
	__local unsigned int local_sums[GROUP_SIZE];
	local_sums[local_i] = global_as[global_i];
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int block = 1; block < GROUP_SIZE; block *= 2) {
		if (local_i % (2 * block) == 0) {
			local_sums[local_i] += local_sums[local_i + block];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (local_i == 0) {
		atomic_add(total_sum, local_sums[0]);
	}
}