#define GROUP_SIZE 1024

__kernel void fill_zero(__global int *as) {
	as[get_global_id(0)] = 0;
}

__kernel void fill_zero_i(__global int *as, int i) {
	if (get_global_id(0) == 0) {
		as[i] = 0;
	}
}

__kernel void prefix_sum_fwd_local(__global int *global_prefsum) {
	int global_i = get_global_id(0);
	global_i += global_i / GROUP_SIZE * GROUP_SIZE;

	int local_i = get_local_id(0);

	__local int local_prefsum[2 * GROUP_SIZE];
	local_prefsum[local_i] = global_prefsum[global_i];
	local_prefsum[local_i + GROUP_SIZE] = global_prefsum[global_i + GROUP_SIZE];
	barrier(CLK_LOCAL_MEM_FENCE);

	int max_local_i = GROUP_SIZE;
	for (int step = 1; step <= GROUP_SIZE; step *= 2, max_local_i /= 2) {
		if (local_i < max_local_i) {
			int mid_id = local_i * 2 * step + step - 1;
			int last_id = mid_id + step;
			local_prefsum[last_id] += local_prefsum[mid_id];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	global_prefsum[global_i] = local_prefsum[local_i];
	global_prefsum[global_i + GROUP_SIZE] = local_prefsum[local_i + GROUP_SIZE];
}

__kernel void prefix_sum_fwd(__global int *global_prefsum, int n, int step) {
	int max_id = n / (2 * step);
	int start_id = get_global_id(0);
	if (start_id >= max_id) return;

	start_id = start_id * 2 * step;
	int mid_id = start_id + step - 1;
	int last_id = mid_id + step;

	global_prefsum[last_id] += global_prefsum[mid_id];
}

__kernel void prefix_sum_bwd(__global int *global_prefsum, int n, int step) {
	int max_id = n / (2 * step);
	int start_id = get_global_id(0);
	if (start_id >= max_id) return;

	start_id = start_id * 2 * step;
	int mid_id = start_id + step  -1;
	int last_id = mid_id + step;

	int sumL = global_prefsum[mid_id];
	int toAdd = global_prefsum[last_id];
	global_prefsum[mid_id] = toAdd;
	global_prefsum[last_id] += sumL;
}

__kernel void prefix_sum_bwd_local(__global int *global_prefsum) {
	int global_i = get_global_id(0);
	global_i += global_i / GROUP_SIZE * GROUP_SIZE;

	int local_i = get_local_id(0);

	__local int local_prefsum[2 * GROUP_SIZE];
	local_prefsum[local_i] = global_prefsum[global_i];
	local_prefsum[local_i + GROUP_SIZE] = global_prefsum[global_i + GROUP_SIZE];
	barrier(CLK_LOCAL_MEM_FENCE);

	int max_local_i = 1;
	for (int step = GROUP_SIZE; step >= 1; step /= 2, max_local_i *= 2) {
		if (local_i < max_local_i) {
			int mid_id = local_i * 2 * step + step - 1;
			int last_id = mid_id + step;
			int sumL = local_prefsum[mid_id];
			int toAdd = local_prefsum[last_id];
			local_prefsum[mid_id] = toAdd;
			local_prefsum[last_id] += sumL;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	global_prefsum[global_i] = local_prefsum[local_i];
	global_prefsum[global_i + GROUP_SIZE] = local_prefsum[local_i + GROUP_SIZE];
}

__kernel void max_val(__global int *global_as, __global int *global_max_id, int n, int step)
{
	int max_id = n / (2 * step);
	int start_id = get_global_id(0);
	if (start_id >= max_id) return;

	start_id = start_id * 2 * step;
	int mid_id = start_id + step;

	if (step == 1) {
		global_max_id[start_id] = start_id;
		global_max_id[mid_id] = mid_id;
	}
	if (global_as[mid_id] > global_as[start_id]) {
		global_as[start_id] = global_as[mid_id];
		global_max_id[start_id] = global_max_id[mid_id];
	}
}