#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

__kernel void fill_zero(__global int *as) {
	as[get_global_id(0)] = 0;
}

__kernel void prefix_sum(__global int *global_prefsum, int n, int step) {
	int start_id = get_global_id(0) * 2 * step + step;
	int end_id = start_id + step;
	if (start_id >= n) return;

	for (int i = start_id; i < end_id; i++) {
		global_prefsum[i] += global_prefsum[start_id - 1];
	}
}

__kernel void max_val(__global int *global_as, __global int *global_max_id, int n, int step)
{
	int start_id = get_global_id(0) * 2 * step;
	int mid_id = start_id + step;
	if (start_id >= n) return;

	if (step == 1) {
		global_max_id[start_id] = start_id + 1;
		global_max_id[mid_id] = mid_id + 1;
	}
	if (global_as[mid_id] > global_as[start_id]) {
		global_as[start_id] = global_as[mid_id];
		global_max_id[start_id] = global_max_id[mid_id];
	}
}