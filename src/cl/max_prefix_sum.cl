__kernel void fill_zero(__global int *as) {
	as[get_global_id(0)] = 0;
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

	if (max_id == 1) {
		global_prefsum[last_id] = 0;
	}

	int sumL = global_prefsum[mid_id];
	int toAdd = global_prefsum[last_id];
	global_prefsum[mid_id] = toAdd;
	global_prefsum[last_id] += sumL;
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