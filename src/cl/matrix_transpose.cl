__kernel void matrix_transpose(__global float *global_as, __global float *global_bs, int h, int w)
{
	int global_y = get_global_id(0);
	int global_x = get_global_id(1);
	if (global_y >= h || global_x >= w) return;

	global_bs[global_x * h + global_y] = global_as[global_y * w + global_x];
}
