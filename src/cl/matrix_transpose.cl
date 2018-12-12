__kernel void matrix_transpose(__global float *global_as, __global float *global_bs, int h, int w)
{
    int global_y = get_global_id(0);
    int global_x = get_global_id(1);
    if (global_y >= h || global_x >= w) return;

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

#ifdef NAIVE
    global_bs[global_x * h + global_y] = global_as[global_y * w + global_x];
#else
    int global_xt = global_x - local_x + local_y;
    int global_yt = global_y - local_y + local_x;

    __local float local_mat[GROUP_SIZE][GROUP_SIZE];
#ifndef NON_COALESCED
    float val = global_as[global_yt * w + global_xt]; // Coalesced.
#else
    float val = global_as[global_y * w + global_x]; // Non-coalesced.
#endif
    local_mat[local_x][local_y] = val; // No bank conflict; local_x is constant inside one warp.
    barrier(CLK_LOCAL_MEM_FENCE);

#ifndef NON_COALESCED
    global_bs[global_x * h + global_y] = // Coalesced.
#else
    global_bs[global_xt * h + global_yt] = // Non-coalesced.
#endif
        local_mat[local_y][local_x]; // Bank conflict; local_x is constant inside one warp.
#endif
}
