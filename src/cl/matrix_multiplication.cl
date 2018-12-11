__kernel void matrix_multiplication(__global float *as, __global float *bs, __global float *cs, int m, int n, int k)
{
    int j = get_global_id(0);
    int i = get_global_id(1);

    int local_j = get_local_id(0);
    int local_i = get_local_id(1);

    __local float local_as[TILE_SIZE][TILE_SIZE];
    __local float local_bs[TILE_SIZE][TILE_SIZE];

    float result = 0;
    for (int p0 = 0; p0 < k; p0 += TILE_SIZE) {
        if (p0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        local_as[local_j][local_i] = as[j * k + p0 + local_i];
        local_bs[local_j][local_i] = bs[(p0 + local_j) * n + i];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int p = 0; p < TILE_SIZE; p++) {
            result += local_as[local_j][p] * local_bs[p][local_i];
        }
    }
    cs[j * n + i] = result;
}