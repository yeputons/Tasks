__kernel void matrix_multiplication(__global float *as, __global float *bs, __global float *cs, int m, int n, int k)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    if (j >= m || i >= n) return;

    float result = 0;
    for (int p = 0; p < k; p++) {
        result += as[j * k + p] * bs[p * n + i];
    }
    cs[j * n + i] = result;
}