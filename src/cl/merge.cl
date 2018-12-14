__kernel void merge(__global float *as, __global float *as_merged, int n, int step) {
    int global_pos = get_global_id(0);
    if (global_pos >= n) return;

    int a_start = global_pos / (2 * step) * (2 * step);
    int a_end = min(a_start + step, n);
    int b_start = a_end;
    int b_end = min(b_start + step, n);

    int in_block_pos = global_pos - a_start;

    // We took X elements from A and (in_block_pos - X) elements from B:
    // as[a_start + X - 1] <= bs[b_start + in_block_pos - (X - 1) - 1]
    // as[a_start + X] > bs[b_start + in_block_pos - X - 1]
    int L = max(in_block_pos - (b_end - b_start), 0) - 1;
    int R = min(a_end - a_start, in_block_pos);
    while (L + 1 < R) {
        int M = (L + R) / 2;
        if (as[a_start + M] <= as[b_start + in_block_pos - M - 1]) {
            L = M;
        }
        else {
            R = M;
        }
    }
    int X = R;
    int p1 = a_start + X;
    int p2 = b_start + (in_block_pos - X);

    /*printf("global_pos=%d; %d..%d; %d..%d; in_block_pos=%d; X=%d, p1=%d, p2=%d; initialLR=%d..%d", global_pos, a_start, a_end, b_start, b_end, in_block_pos, X, p1, p2,
        max(in_block_pos - (b_end - b_start), 0) - 1,
        min(a_end - a_start, in_block_pos)
        );*/
    if (p1 < a_end && (p2 >= b_end || as[p1] <= as[p2])) {
        //printf("; getA\n");
        as_merged[global_pos] = as[p1];
    }
    else {
        //printf("; getB\n");
        as_merged[global_pos] = as[p2];
    }
}