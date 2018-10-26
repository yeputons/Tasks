#define BITS_PER_PASS 2
#define BITS_MSK ((1 << BITS_PER_PASS) - 1)

__kernel void radixInit(__global unsigned int* as, __global int* prefsums, int n, int bit) {
    int i = get_global_id(0);
    if (i >= n) return;

    for (int msk = 0; msk <= BITS_MSK; msk++) {
        prefsums[n * msk + i] = 0;
    }
    int val = (as[i] >> bit) & BITS_MSK;
    prefsums[n * val + i] = 1;
}

// Hillis-Steele algorithm, O(n log n) operations, O(log n) runs.
__kernel void radixSum(__global int* prefsums, __global int* prefsums_next, int size, int sumBlock) {
    int i = get_global_id(0);
    if (i >= size) return;

    prefsums_next[i] = prefsums[i] + prefsums[i - sumBlock];
}

__kernel void radixShuffle(__global unsigned int* as, __global unsigned int *as_next, __global int* prefsums, int n, int bit) {
    int i = get_global_id(0);
    int pos;

    if (i < n) {
        int val = (as[i] >> bit) & BITS_MSK;
        pos = prefsums[val * n + i];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i < n) {
        as_next[pos - 1] = as[i];
    }
}
