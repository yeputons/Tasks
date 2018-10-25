__kernel void radixInit(__global unsigned int* as, __global int* prefsums0, __global int* prefsums1, int n, int bit) {
    int i = get_global_id(0);
    if (i >= n) return;

    int val = !!(as[i] & (1 << bit));
    prefsums0[i] = val == 0 ? 1 : 0;
    prefsums1[i] = val == 1 ? 1 : 0;
}

// Hillis-Steele algorithm, O(n log n) operations, O(log n) runs.
__kernel void radixSum(__global int* prefsums0, __global int* prefsums1, __global int* prefsums0_next, __global int* prefsums1_next, int n, int bit, int sumBlock) {
    int i = get_global_id(0);
    if (i >= n) return;

    prefsums0_next[i] = prefsums0[i] + prefsums0[i - sumBlock];
    prefsums1_next[i] = prefsums1[i] + prefsums1[i - sumBlock];
}

__kernel void radixShuffle(__global unsigned int* as, __global unsigned int *as_next, __global int* prefsums0, __global int* prefsums1, int n, int bit) {
    int i = get_global_id(0);
    int pos;

    if (i < n) {
        pos = (as[i] & (1 << bit)) == 0 ? prefsums0[i] : prefsums1[i] + prefsums0[n - 1];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i < n) {
        as_next[pos - 1] = as[i];
    }
}
