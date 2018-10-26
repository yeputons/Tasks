#define BITS_PER_PASS 4
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

__kernel void radixFwd(__global int* prefsums, int size, int sumBlock) {
    int i = 2 * sumBlock * (get_global_id(0) + 1) - 1;
    if (i >= size) return;

    prefsums[i] += prefsums[i - sumBlock];
}


__kernel void radixMid(__global int* prefsums, int size) {
    prefsums[size - 1] = 0;
}

__kernel void radixBwd(__global int* prefsums, int size, int sumBlock) {
    int i = 2 * sumBlock * (get_global_id(0) + 1) - 1;

    int beforeMe, leftChild;
    if (i < size) {
        beforeMe = prefsums[i];
        leftChild = prefsums[i - sumBlock];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (i < size) {
        prefsums[i - sumBlock] = beforeMe;
        prefsums[i] = beforeMe + leftChild;
    }
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
        as_next[pos] = as[i];
    }
}
