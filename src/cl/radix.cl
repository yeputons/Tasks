#define WORK_GROUP_SIZE 128
#define BITS_PER_PASS 4
#define BITS_MSK ((1 << BITS_PER_PASS) - 1)
#define LSIZE (2 * WORK_GROUP_SIZE)

__kernel void radixInit(__global unsigned int* as, __global int* prefsums, int n, int bit) {
    int i = get_global_id(0);
    if (i >= n) return;

    for (int msk = 0; msk <= BITS_MSK; msk++) {
        prefsums[n * msk + i] = 0;
    }
    int val = (as[i] >> bit) & BITS_MSK;
    prefsums[n * val + i] = 1;
}

__kernel void radixLocalFwd(__global int* global_prefsums) {
    int local_i = get_local_id(0);
    int global_i = 2 * (get_global_id(0) - local_i) + local_i;

    __local int prefsums[LSIZE];
    prefsums[local_i] = global_prefsums[global_i];
    prefsums[local_i + WORK_GROUP_SIZE] = global_prefsums[global_i + WORK_GROUP_SIZE];

    for (int sumBlock = 1; sumBlock < LSIZE; sumBlock *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int pos = 2 * sumBlock * (local_i + 1) - 1;
        if (pos < LSIZE) {
            prefsums[pos] += prefsums[pos - sumBlock];
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    global_prefsums[global_i] = prefsums[local_i];
    global_prefsums[global_i + WORK_GROUP_SIZE] = prefsums[local_i + WORK_GROUP_SIZE];
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

__kernel void radixLocalBwd(__global int* global_prefsums) {
    int local_i = get_local_id(0);
    int global_i = 2 * (get_global_id(0) - local_i) + local_i;

    __local int prefsums[LSIZE];
    prefsums[local_i] = global_prefsums[global_i];
    prefsums[local_i + WORK_GROUP_SIZE] = global_prefsums[global_i + WORK_GROUP_SIZE];

    for (int sumBlock = LSIZE / 2; sumBlock >= 1; sumBlock /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int pos = 2 * sumBlock * (local_i + 1) - 1;

        int beforeMe, leftChild;
        if (pos < LSIZE) {
            beforeMe = prefsums[pos];
            leftChild = prefsums[pos - sumBlock];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (pos < LSIZE) {
            prefsums[pos - sumBlock] = beforeMe;
            prefsums[pos] = beforeMe + leftChild;
        }
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    global_prefsums[global_i] = prefsums[local_i];
    global_prefsums[global_i + WORK_GROUP_SIZE] = prefsums[local_i + WORK_GROUP_SIZE];
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
