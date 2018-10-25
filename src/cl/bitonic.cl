#define WORK_GROUP_SIZE 128

void bitonicCalc(int n, int halfBlockSize, bool directPass, int i, int *a, int *b) {
    int blockId = i / halfBlockSize;
    int inBlockId = i % halfBlockSize;

    int blockSize = 2 * halfBlockSize;

    int blockStart = blockId * blockSize;
    *a = blockStart + inBlockId;
    *b = directPass
        ? blockStart + halfBlockSize + inBlockId
        : blockStart + blockSize - 1 - inBlockId;
}

__kernel void bitonic(__global float* as, int n, int halfBlockSize, bool directPass)
{
    int a, b;
    bitonicCalc(n, halfBlockSize, directPass, get_global_id(0), &a, &b);
    if (a < n && b < n) {
        if (as[a] > as[b]) {
            float x = as[a];
            as[a] = as[b];
            as[b] = x;
        }
    }
}

void bitonicLocalIteration(__local float *data, int n, int halfBlockSize, int i, bool directPass) {
    int a, b;
    bitonicCalc(n, halfBlockSize, directPass, i, &a, &b);
    if (a < n && b < n) {
        if (data[a] > data[b]) {
            float x = data[a];
            data[a] = data[b];
            data[b] = x;
        }
    }
}

__kernel void bitonicLocal(__global float *global_as, int global_n)
{
    __local float local_as[2 * WORK_GROUP_SIZE];

    int i = get_local_id(0);
    int global_block_start = 2 * (get_global_id(0) - i);
    int global_i = global_block_start + i;

    if (global_i < global_n) {
        local_as[i] = global_as[global_i];
    }
    if (global_i + WORK_GROUP_SIZE < global_n) {
        local_as[i + WORK_GROUP_SIZE] = global_as[global_i + WORK_GROUP_SIZE];
    }

    int local_n = global_n - global_block_start;
    if (local_n > 2 * WORK_GROUP_SIZE) {
        local_n = 2 * WORK_GROUP_SIZE;
    }

    for (int sortedBlockSize = 1; sortedBlockSize <= WORK_GROUP_SIZE; sortedBlockSize *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        bitonicLocalIteration(local_as, local_n, sortedBlockSize, i, false);
        for (int alignBlock = sortedBlockSize / 2; alignBlock > 0; alignBlock /= 2) {
            barrier(CLK_LOCAL_MEM_FENCE);
            bitonicLocalIteration(local_as, local_n, alignBlock, i, true);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if (global_i < global_n) {
        global_as[global_i] = local_as[i];
    }
    if (global_i + WORK_GROUP_SIZE < global_n) {
        global_as[global_i + WORK_GROUP_SIZE] = local_as[i + WORK_GROUP_SIZE];
    }
}
