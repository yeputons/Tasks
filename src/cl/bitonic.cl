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
