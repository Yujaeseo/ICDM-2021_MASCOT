__forceinline__ __device__ float reduce_min_input(float input){
    float tmp_min = input;
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 16));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 8));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 4));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 2));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 1));
    tmp_min = __shfl_sync(0xffffffff,tmp_min,0);

    return tmp_min;
}

__forceinline__ __device__ float reduce_max_input(float input){
    float tmp_max = input;
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 16));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 8));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 4));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 2));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 1));
    tmp_max = __shfl_sync(0xffffffff,tmp_max,0);

    return tmp_max;
}