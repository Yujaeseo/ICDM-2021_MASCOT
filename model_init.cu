#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include "common_struct.h"
#include "common.h"
#include "model_init.h"
#include <iostream>
using namespace std;

__global__ void init_rand_state(curandState*state, int size)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < size)curand_init(clock() + tid, tid, 0, &state[tid]);
    // if(tid < size)curand_init(1 + tid, tid, 0, &state[tid]);
    // if(tid < size)curand_init(1, tid, 0, &state[tid]);

}

__global__ void init_rand_feature_single(curandState*state, unsigned int state_size, float* array , unsigned int dim, unsigned int k)
{
    unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int number_threads = gridDim.x*blockDim.x;
    if (state_size > tid){
        for (unsigned int i = tid; i < dim * k; i+= state_size){
            array[i] = (float)((curand_normal_double(&state[tid])* 0.01)) ;
            // array[i] = (float)((0.05)) ;
        }
    }
}

__global__ void cpyparams2grouped_params(float* original_params, __half** converted_params, Index_info_node* index_info, unsigned int k, unsigned int n){
    unsigned int g_wid = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    unsigned int num_w = gridDim.x*blockDim.x/32;
    unsigned int lane_id = threadIdx.x%32;

    for (; g_wid < n; g_wid += num_w){
        unsigned int group_idx = index_info[g_wid].g;
        unsigned int base_idx= index_info[g_wid].v * k;
        
        ((__half*)converted_params[group_idx])[base_idx + lane_id] = __float2half_rn(original_params[g_wid * k + lane_id]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 32] = __float2half_rn(original_params[g_wid * k + lane_id + 32]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 64] = __float2half_rn(original_params[g_wid * k + lane_id + 64]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 96] = __float2half_rn(original_params[g_wid * k + lane_id + 96]); 
    }
}

__global__ void cpyparams2grouped_params_fp32_version(float* original_params, float** converted_params, Index_info_node* index_info, unsigned int k, unsigned int n){
    unsigned int g_wid = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    unsigned int num_w = gridDim.x*blockDim.x/32;
    unsigned int lane_id = threadIdx.x%32;

    for (; g_wid < n; g_wid += num_w){
        unsigned int group_idx = index_info[g_wid].g;
        unsigned int base_idx= index_info[g_wid].v * k;
        
        ((float*)converted_params[group_idx])[base_idx + lane_id] = original_params[g_wid * k + lane_id]; 
        ((float*)converted_params[group_idx])[base_idx + lane_id + 32] = original_params[g_wid * k + lane_id + 32]; 
        ((float*)converted_params[group_idx])[base_idx + lane_id + 64] = original_params[g_wid * k + lane_id + 64]; 
        ((float*)converted_params[group_idx])[base_idx + lane_id + 96] = original_params[g_wid * k + lane_id + 96]; 
    }
}


__global__ void cpyparams2grouped_params_for_comparison_indexing(float* original_params, __half** converted_params, unsigned int *group_end_idx, unsigned int *entity2group, unsigned int *entity2sorted_idx, unsigned int k, unsigned int n, unsigned int group_num){
    
    extern __shared__ unsigned int end_idx_s[];

    // if (threadIdx.x < group_num){ end_idx_s[threadIdx.x+1] = group_end_idx[threadIdx.x];}
    for (int i = threadIdx.x; i < group_num; i+= blockDim.x){
        end_idx_s[i+1] = group_end_idx[i];
    }

    if (threadIdx.x == 0){
        end_idx_s[0] = -1;
    }
    __syncthreads();

    unsigned int g_wid = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    unsigned int num_w = gridDim.x*blockDim.x/32;
    unsigned int lane_id = threadIdx.x%32;

    for (; g_wid < n; g_wid += num_w){
        unsigned int group_idx = entity2group[g_wid];
        unsigned int base_idx = (entity2sorted_idx[g_wid]- (end_idx_s[group_idx] + 1)) * k;
        
        ((__half*)converted_params[group_idx])[base_idx + lane_id] = __float2half_rn(original_params[g_wid * k + lane_id]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 32] = __float2half_rn(original_params[g_wid * k + lane_id + 32]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 64] = __float2half_rn(original_params[g_wid * k + lane_id + 64]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 96] = __float2half_rn(original_params[g_wid * k + lane_id + 96]); 
    }
}

__global__ void cpyparams2grouped_params_for_comparison_indexing_k64(float* original_params, __half** converted_params, unsigned int *group_end_idx, unsigned int *entity2group, unsigned int *entity2sorted_idx, unsigned int k, unsigned int n, unsigned int group_num){
    
    extern __shared__ unsigned int end_idx_s[];

    // if (threadIdx.x < group_num){ end_idx_s[threadIdx.x+1] = group_end_idx[threadIdx.x];}
    for (int i = threadIdx.x; i < group_num; i+= blockDim.x){
        end_idx_s[i+1] = group_end_idx[i];
    }

    if (threadIdx.x == 0){
        end_idx_s[0] = -1;
    }
    __syncthreads();

    unsigned int g_wid = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    unsigned int num_w = gridDim.x*blockDim.x/32;
    unsigned int lane_id = threadIdx.x%32;

    for (; g_wid < n; g_wid += num_w){
        unsigned int group_idx = entity2group[g_wid];
        unsigned int base_idx = (entity2sorted_idx[g_wid]- (end_idx_s[group_idx] + 1)) * k;
        
        ((__half*)converted_params[group_idx])[base_idx + lane_id] = __float2half_rn(original_params[g_wid * k + lane_id]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 32] = __float2half_rn(original_params[g_wid * k + lane_id + 32]); 
    }
}

__global__ void cpyparams2grouped_params_for_division_indexing(float* original_params, __half** converted_params, unsigned int* entity2sorted_idx, unsigned int k, unsigned int n, unsigned int group_size){
    
    unsigned int g_wid = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    unsigned int num_w = gridDim.x*blockDim.x/32;
    unsigned int lane_id = threadIdx.x%32;

    for (; g_wid < n; g_wid += num_w){
        unsigned int converted_idx = entity2sorted_idx[g_wid];
        unsigned int group_idx = (converted_idx / group_size);
        unsigned int base_idx= (converted_idx % group_size) * k;
        
        ((__half*)converted_params[group_idx])[base_idx + lane_id] = __float2half_rn(original_params[g_wid * k + lane_id]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 32] = __float2half_rn(original_params[g_wid * k + lane_id + 32]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 64] = __float2half_rn(original_params[g_wid * k + lane_id + 64]); 
        ((__half*)converted_params[group_idx])[base_idx + lane_id + 96] = __float2half_rn(original_params[g_wid * k + lane_id + 96]); 
    }
}

__global__ void cpyparams2grouped_params_for_division_indexing_fp32_version(float* original_params, float** converted_params, unsigned int* entity2sorted_idx, unsigned int k, unsigned int n, unsigned int group_size){
    
    unsigned int g_wid = (blockIdx.x*blockDim.x + threadIdx.x)/32;
    unsigned int num_w = gridDim.x*blockDim.x/32;
    unsigned int lane_id = threadIdx.x%32;

    for (; g_wid < n; g_wid += num_w){
        unsigned int converted_idx = entity2sorted_idx[g_wid];
        unsigned int group_idx = (converted_idx / group_size);
        unsigned int base_idx= (converted_idx % group_size) * k;
        
        ((float*)converted_params[group_idx])[base_idx + lane_id] = __float2half_rn(original_params[g_wid * k + lane_id]); 
        ((float*)converted_params[group_idx])[base_idx + lane_id + 32] = __float2half_rn(original_params[g_wid * k + lane_id + 32]); 
        ((float*)converted_params[group_idx])[base_idx + lane_id + 64] = __float2half_rn(original_params[g_wid * k + lane_id + 64]); 
        ((float*)converted_params[group_idx])[base_idx + lane_id + 96] = __float2half_rn(original_params[g_wid * k + lane_id + 96]); 
    }
}

__global__ void transform_half2float(float *gpu_float_feature, half *gpu_half_feature, unsigned int vec_size)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int number_threads = gridDim.x*blockDim.x;

    for (unsigned int i = tid; i < vec_size; i += number_threads){
        if (i < vec_size)
            gpu_float_feature[i] = __half2float(gpu_half_feature[i]);
    }

}
__global__ void transform_float2half(__half* half_feature, float *gpu_float_feature, unsigned int vec_size){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int number_threads = gridDim.x*blockDim.x;

    for (unsigned int i = tid; i < vec_size; i += number_threads){
        if (i < vec_size){
            half_feature[i] = __float2half_rn(gpu_float_feature[i]);
        }
    }
}

__global__ void mem_cpy_fp16tofp32(float* out, __half* in, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (; i < n; i += gridDim.x * blockDim.x)
        out[i] = __half2float(in[i]);
}

void init_features_single(float *feature_vec, unsigned int dim, unsigned int k){
    float* gpu_vec;
    cudaMalloc(&gpu_vec, sizeof(float) * dim * k);

    unsigned int workers = 3200;
    curandState* d_state;
    int state_size = workers * 32;
    
    cudaMalloc(&d_state, sizeof(curandState) * state_size);
    init_rand_state<<<(state_size + 255)/256, 256>>>(d_state, state_size);
    cudaDeviceSynchronize();

    gpuErr(cudaPeekAtLastError());

    init_rand_feature_single<<<(state_size + 255)/256, 256>>>(d_state, state_size, gpu_vec, dim, k);
    cudaDeviceSynchronize();

    cudaMemcpy(feature_vec, gpu_vec, sizeof(float)*dim*k, cudaMemcpyDeviceToHost);
    gpuErr(cudaPeekAtLastError());
    
    cudaFree(d_state);
    cudaFree(gpu_vec);
}

void init_features_single_on_device(float *d_feature_vec, unsigned int dim, unsigned int k){
    cudaMalloc(&d_feature_vec, sizeof(float) * dim * k);

    unsigned int workers = 3200;
    curandState* d_state;
    int state_size = workers * 32;
    
    cudaMalloc(&d_state, sizeof(curandState) * state_size);
    init_rand_state<<<(state_size + 255)/256, 256>>>(d_state, state_size);
    cudaDeviceSynchronize();

    gpuErr(cudaPeekAtLastError());

    init_rand_feature_single<<<(state_size + 255)/256, 256>>>(d_state, state_size, d_feature_vec, dim, k);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
}

void init_model_single(Mf_info *mf_info, SGD *sgd_info){
    cudaMallocHost(&sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMallocHost(&sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k);
    gpuErr(cudaPeekAtLastError());

    init_features_single(sgd_info->p, mf_info->max_user, mf_info->params.k);
    init_features_single(sgd_info->q, mf_info->max_item, mf_info->params.k);

    cudaMalloc(&sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k);

    cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);
}

void init_model_half(Mf_info *mf_info, SGD *sgd_info){
    cudaMallocHost(&sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMallocHost(&sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k);
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->half_p, sizeof(short) * mf_info->max_user * mf_info->params.k);
    cudaMallocHost(&sgd_info->half_q, sizeof(short) * mf_info->max_item * mf_info->params.k);
    gpuErr(cudaPeekAtLastError());

    init_features_single(sgd_info->p, mf_info->max_user, mf_info->params.k);
    init_features_single(sgd_info->q, mf_info->max_item, mf_info->params.k);

    conversion_features_half(sgd_info->half_p, sgd_info->p ,mf_info->max_user, mf_info->params.k);
    conversion_features_half(sgd_info->half_q, sgd_info->q ,mf_info->max_item, mf_info->params.k);

    cudaMalloc(&sgd_info->d_half_p, sizeof(short) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&sgd_info->d_half_q, sizeof(short) * mf_info->max_item * mf_info->params.k);

    cudaMemcpy(sgd_info->d_half_p, sgd_info->half_p, sizeof(short) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_half_q, sgd_info->half_q, sizeof(short) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);
}

void cpy2grouped_parameters(Mf_info *mf_info, SGD *sgd_info){
    //! HOST에서 COPY하지 않고 GPU에서 바로 실행하도록 코드 수정하기
    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    for (int i = 0; i < mf_info->user_group_num; i++){
        unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k; 
        sgd_info->user_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
    }
    
    for (int i = 0; i < mf_info->item_group_num; i++){
        unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
        sgd_info->item_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
    }

    for (int i = 0; i < mf_info->max_user; i++){
        unsigned int user_group = mf_info->user_index_info[i].g;
        unsigned int index = mf_info->user_index_info[i].v;

        for (int k = 0; k < mf_info->params.k; k++){
            ((__half*)sgd_info->user_group_ptr[user_group])[index * mf_info->params.k + k] = __float2half_rn(sgd_info->p[i * mf_info->params.k + k]);
        }
    }

    for (int i = 0; i < mf_info->max_item; i++){
        unsigned int item_group = mf_info->item_index_info[i].g;
        unsigned int index = mf_info->item_index_info[i].v;

        for (int k = 0; k < mf_info->params.k; k++){
            ((__half*)sgd_info->item_group_ptr[item_group])[index * mf_info->params.k + k] = __float2half_rn(sgd_info->q[i * mf_info->params.k + k]);
        }
    }

    //* TRANSFER GROUPED PARAMETER TO GPU
    for (int i = 0; i < mf_info->user_group_num; i++){
        unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size);
        cudaMemcpy(sgd_info->user_group_d_ptr[i], sgd_info->user_group_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size);
        cudaMemcpy(sgd_info->item_group_d_ptr[i], sgd_info->item_group_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyHostToDevice);
    }
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void**) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Copy parameters             : " << cpy2grouped_parameters_exec_time << endl; 
}

void cpy2grouped_parameters_gpu(Mf_info *mf_info, SGD *sgd_info){
    // double cpy2grouped_parameters_exec_time = 0;
    // double alloc_device_host_parameter_exec_time = 0;

    // std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    for (int i = 0; i < mf_info->user_group_num; i++){
        unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size);
        // sgd_info->user_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->user_group_ptr[i], sizeof(__half)*group_params_size);
        // sgd_info->user_group_ptr[i] = malloc(sizeof(__half) * group_params_size);
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size);
        // sgd_info->item_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->item_group_ptr[i], sizeof(__half)*group_params_size);
        // sgd_info->item_group_ptr[i] = malloc(sizeof(__half) * group_params_size);
    }

    // alloc_device_host_parameter_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    // cpy2grouped_parameters_exec_time += alloc_device_host_parameter_exec_time;
    
    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void**) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    
    unsigned int w_num = 2048;
    unsigned int block_size = 256;
    // cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    
    cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_p, (__half**)sgd_info->d_user_group_ptr, mf_info->d_user_index_info, mf_info->params.k, mf_info->max_user);
    cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_q, (__half**)sgd_info->d_item_group_ptr, mf_info->d_item_index_info, mf_info->params.k, mf_info->max_item);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    // cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    
    // cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    // cout << "Alloc parameter host & device time : " << alloc_device_host_parameter_exec_time << endl;
    // cout << "Copy parameters                    : " << cpy2grouped_parameters_exec_time << endl; 
}

void cpy2grouped_parameters_gpu_float_version(Mf_info *mf_info, SGD *sgd_info){
    double cpy2grouped_parameters_exec_time = 0;
    double alloc_device_host_parameter_exec_time = 0;

    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    for (int i = 0; i < mf_info->user_group_num; i++){
        unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size);
        // sgd_info->user_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->user_group_ptr[i], sizeof(float)*group_params_size);
        // sgd_info->user_group_ptr[i] = malloc(sizeof(__half) * group_params_size);
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size);
        // sgd_info->item_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->item_group_ptr[i], sizeof(float)*group_params_size);
        // sgd_info->item_group_ptr[i] = malloc(sizeof(__half) * group_params_size);
    }

    alloc_device_host_parameter_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    cpy2grouped_parameters_exec_time += alloc_device_host_parameter_exec_time;
    
    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void**) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    
    unsigned int w_num = 2048;
    unsigned int block_size = 256;
    cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    
    cpyparams2grouped_params_fp32_version<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_p, (float**)sgd_info->d_user_group_ptr, mf_info->d_user_index_info, mf_info->params.k, mf_info->max_user);
    cpyparams2grouped_params_fp32_version<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_q, (float**)sgd_info->d_item_group_ptr, mf_info->d_item_index_info, mf_info->params.k, mf_info->max_item);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    
    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Alloc parameter host & device time : " << alloc_device_host_parameter_exec_time << endl;
    cout << "Copy parameters                    : " << cpy2grouped_parameters_exec_time << endl; 
}

void cpy2grouped_parameters_gpu_for_comparison_indexing(Mf_info *mf_info, SGD *sgd_info){

    double cpy2grouped_parameters_exec_time = 0;
    
    unsigned int* d_user_group_idx;
    unsigned int* d_item_group_idx;

    cudaMalloc(&d_user_group_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&d_item_group_idx, sizeof(unsigned int) * mf_info->max_item);

    cudaMemcpy(d_user_group_idx, mf_info->user_group_idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyHostToDevice);
    cudaMemcpy(d_item_group_idx, mf_info->item_group_idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyHostToDevice);

    for (int i = 0; i < mf_info->user_group_num; i++){
        unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size);
        // sgd_info->user_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->user_group_ptr[i], sizeof(__half)*group_params_size);
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size);
        // sgd_info->item_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->item_group_ptr[i], sizeof(__half)*group_params_size);
    }

    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void**) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    unsigned int w_num = 2048;
    unsigned int block_size = 256;
    // std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    if (mf_info->params.k == 128){
        cpyparams2grouped_params_for_comparison_indexing<<<(w_num)/(block_size/32), block_size, sizeof(unsigned int)*(mf_info->user_group_num+1)>>>(sgd_info->d_p, (__half**)sgd_info->d_user_group_ptr, mf_info->d_user_group_end_idx, d_user_group_idx, mf_info->d_user2sorted_idx, mf_info->params.k, mf_info->max_user, mf_info->user_group_num);
        cpyparams2grouped_params_for_comparison_indexing<<<(w_num)/(block_size/32), block_size, sizeof(unsigned int)*(mf_info->item_group_num+1)>>>(sgd_info->d_q, (__half**)sgd_info->d_item_group_ptr, mf_info->d_item_group_end_idx, d_item_group_idx, mf_info->d_item2sorted_idx, mf_info->params.k, mf_info->max_item, mf_info->item_group_num);
    }else if (mf_info->params.k == 64){
        cpyparams2grouped_params_for_comparison_indexing_k64<<<(w_num)/(block_size/32), block_size, sizeof(unsigned int)*(mf_info->user_group_num+1)>>>(sgd_info->d_p, (__half**)sgd_info->d_user_group_ptr, mf_info->d_user_group_end_idx, d_user_group_idx, mf_info->d_user2sorted_idx, mf_info->params.k, mf_info->max_user, mf_info->user_group_num);
        cpyparams2grouped_params_for_comparison_indexing_k64<<<(w_num)/(block_size/32), block_size, sizeof(unsigned int)*(mf_info->item_group_num+1)>>>(sgd_info->d_q, (__half**)sgd_info->d_item_group_ptr, mf_info->d_item_group_end_idx, d_item_group_idx, mf_info->d_item2sorted_idx, mf_info->params.k, mf_info->max_item, mf_info->item_group_num);
    }

    // cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_p, (__half**)sgd_info->d_user_group_ptr, mf_info->d_user_index_info, mf_info->params.k, mf_info->max_user);
    // cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_q, (__half**)sgd_info->d_item_group_ptr, mf_info->d_item_index_info, mf_info->params.k, mf_info->max_item);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    // cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    
    // cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    // cout << "Copy parameters             : " << cpy2grouped_parameters_exec_time << endl;
}

void cpy2grouped_parameters_gpu_for_division_indexing(Mf_info *mf_info, SGD *sgd_info){

    double cpy2grouped_parameters_exec_time = 0;

    for (int i = 0; i < mf_info->user_group_num; i++){
        unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size);
        // sgd_info->user_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->user_group_ptr[i], sizeof(__half)*group_params_size);
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size);
        // sgd_info->item_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->item_group_ptr[i], sizeof(__half)*group_params_size);
    }

    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void**) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    
    unsigned int w_num = 2048;
    unsigned int block_size = 256;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    cpyparams2grouped_params_for_division_indexing<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_p, (__half**)sgd_info->d_user_group_ptr, mf_info->d_user2sorted_idx, mf_info->params.k, mf_info->max_user, (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num));
    cpyparams2grouped_params_for_division_indexing<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_q, (__half**)sgd_info->d_item_group_ptr, mf_info->d_item2sorted_idx, mf_info->params.k, mf_info->max_item, (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num));

    // cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_p, (__half**)sgd_info->d_user_group_ptr, mf_info->d_user_index_info, mf_info->params.k, mf_info->max_user);
    // cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_q, (__half**)sgd_info->d_item_group_ptr, mf_info->d_item_index_info, mf_info->params.k, mf_info->max_item);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    
    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Copy parameters             : " << cpy2grouped_parameters_exec_time << endl; 
}

void cpy2grouped_parameters_gpu_for_division_indexing_float_version(Mf_info *mf_info, SGD *sgd_info){

    double cpy2grouped_parameters_exec_time = 0;

    for (int i = 0; i < mf_info->user_group_num; i++){
        unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size);
        // sgd_info->user_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->user_group_ptr[i], sizeof(float)*group_params_size);
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
        cudaMalloc((void**)&sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size);
        // sgd_info->item_group_ptr[i] = (__half*)malloc(sizeof(__half)*group_params_size);
        cudaMallocHost(&sgd_info->item_group_ptr[i], sizeof(float)*group_params_size);
    }

    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void**) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    
    unsigned int w_num = 2048;
    unsigned int block_size = 256;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    cpyparams2grouped_params_for_division_indexing_fp32_version<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_p, (float**)sgd_info->d_user_group_ptr, mf_info->d_user2sorted_idx, mf_info->params.k, mf_info->max_user, (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num));
    cpyparams2grouped_params_for_division_indexing_fp32_version<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_q, (float**)sgd_info->d_item_group_ptr, mf_info->d_item2sorted_idx, mf_info->params.k, mf_info->max_item, (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num));

    // cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_p, (__half**)sgd_info->d_user_group_ptr, mf_info->d_user_index_info, mf_info->params.k, mf_info->max_user);
    // cpyparams2grouped_params<<<(w_num)/(block_size/32), block_size>>>(sgd_info->d_q, (__half**)sgd_info->d_item_group_ptr, mf_info->d_item_index_info, mf_info->params.k, mf_info->max_item);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    
    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Copy parameters             : " << cpy2grouped_parameters_exec_time << endl; 
}

void transform_feature_vector_half2float(short *half_feature, float *float_feature, unsigned int dim, unsigned int k){
    float *gpu_float_feature;
    half *gpu_half_feature;

    cudaMalloc(&gpu_half_feature, sizeof(half)*dim*k);
    cudaMalloc(&gpu_float_feature, sizeof(float)*dim*k);
    gpuErr(cudaPeekAtLastError());

    cudaMemcpy(gpu_half_feature, half_feature, sizeof(half)*dim*k, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    transform_half2float<<<(dim*k+255)/256, 256>>>(gpu_float_feature, gpu_half_feature, dim*k);
    cudaDeviceSynchronize();

    gpuErr(cudaPeekAtLastError());
    
    cudaMemcpy(float_feature, gpu_float_feature, sizeof(float)*dim*k, cudaMemcpyDeviceToHost);
    gpuErr(cudaPeekAtLastError());
    
    cudaFree(gpu_float_feature);
    cudaFree(gpu_half_feature);
    gpuErr(cudaPeekAtLastError());
}

void conversion_features_half(short *feature_vec, float *feature_vec_from ,unsigned int dim, unsigned int k){
    __half* gpu_vec;
    float* gpu_from_vec;

    cudaMalloc(&gpu_vec, sizeof(__half) * dim * k);
    cudaMalloc(&gpu_from_vec, sizeof(float) * dim * k);
    cudaMemcpy(gpu_from_vec, feature_vec_from, sizeof(float) * dim * k, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    transform_float2half<<< (dim * k + 255) / 256, 256>>>(gpu_vec, gpu_from_vec, dim * k);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
        
    cudaMemcpy(feature_vec, gpu_vec, sizeof(__half) * dim * k, cudaMemcpyDeviceToHost);
    cudaFree(gpu_vec);
    cudaFree(gpu_from_vec);
}

void transition_params_half2float(Mf_info *mf_info, SGD *sgd_info){
    int num_groups = 10000;
    mem_cpy_fp16tofp32<<<num_groups, 512>>>(sgd_info->d_p, sgd_info->d_half_p, mf_info->params.k * mf_info->max_user);
    mem_cpy_fp16tofp32<<<num_groups, 512>>>(sgd_info->d_q, sgd_info->d_half_q, mf_info->params.k * mf_info->max_item);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
}