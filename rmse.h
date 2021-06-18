#ifndef RMSE_H
#define RMSE_H
#define BLOCK_SIZE 512

__device__ void warpReduce(volatile float* sdata, unsigned int tid) {
	if (BLOCK_SIZE >= 64) {
		sdata[tid] += sdata[tid + 32];
	}
	if (BLOCK_SIZE >= 32) {
		sdata[tid] += sdata[tid + 16];
	}
	if (BLOCK_SIZE >= 16) {
		sdata[tid] += sdata[tid + 8];
	}
	if (BLOCK_SIZE >= 8) {
		sdata[tid] += sdata[tid + 4];
	}
	if (BLOCK_SIZE >= 4) {
		sdata[tid] += sdata[tid + 2];
	}
	if (BLOCK_SIZE >= 2) {
		sdata[tid] += sdata[tid + 1];
	}
}

__global__ void get_test_rmse_coalesced_k128(
    Node* node_arr,
	float* row_P,
	float* row_Q,
    float* group_err,
    unsigned int iter_num,
    unsigned int seg_size,
    unsigned int nonzero_num,
    unsigned int K
){
    __shared__ volatile float inner_prod_arr[BLOCK_SIZE];
    __shared__ volatile float temp_arr[BLOCK_SIZE];

    int lane_id = threadIdx.x % 32;
    int local_wid = threadIdx.x / 32;
    int local_id = threadIdx.x; 
    int group_id = blockIdx.x;
    int w_start_local_idx = local_wid * seg_size;

    for (int ite = 0; ite < iter_num; ite++){
        int b_start_idx = group_id * BLOCK_SIZE + ite * BLOCK_SIZE * gridDim.x;
        int w_start_idx = b_start_idx + w_start_local_idx;
       
        for (int si = 0; si < seg_size; si++){
            int idx = w_start_idx + si;
            
            if (idx >= nonzero_num){
                temp_arr[w_start_local_idx + si] =  0;
                continue;
            }

            float r = node_arr[idx].r;
            unsigned int u = node_arr[idx].u;
            unsigned int i = node_arr[idx].i;

            int base_p = u*K;
            int base_q = i*K;

            float tmp_p1 = row_P[base_p + lane_id];
            float tmp_q1 = row_Q[base_q + lane_id];

            float tmp_p2 = row_P[base_p + lane_id + 32];
            float tmp_q2 = row_Q[base_q + lane_id + 32];

            float tmp_p3 = row_P[base_p + lane_id + 64];
            float tmp_q3 = row_Q[base_q + lane_id + 64];

            float tmp_p4 = row_P[base_p + lane_id + 96];
            float tmp_q4 = row_Q[base_q + lane_id + 96];

            inner_prod_arr[local_id] = tmp_p1 * tmp_q1 + tmp_p2 * tmp_q2 + tmp_p3 * tmp_q3 + tmp_p4 * tmp_q4;

            if (lane_id < 16) inner_prod_arr[local_id] += inner_prod_arr[local_id + 16];
            if (lane_id < 8) inner_prod_arr[local_id] += inner_prod_arr[local_id + 8];
            if (lane_id < 4) inner_prod_arr[local_id] += inner_prod_arr[local_id + 4];
            if (lane_id < 2) inner_prod_arr[local_id] += inner_prod_arr[local_id + 2];
            if (lane_id < 1) inner_prod_arr[local_id] += inner_prod_arr[local_id + 1];

            if (lane_id == 0)
                temp_arr[w_start_local_idx + si] =  pow(r - inner_prod_arr[local_wid * 32], 2);
        }

        __syncthreads();

        if (BLOCK_SIZE >= 512) { if (local_id < 256) { temp_arr[local_id] += temp_arr[local_id + 256]; __syncthreads(); } }
        if (BLOCK_SIZE >= 256) { if (local_id < 128) { temp_arr[local_id] += temp_arr[local_id + 128]; __syncthreads(); } }
        if (BLOCK_SIZE >= 128) { if (local_id < 64) { temp_arr[local_id] += temp_arr[local_id + 64]; __syncthreads(); } }

        if (local_id < 32) warpReduce(temp_arr, local_id);
        
        if (local_id == 0) group_err[ite * gridDim.x + group_id] = temp_arr[0];    
    }
}

__global__ void get_test_rmse_coalesced_k64(
    Node* node_arr,
	float* row_P,
	float* row_Q,
    float* group_err,
    unsigned int iter_num,
    unsigned int seg_size,
    unsigned int nonzero_num,
    unsigned int K
){
    __shared__ volatile float inner_prod_arr[BLOCK_SIZE];
    __shared__ volatile float temp_arr[BLOCK_SIZE];

    int lane_id = threadIdx.x % 32;
    int local_wid = threadIdx.x / 32;
    int local_id = threadIdx.x; 
    int group_id = blockIdx.x;
    int w_start_local_idx = local_wid * seg_size;

    for (int ite = 0; ite < iter_num; ite++){
        int b_start_idx = group_id * BLOCK_SIZE + ite * BLOCK_SIZE * gridDim.x;
        int w_start_idx = b_start_idx + w_start_local_idx;
       
        for (int si = 0; si < seg_size; si++){
            int idx = w_start_idx + si;
            
            if (idx >= nonzero_num){
                temp_arr[w_start_local_idx + si] =  0;
                continue;
            }

            float r = node_arr[idx].r;
            unsigned int u = node_arr[idx].u;
            unsigned int i = node_arr[idx].i;

            int base_p = u*K;
            int base_q = i*K;

            float tmp_p1 = row_P[base_p + lane_id];
            float tmp_q1 = row_Q[base_q + lane_id];

            float tmp_p2 = row_P[base_p + lane_id + 32];
            float tmp_q2 = row_Q[base_q + lane_id + 32];

            inner_prod_arr[local_id] = tmp_p1 * tmp_q1 + tmp_p2 * tmp_q2;

            if (lane_id < 16) inner_prod_arr[local_id] += inner_prod_arr[local_id + 16];
            if (lane_id < 8) inner_prod_arr[local_id] += inner_prod_arr[local_id + 8];
            if (lane_id < 4) inner_prod_arr[local_id] += inner_prod_arr[local_id + 4];
            if (lane_id < 2) inner_prod_arr[local_id] += inner_prod_arr[local_id + 2];
            if (lane_id < 1) inner_prod_arr[local_id] += inner_prod_arr[local_id + 1];

            if (lane_id == 0)
                temp_arr[w_start_local_idx + si] =  pow(r - inner_prod_arr[local_wid * 32], 2);
        }

        __syncthreads();

        if (BLOCK_SIZE >= 512) { if (local_id < 256) { temp_arr[local_id] += temp_arr[local_id + 256]; __syncthreads(); } }
        if (BLOCK_SIZE >= 256) { if (local_id < 128) { temp_arr[local_id] += temp_arr[local_id + 128]; __syncthreads(); } }
        if (BLOCK_SIZE >= 128) { if (local_id < 64) { temp_arr[local_id] += temp_arr[local_id + 64]; __syncthreads(); } }

        if (local_id < 32) warpReduce(temp_arr, local_id);
        
        if (local_id == 0) group_err[ite * gridDim.x + group_id] = temp_arr[0];    
    }
}

float gpu_test_rmse(Mf_info* mf_info, SGD* sgd_info, Node* d_test_COO, float * d_group_error, unsigned int error_kernel_work_groups ,unsigned int iter_num, unsigned int seg_size, unsigned int group_error_size){

    cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);

    if (mf_info->params.k == 128) get_test_rmse_coalesced_k128<<<error_kernel_work_groups, 512>>>(d_test_COO, sgd_info->d_p, sgd_info->d_q, d_group_error, iter_num, seg_size, mf_info->test_n, mf_info->params.k);
    else if (mf_info->params.k == 64) get_test_rmse_coalesced_k64<<<error_kernel_work_groups, 512>>>(d_test_COO, sgd_info->d_p, sgd_info->d_q, d_group_error, iter_num, seg_size, mf_info->test_n, mf_info->params.k);
    
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    float* e_group = new float[group_error_size];
    cudaMemcpy(e_group, d_group_error, sizeof(float) * group_error_size, cudaMemcpyDeviceToHost);
    gpuErr(cudaPeekAtLastError());

    float sum = 0;
    for (int i = 0 ; i < group_error_size; i++){
        sum += e_group[i];
    }

    return sqrt(sum/(double)mf_info->test_n);
}

#endif