__global__ void mem_cpy_fp162fp32(float* out, __half* in, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (; i < n; i += gridDim.x * blockDim.x)
        out[i] = __half2float(in[i]);
}

void precision_switching_by_groups_grad_diversity(Mf_info* mf_info, SGD* sgd_info){
    int num_groups = 10000;
    float threshold = mf_info->params.error_threshold;
    
    for (int i = 0; i < mf_info->params.user_group_num; i++){
        if (mf_info->user_group_error[i] > threshold){
            float* d_new_group_ptr;
            __half* temp_ptr;
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            cudaMalloc(&d_new_group_ptr, sizeof(float) * group_params_size);
            mem_cpy_fp162fp32<<<num_groups, 512>>>(d_new_group_ptr, (__half*)sgd_info->user_group_d_ptr[i], group_params_size);
            cudaDeviceSynchronize();
            temp_ptr = (__half*)sgd_info->user_group_d_ptr[i];
            (sgd_info->user_group_d_ptr[i]) = d_new_group_ptr;
            cudaFree(temp_ptr);
            mf_info->user_group_prec_info[i] = (unsigned char)1;
            cudaFreeHost(sgd_info->user_group_ptr[i]);
            cudaMallocHost((&(sgd_info->user_group_ptr[i])), sizeof(float)*group_params_size);
        }
    }

    cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->params.user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->params.user_group_num, cudaMemcpyHostToDevice);
    
    for (int i = 0; i < mf_info->params.item_group_num; i++){
        if (mf_info->item_group_error[i] > threshold){
            float* d_new_group_ptr;
            __half* temp_ptr;
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k;
            cudaMalloc(&d_new_group_ptr, sizeof(float) * group_params_size);
            mem_cpy_fp162fp32<<<num_groups, 512>>>(d_new_group_ptr, (__half*)sgd_info->item_group_d_ptr[i], group_params_size);
            cudaDeviceSynchronize();
            temp_ptr = (__half*)sgd_info->item_group_d_ptr[i];
            (sgd_info->item_group_d_ptr[i]) = d_new_group_ptr;
            cudaFree(temp_ptr);
            mf_info->item_group_prec_info[i] = (unsigned char)1;  
            cudaFreeHost(sgd_info->item_group_ptr[i]);
            cudaMallocHost(&sgd_info->item_group_ptr[i], sizeof(float)*group_params_size);
        }
    }

    cudaMemcpy(mf_info->d_item_group_prec_info,  mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->params.item_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->params.item_group_num, cudaMemcpyHostToDevice);
}

