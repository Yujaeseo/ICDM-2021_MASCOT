// #include "model_init.h"
__global__ void mem_cpy_fp162fp32(float* out, __half* in, int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (; i < n; i += gridDim.x * blockDim.x)
        out[i] = __half2float(in[i]);
}

void precision_switching_by_groups(Mf_info* mf_info, SGD* sgd_info){
    // float threshold = 0.0016;
    // float threshold = 0.0011;
    // float threshold = 0.00008 * 0.1f;
    // float threshold = 0.00105;
    float threshold = mf_info->error_threshold;

    int num_groups = 10000;

    // //! Virtual error
    // for (int i = 0; i < mf_info->user_group_num; i++){
    //     mf_info->user_group_error[i] = 100;
    // }
    // for (int i = 0; i < mf_info->item_group_num; i++){
    //     mf_info->item_group_error[i] = 0;
    // }
    
    // double precision_switching_exec_time = 0;
    // std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();

    for (int i = 0; i < mf_info->user_group_num; i++){
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

    cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // gpuErr(cudaPeekAtLastError());    

    for (int i = 0; i < mf_info->item_group_num; i++){
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

    cudaMemcpy(mf_info->d_item_group_prec_info,  mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    // precision_switching_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
    // gpuErr(cudaPeekAtLastError());    

    // cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    // cout << "Precision switching         : " << precision_switching_exec_time << endl; 
}

void precision_switching_by_groups_grad_diversity(Mf_info* mf_info, SGD* sgd_info){
    // float threshold = 0.05; // ML10M
    //float threshold = 0.004; .. ML25M
    int num_groups = 10000;
    float threshold = mf_info->error_threshold;

    // //! Virtual error
    // for (int i = 0; i < mf_info->user_group_num; i++){
    //     mf_info->user_group_error[i] = 100;
    // }
    // for (int i = 0; i < mf_info->item_group_num; i++){
    //     mf_info->item_group_error[i] = 0;
    // }
    
    // double precision_switching_exec_time = 0;
    // std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
    // cout << "User group : ";
    if (mf_info->version == 15) cout << "\nChanged user groups :";
    
    for (int i = 0; i < mf_info->user_group_num; i++){
        if (mf_info->user_group_error[i] < threshold){
            if (mf_info->version == 15) cout << i + 1 << " ";
            // cout << i  << " ";
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

    cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_user_group_ptr, sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // gpuErr(cudaPeekAtLastError());    
    
    // cout << "\nItem group : ";
    if (mf_info->version == 15) cout << "\nChanged item groups :";

    for (int i = 0; i < mf_info->item_group_num; i++){
        if (mf_info->item_group_error[i] < threshold){
            if (mf_info->version == 15) cout << i + 1 << " ";

            // cout << i  << " ";
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
    if (mf_info->version == 15) cout << "\n";

    cudaMemcpy(mf_info->d_item_group_prec_info,  mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_item_group_ptr, sgd_info->item_group_d_ptr, sizeof(void**) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    // precision_switching_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
    // gpuErr(cudaPeekAtLastError());    

    // cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    // cout << "Precision switching         : " << precision_switching_exec_time << endl; 
}

