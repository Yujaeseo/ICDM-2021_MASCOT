#include <iostream>
#include <string>
#include <random>
#include <curand.h>
#include <chrono>
#include <curand_kernel.h>
#include <algorithm>
#include <limits>
#include <cuda_fp16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <iomanip>
#include "common.h"
#include "common_struct.h"
#include "preprocess_utils.h"
#include "model_init.h"
#include "reduce_kernel.h"
#include "afp_sgd_kernel.h"
#include "afp_sgd_kernel_k64.h"
#include "mascot_sgd_kernel.h"
#include "mascot_sgd_kernel_k64.h"
#include "muppet_sgd_kernel.h"
#include "muppet_sgd_kernel_k64.h"
#include "mpt_sgd_kernel.h"
#include "mpt_sgd_kernel_k64.h"
#include "sgd_kernel.h"
#include "sgd_kernel_k64.h"
#include "rmse.h"
#include "precision_switching.h"

using namespace std;

extern __global__ void init_rand_state(curandState*state, int size);

void mascot_training_mf(Mf_info* mf_info, SGD* sgd_info){
    // Random shuffle rating matrix
    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    // Testset format conversion
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    // Histogram
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    // Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    // Matrix reconstruction
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    // Random state initialization
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    // Group info allocation on host
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->params.user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->params.item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->params.user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->params.item_group_num);

    // Group info allocation on device
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->params.user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->params.item_group_num);

    // Copy grouped parameter from cpu to device    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    // Learning rate scheduling 
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    // Variables initialization
    unsigned int update_vector_size = 128;
    unsigned int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    unsigned int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->params.sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    unsigned int div = mf_info->params.thread_block_size/32;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    unsigned int block_num = mf_info->params.num_workers/div;
    unsigned int start_idx = 5;
    unsigned int num_workers_other_kernels = 10000;

    float* d_e_group;
    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->params.user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->params.item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->params.user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->params.item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->params.user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->params.item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->params.user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->params.item_group_num);

    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->params.user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->params.item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->params.user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->params.item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->params.user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->params.item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->params.user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->params.item_group_num);
    
    float* initial_user_group_error = new float[mf_info->params.user_group_num];
    float* initial_item_group_error = new float[mf_info->params.item_group_num];

    for (int i = 0; i <mf_info->params.user_group_num; i++) initial_user_group_error[i] = 1.0f;
    for (int i = 0; i <mf_info->params.item_group_num; i++) initial_item_group_error[i] = 1.0f;

    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    size_t user_idx_table_cache_size = mf_info->params.user_group_num > 62 ? sizeof(unsigned int) * (mf_info->params.user_group_num - 62) : 0;
    size_t item_idx_table_cache_size = mf_info->params.item_group_num > 62 ? sizeof(unsigned int) * (mf_info->params.item_group_num - 62) : 0;
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->params.user_group_num + mf_info->params.item_group_num)) +
                              user_idx_table_cache_size +
                              item_idx_table_cache_size + 
                              (sizeof(unsigned char)*(mf_info->params.user_group_num + mf_info->params.item_group_num));

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;  

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;

        if ((e >= start_idx ) && (e % mf_info->params.interval == (start_idx % mf_info->params.interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check && mf_info->params.epoch -1 != e){
            initialize_float_array_to_val<<<num_workers_other_kernels, 512>>>(d_grad_sum_norm_p, mf_info->params.user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_workers_other_kernels, 512>>>(d_grad_sum_norm_q, mf_info->params.item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;
    
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128){
            mascot_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda,
                                                        mf_info->d_user_group_end_idx,
                                                        mf_info->d_item_group_end_idx,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_grad_sum_norm_p,
                                                        d_grad_sum_norm_q,
                                                        d_norm_sum_p,
                                                        d_norm_sum_q,
                                                        first_sample_rating_idx,
                                                        (int)mf_info->params.user_group_num,
                                                        (int)mf_info->params.item_group_num
                                                        );
        }
        else if (mf_info->params.k == 64){
            mascot_sgd_k64_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda,
                                                        mf_info->d_user_group_end_idx,
                                                        mf_info->d_item_group_end_idx,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_grad_sum_norm_p,
                                                        d_grad_sum_norm_q,
                                                        d_norm_sum_p,
                                                        d_norm_sum_q,
                                                        first_sample_rating_idx,
                                                        (int)mf_info->params.user_group_num,
                                                        (int)mf_info->params.item_group_num
                                                        );
        }

        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        gpuErr(cudaPeekAtLastError());  

        error_computation_start_time = std::chrono::system_clock::now();
        double error_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_copy_time;

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->params.user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            if (mf_info->user_group_prec_info[i] == 0) {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            }
            else {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->params.item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->params.user_group_num; g++){
            user_group_end_idx += mf_info->user_group_size[g];
            unsigned int user_idx_in_group = 0;
            if (mf_info->user_group_prec_info[g] == 0){
                for (int u = user_group_start_idx; u < user_group_end_idx; u++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[u * mf_info->params.k + k] = __half2float(((__half*)sgd_info->user_group_ptr[g])[user_idx_in_group * mf_info->params.k + k]);
                    }
                    user_idx_in_group += 1;
                }
            }else{
                for (int u = user_group_start_idx; u < user_group_end_idx; u++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[u * mf_info->params.k + k] = (((float*)sgd_info->user_group_ptr[g])[user_idx_in_group * mf_info->params.k + k]);
                    }
                    user_idx_in_group += 1;
                }
            }
            user_group_start_idx = user_group_end_idx;
        }

       item_group_start_idx = 0;
        unsigned int item_group_end_idx = 0;
        for (int g = 0; g < mf_info->params.item_group_num; g++){
            item_group_end_idx += mf_info->item_group_size[g];
            unsigned int item_idx_in_group = 0;
            if (mf_info->item_group_prec_info[g] == 0){
                for (int i = item_group_start_idx; i < item_group_end_idx; i++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i * mf_info->params.k + k] = __half2float(((__half*)sgd_info->item_group_ptr[g])[item_idx_in_group * mf_info->params.k + k]);
                    }
                    item_idx_in_group += 1;
                }
            }else{
                for (int i = item_group_start_idx; i < item_group_end_idx; i++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i * mf_info->params.k + k] = (((float*)sgd_info->item_group_ptr[g])[item_idx_in_group * mf_info->params.k + k]);
                    }
                    item_idx_in_group += 1;
                }
            }
            item_group_start_idx = item_group_end_idx;
        }

        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check && e != mf_info->params.epoch - 1){
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->params.user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->params.item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->params.user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->params.item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        if (error_check && e != mf_info->params.epoch - 1){

            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->params.user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    mf_info->user_group_error[i] = each_group_grad_sum_norm_acc/(float)each_group_norm_acc;
                    mf_info->user_group_error[i] /= initial_user_group_error[i];
                }
                else{
                    mf_info->user_group_error[i] = -1;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                if (mf_info->user_group_error[i] == -1) cout << "-1" << " ";
                else cout << mf_info->user_group_error[i] << " ";
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->params.item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_grad_sum_norm_acc/(float)each_group_norm_acc;
                    mf_info->item_group_error[i] /= initial_item_group_error[i];
                }
                else{
                    mf_info->item_group_error[i] = -1;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                if (mf_info->item_group_error[i] == -1) cout << "-1" << " ";    
                else cout << mf_info->item_group_error[i] << " ";
            }
            cout << "\n";

            if (e == start_idx){
                for (int i = 0; i < mf_info->params.user_group_num; i++){
                    initial_user_group_error[i] = mf_info->user_group_error[i];
                }
                for (int i = 0; i < mf_info->params.item_group_num; i++){
                    initial_item_group_error[i] = mf_info->item_group_error[i];
                }
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check && e > start_idx && e != mf_info->params.epoch - 1) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;         
    }

    cudaMemcpy(mf_info->test_COO, mf_info->d_test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyDeviceToHost);

    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + reconst_exec_time + cpy2grouped_parameters_exec_time + additional_info_init_exec_time;
    cout << "\n<Preprocessing time (micro sec)>" << endl;
    cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    cout << "Grouping                         : " << grouping_exec_time << endl;
    cout << "Matrix reconstruction            : " << reconst_exec_time << endl;
    cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
    cout << "Additional info init             : " << additional_info_init_exec_time << endl;
    cout << "Total preprocessing time         : " << preprocess_exec_time << endl;
    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time)/1000 << endl;

    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(d_e_group);
    cudaFree(sgd_info->d_user_group_ptr);
    cudaFree(sgd_info->d_item_group_ptr);
    cudaFree(mf_info->d_user_group_prec_info);
    cudaFree(mf_info->d_item_group_prec_info);
    cudaFree(d_grad_sum_norm_p);
    cudaFree(d_grad_sum_norm_q);
    cudaFree(d_norm_sum_p);
    cudaFree(d_norm_sum_q);
    cudaFree(mf_info->d_user_group_end_idx);
    cudaFree(mf_info->d_item_group_end_idx);
}

void adaptive_fixed_point_training_mf(Mf_info* mf_info, SGD* sgd_info){
    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));    
    unsigned int div = mf_info->params.thread_block_size/32;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    unsigned int block_num = mf_info->params.num_workers/div;
    int start_idx = 5;
    unsigned int num_groups = 10000;
    double precision_switching_and_error_comp_execution_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    

    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        
        if (mf_info->params.k == 128){
            afp_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                e,
                                update_count,
                                update_vector_size,
                                mf_info->params.lambda
            );
        }else if (mf_info->params.k == 64){
            afp_sgd_k64_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                e,
                                update_count,
                                update_vector_size,
                                mf_info->params.lambda
            );
        }
        cudaDeviceSynchronize(); 
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;

        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (sgd_update_execution_time)/1000 << endl;
    
    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(d_e_group);
}

void muppet_training_mf(Mf_info* mf_info, SGD* sgd_info){
    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->params.sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    unsigned int div = mf_info->params.thread_block_size/32;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    unsigned int block_num = mf_info->params.num_workers/div;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    float *sum_norms;
    float *d_sum_norms;
    float *sum_updated_val;
    float *d_sum_updated_val;

    double additional_info_init_exec_time = 0;
    double sgd_update_execution_time = 0;
    double error_computation_time = 0;

    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    float *sum_norms_epoch = new float[mf_info->params.epoch];
    float *sum_updated_val_epoch = new float[mf_info->params.k * mf_info->params.epoch];
    float *gradient_diversity_per_epoch = new float[mf_info->params.epoch];
    float *sum_updated_val_acc = new float[mf_info->params.k];
    cudaMalloc(&d_sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMallocHost(&sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMalloc(&d_sum_updated_val, sizeof(float) * mf_info->params.k);
    cudaMallocHost(&sum_updated_val, sizeof(float) * mf_info->params.k);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;
    float initial_error = 1; 
    int resolution_size = 3;
    float max_grad_diversity = -1.f;
    float alpha  = 1.0f;
    float beta = 1.5f;
    float lambda_for_threshold = 0.3f;
    int gamma = 1;
    int violation_times = 0;
    unsigned char bit_width_set[5] = {8, 12, 14 ,16, 32};
    int switched_epoch = -1;
    int precision_idx;

    if (mf_info->is_yahoo) precision_idx = 1;
    else precision_idx = 0;

    for (int e = 0; e < mf_info->params.epoch; e++){
        float decaying_threshold = alpha + (beta * exp(-1*lambda_for_threshold*(e)));
        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        //! clean device mem (not neccesary)
        for (int k = 0; k < mf_info->params.k; k++) sum_updated_val[k] = 0;
        cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        
        if (mf_info->params.k == 128){
            muppet_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                update_count,
                                update_vector_size,
                                mf_info->params.lambda,
                                first_sample_rating_idx,
                                d_sum_updated_val,
                                d_sum_norms,
                                (unsigned char)bit_width_set[precision_idx],
                                (unsigned char)bit_width_set[precision_idx]
                            );
        }else if (mf_info->params.k == 64){
            muppet_sgd_k64_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                update_count,
                                update_vector_size,
                                mf_info->params.lambda,
                                first_sample_rating_idx,
                                d_sum_updated_val,
                                d_sum_norms,
                                (unsigned char)bit_width_set[precision_idx],
                                (unsigned char)bit_width_set[precision_idx]
                            );
        }

        cudaDeviceSynchronize(); 
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        
        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

        error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(sum_norms, d_sum_norms, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
        cudaMemcpy(sum_updated_val, d_sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);
        
        float norm_acc = 0 ;
        
        for (int w = 0; w < mf_info->params.num_workers; w++) norm_acc += sum_norms[w];
        sum_norms_epoch[e] = norm_acc;
        for (int k = 0; k < mf_info->params.k; k++) {
            sum_updated_val_acc[k] = 0;
            sum_updated_val_epoch[e * mf_info->params.k + k] = sum_updated_val[k];
        }
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();

        if (e - switched_epoch >= resolution_size){
            error_computation_start_time = std::chrono::system_clock::now();
            float sum_norms_resolution = 0;
            float sum_updated_val_norm = 0;
            for (int wi = e - resolution_size + 1; wi <= e; wi++){
                sum_norms_resolution += sum_norms_epoch[wi];
                for (int k = 0; k < mf_info->params.k; k++) sum_updated_val_acc[k] += sum_updated_val_epoch[wi * mf_info->params.k + k];
            }

            for (int k = 0; k < mf_info->params.k; k++) {
                sum_updated_val_norm += powf(sum_updated_val_acc[k],2);
            }

            float grad_diversity_this_epoch = sum_norms_resolution/sum_updated_val_norm;
            max_grad_diversity = max_grad_diversity < grad_diversity_this_epoch ? grad_diversity_this_epoch : max_grad_diversity;
            float grad_ratio = max_grad_diversity/grad_diversity_this_epoch;
            
            if (grad_ratio > decaying_threshold && ++violation_times == gamma) {
                precision_idx++;
                switched_epoch = e;
                max_grad_diversity = -1.f;
                violation_times = 0;
            }
            error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        }
    }

    cout << "Error computation time           : " << error_computation_time << endl; 
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (error_computation_time + sgd_update_execution_time)/1000 << endl;
    
    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(d_e_group);
    cudaFree(d_sum_norms);
    cudaFree(d_sum_updated_val);
}

void mixed_precision_training_mf(Mf_info* mf_info, SGD* sgd_info){
    
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    
    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double master_copy_setting_time = 0;
    std::chrono::time_point<std::chrono::system_clock> master_copy_setting_start_time = std::chrono::system_clock::now();
    transform_feature_vector_half2float((short*)sgd_info->half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
    transform_feature_vector_half2float((short*)sgd_info->half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
    cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);
    master_copy_setting_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - master_copy_setting_start_time).count();
    
    gpuErr(cudaPeekAtLastError());

    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    gpuErr(cudaPeekAtLastError());

    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);  

    float scaling_factor = 1000;
    __half scaled_lambda = __float2half_rn(mf_info->params.lambda * scaling_factor);
    double rmse;
    double sgd_update_execution_time = 0;

    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128){
            mpt_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    sgd_info->d_p,
                                                    sgd_info->d_q,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    update_count,
                                                    update_vector_size,
                                                    scaled_lambda,
                                                    scaling_factor
                                                    );
        }else{
            mpt_sgd_k64_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    sgd_info->d_p,
                                                    sgd_info->d_q,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    update_count,
                                                    update_vector_size,
                                                    scaled_lambda,
                                                    scaling_factor
                                                    );
        }
        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;

        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);

        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "Parameters update per epoch                     : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update                         : " << sgd_update_execution_time / 1000 << endl; 
    
    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(d_e_group);
}

void training_single_mf(Mf_info *mf_info, SGD *sgd_info){

    size_t triplet_host_to_device_transfer_size = 0;

    size_t P_device_to_host_transfer_size = 0;
    size_t Q_device_to_host_transfer_size = 0;
    size_t total_device_to_host_transfer_size = 0;
    
    double triplet_host_to_device_transfer_time{};

    double P_device_to_host_transfer_time{};
    double Q_device_to_host_transfer_time{};
    double total_device_to_host_transfer_time{};

    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    gpuErr(cudaPeekAtLastError());
    
    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    gpuErr(cudaPeekAtLastError());

    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    float* lr_decay_arr = new float[2048];

    for (int i = 0; i < mf_info->params.epoch + 4; i++){
        lr_decay_arr[i] = static_cast<float>(mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5))));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    unsigned int div = mf_info->params.thread_block_size/32;
    double sgd_update_execution_time = 0;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    float rmse = 0; 

    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    

    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128)
        single_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        sgd_info->d_p,
                                                        sgd_info->d_q,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda
                                                        );

        else if (mf_info->params.k == 64)
        single_sgd_k64_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        sgd_info->d_p,
                                                        sgd_info->d_q,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda
                                                        );

        cudaDeviceSynchronize();
        double sgd_execution_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_execution_time_per_epoch;
        
        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);

        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "Execution time(avg per epoch)        : " << sgd_update_execution_time / mf_info->params.epoch << endl;
    cout << "Total execution time                 : " << sgd_update_execution_time / 1000 << endl;
    
    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(d_e_group);
}

void mascot_training_mf_naive(Mf_info* mf_info, SGD* sgd_info){
    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->params.user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->params.item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->params.user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->params.item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->params.user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->params.item_group_num);

    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->params.sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    unsigned int div = mf_info->params.thread_block_size/32;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    unsigned int block_num = mf_info->params.num_workers/div;
    
    float* d_e_group;
    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;
    float* d_user_group_sum_updated_val;
    float* d_item_group_sum_updated_val;

    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->params.user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->params.item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->params.user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->params.item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->params.user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->params.item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->params.user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->params.item_group_num);

    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->params.user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->params.item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->params.user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->params.item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->params.user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->params.item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->params.user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->params.item_group_num);

    unsigned int* user_group_end_idx_shift = new unsigned int[mf_info->params.user_group_num + 1];
    unsigned int* item_group_end_idx_shift = new unsigned int[mf_info->params.item_group_num + 1];
    unsigned int* d_user_group_end_idx_shift;
    unsigned int* d_item_group_end_idx_shift;

    user_group_end_idx_shift[0] = -1;
    item_group_end_idx_shift[0] = -1;

    for (int i = 0; i < mf_info->params.user_group_num; i++){
        user_group_end_idx_shift[i+1] = mf_info->user_group_end_idx[i];
    }

    for (int i = 0; i < mf_info->params.item_group_num; i++){
        item_group_end_idx_shift[i+1] = mf_info->item_group_end_idx[i];
    }

    cudaMalloc(&d_user_group_end_idx_shift, sizeof(unsigned int) * (mf_info->params.user_group_num + 1));
    cudaMalloc(&d_item_group_end_idx_shift, sizeof(unsigned int) * (mf_info->params.item_group_num + 1));
    cudaMemcpy(d_user_group_end_idx_shift, user_group_end_idx_shift, sizeof(unsigned int) * (mf_info->params.user_group_num + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_item_group_end_idx_shift, item_group_end_idx_shift, sizeof(unsigned int) * (mf_info->params.item_group_num + 1), cudaMemcpyHostToDevice);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    float* initial_user_group_error = new float[mf_info->params.user_group_num];
    float* initial_item_group_error = new float[mf_info->params.item_group_num];

    for (int i = 0; i <mf_info->params.user_group_num; i++) initial_user_group_error[i] = 1.0f;
    for (int i = 0; i <mf_info->params.item_group_num; i++) initial_item_group_error[i] = 1.0f;
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;

        if ((e >= start_idx ) && (e % mf_info->params.interval == (start_idx % mf_info->params.interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            initialize_float_array_to_val<<<num_groups, 512>>>(d_grad_sum_norm_p, mf_info->params.user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_groups, 512>>>(d_grad_sum_norm_q, mf_info->params.item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;
    
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128){
            naive_mascot_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda,
                                                        d_user_group_end_idx_shift,
                                                        d_item_group_end_idx_shift,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_grad_sum_norm_p,
                                                        d_grad_sum_norm_q,
                                                        d_norm_sum_p,
                                                        d_norm_sum_q,
                                                        first_sample_rating_idx,
                                                        (int)mf_info->params.user_group_num,
                                                        (int)mf_info->params.item_group_num
                                                        );
        }

        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        gpuErr(cudaPeekAtLastError());  

        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->params.user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            if (mf_info->user_group_prec_info[i] == 0) {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                // gpuErr(cudaPeekAtLastError());
            }
            else {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                // gpuErr(cudaPeekAtLastError());
            }
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->params.item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->params.user_group_num; g++){
            user_group_end_idx += mf_info->user_group_size[g];
            unsigned int user_idx_in_group = 0;
            if (mf_info->user_group_prec_info[g] == 0){
                for (int u = user_group_start_idx; u < user_group_end_idx; u++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[u * mf_info->params.k + k] = __half2float(((__half*)sgd_info->user_group_ptr[g])[user_idx_in_group * mf_info->params.k + k]);
                    }
                    user_idx_in_group += 1;
                }
            }else{
                for (int u = user_group_start_idx; u < user_group_end_idx; u++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[u * mf_info->params.k + k] = (((float*)sgd_info->user_group_ptr[g])[user_idx_in_group * mf_info->params.k + k]);
                    }
                    user_idx_in_group += 1;
                }
            }
            user_group_start_idx = user_group_end_idx;
        }

       item_group_start_idx = 0;
        unsigned int item_group_end_idx = 0;
        for (int g = 0; g < mf_info->params.item_group_num; g++){
            item_group_end_idx += mf_info->item_group_size[g];
            unsigned int item_idx_in_group = 0;
            if (mf_info->item_group_prec_info[g] == 0){
                for (int i = item_group_start_idx; i < item_group_end_idx; i++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i * mf_info->params.k + k] = __half2float(((__half*)sgd_info->item_group_ptr[g])[item_idx_in_group * mf_info->params.k + k]);
                    }
                    item_idx_in_group += 1;
                }
            }else{
                for (int i = item_group_start_idx; i < item_group_end_idx; i++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i * mf_info->params.k + k] = (((float*)sgd_info->item_group_ptr[g])[item_idx_in_group * mf_info->params.k + k]);
                    }
                    item_idx_in_group += 1;
                }
            }
            item_group_start_idx = item_group_end_idx;
        }

        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check && e != mf_info->params.epoch - 1){
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->params.user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->params.item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->params.user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->params.item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        if (error_check && e != mf_info->params.epoch - 1){

            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->params.user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    mf_info->user_group_error[i] = each_group_grad_sum_norm_acc/(float)each_group_norm_acc;
                    mf_info->user_group_error[i] /= initial_user_group_error[i];
                }
                else{
                    mf_info->user_group_error[i] = -1;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                if (mf_info->user_group_error[i] == -1) cout << "-1" << " ";                        
                else cout << mf_info->user_group_error[i] << " ";
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->params.item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_grad_sum_norm_acc/(float)each_group_norm_acc;
                    mf_info->item_group_error[i] /= initial_item_group_error[i];
                }
                else{
                    mf_info->item_group_error[i] = -1;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                                
                if (mf_info->item_group_error[i] == -1) cout << "-1" << " ";
                else cout << mf_info->item_group_error[i] << " ";
                
            }
            cout << "\n";

            if (e == start_idx){
                for (int i = 0; i < mf_info->params.user_group_num; i++){
                    initial_user_group_error[i] = mf_info->user_group_error[i];
                }
                for (int i = 0; i < mf_info->params.item_group_num; i++){
                    initial_item_group_error[i] = mf_info->item_group_error[i];
                }
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check && e > start_idx && e != mf_info->params.epoch - 1) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cudaMemcpy(mf_info->test_COO, mf_info->d_test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyDeviceToHost);

    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + reconst_exec_time + cpy2grouped_parameters_exec_time + additional_info_init_exec_time;
    cout << "\n<Preprocessing time (micro sec)>" << endl;
    cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    cout << "Grouping                         : " << grouping_exec_time << endl;
    cout << "Matrix reconstruction            : " << reconst_exec_time << endl;
    cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
    cout << "Additional info init             : " << additional_info_init_exec_time << endl;
    cout << "Total preprocessing time         : " << preprocess_exec_time << endl;
    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time)/1000 << endl;
    
    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(sgd_info->d_user_group_ptr);
    cudaFree(sgd_info->d_item_group_ptr);
    cudaFree(mf_info->d_user_group_prec_info);
    cudaFree(mf_info->d_item_group_prec_info);
    cudaFree(d_grad_sum_norm_p);
    cudaFree(d_grad_sum_norm_q);
    cudaFree(d_norm_sum_p);
    cudaFree(d_norm_sum_q);
    cudaFree(d_e_group);
    cudaFree(d_user_group_end_idx_shift);
    cudaFree(d_item_group_end_idx_shift);
}

void training_mem_quant_mf(Mf_info *mf_info, SGD *sgd_info){
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);

    gpuErr(cudaPeekAtLastError());

    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    cudaMalloc(&sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k);
    gpuErr(cudaPeekAtLastError());

    float* lr_decay_arr = new float[1024];

    for (int i = 0; i < mf_info->params.epoch + 4; i++){
        lr_decay_arr[i] = static_cast<float>(mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5))));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    unsigned int div = mf_info->params.thread_block_size/32;
    double sgd_update_execution_time = 0;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    float rmse = 0;

    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128)        
        mem_quant_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        sgd_info->d_half_p,
                                                        sgd_info->d_half_q,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda
                                                        );
        else if (mf_info->params.k == 64)
        mem_quant_sgd_k64_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        sgd_info->d_half_p,
                                                        sgd_info->d_half_q,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda
                                                        );
        cudaDeviceSynchronize();
        double sgd_execution_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_execution_time_per_epoch;
        
        cudaMemcpy(sgd_info->half_p, sgd_info->d_half_p, sizeof(half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->half_q, sgd_info->d_half_q, sizeof(half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        transform_feature_vector_half2float(sgd_info->half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
        transform_feature_vector_half2float(sgd_info->half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "Execution time(avg per epoch)        : " << sgd_update_execution_time / mf_info->params.epoch << endl;
    cout << "Total execution time                 : " << sgd_update_execution_time / 1000 << endl;
    
    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(d_e_group);
}

void training_switching_only(Mf_info* mf_info, SGD* sgd_info){
    srand(time(0)); 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    cudaMalloc(&sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->params.sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    unsigned int div = mf_info->params.thread_block_size/32;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    float quantization_error = 0;
    float *sum_norms;
    float *d_sum_norms;
    float *sum_updated_val;
    float *d_sum_updated_val;
    unsigned int block_num = mf_info->params.num_workers/div;
    
    cudaMalloc(&d_sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMallocHost(&sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMalloc(&d_sum_updated_val, sizeof(float) * mf_info->params.k);
    cudaMallocHost(&sum_updated_val, sizeof(float) * mf_info->params.k);

    double precision_switching_and_error_comp_execution_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    float initial_error = 1; 
    void* p = (void*)(sgd_info->d_half_p);
    void* q = (void*)(sgd_info->d_half_q);
    unsigned char cur_precision = 0;//# 0 => half, 1 => single

    int switching_point = 0;
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;
        if (!cur_precision &&(e >= start_idx ) && (e % mf_info->params.interval == (start_idx % mf_info->params.interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }
        std::chrono::time_point<std::chrono::system_clock> precision_switching_and_error_comp_start_time = std::chrono::system_clock::now();
        if (error_check){
            initialize_float_array_to_val<<<num_groups, 512>>>(d_sum_updated_val, mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }
        precision_switching_and_error_comp_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_and_error_comp_start_time).count();

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        switching_only_sgd_k128_hogwild_kernel<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                            mf_info->d_R,
                            mf_info->n,
                            (void*)p,
                            (void*)q,
                            d_rand_state,
                            lr_decay_arr[e],
                            mf_info->params.k,
                            update_count,
                            update_vector_size,
                            mf_info->params.lambda,
                            cur_precision,
                            first_sample_rating_idx,
                            d_sum_updated_val,
                            d_sum_norms
                        );

        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;

        gpuErr(cudaPeekAtLastError());
        if (!cur_precision){
            cudaMemcpy(sgd_info->half_p, sgd_info->d_half_p, sizeof(half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
            cudaMemcpy(sgd_info->half_q, sgd_info->d_half_q, sizeof(half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);

            transform_feature_vector_half2float(sgd_info->half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
            transform_feature_vector_half2float(sgd_info->half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
        }else{
            cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
            cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        }

        precision_switching_and_error_comp_start_time = std::chrono::system_clock::now();
        if (error_check && e != mf_info->params.epoch - 1){

            cudaMemcpy(sum_norms, d_sum_norms, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
            cudaMemcpy(sum_updated_val, d_sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);

            float sum_norms_acc = 0;
            float norm_acc = 0 ;
            
            for (int w = 0; w < mf_info->params.num_workers; w++){
                norm_acc += sum_norms[w];
            }

            for (int k = 0; k < mf_info->params.k; k++){
                sum_norms_acc += powf(sum_updated_val[k],2);
                sum_updated_val[k] = 0.0f;
            }

            quantization_error = (float)sum_norms_acc/(float)norm_acc;
            quantization_error = quantization_error/(float)initial_error;
            cout << "Quantization error : " << quantization_error << endl;
            
            if (e == start_idx) initial_error = quantization_error;

            if (e > start_idx && quantization_error > mf_info->params.error_threshold) {
                transition_params_half2float(mf_info, sgd_info);
                cur_precision = 1;
                p = (void*)(sgd_info->d_p);
                q = (void*)(sgd_info->d_q);
                cudaFree(sgd_info->d_half_p);
                cudaFree(sgd_info->d_half_q);
                switching_point = e;
            }
        }
        
        precision_switching_and_error_comp_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_and_error_comp_start_time).count();
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }
    cout << "Additional info init                            : " << additional_info_init_exec_time << endl;
    cout << "Total error comp & precision switching time     : " << precision_switching_and_error_comp_execution_time << endl;
    cout << "Parameters update per epoch                     : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update                         : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                               : " <<(additional_info_init_exec_time + precision_switching_and_error_comp_execution_time + sgd_update_execution_time)/1000 << endl;
    
    cudaFree(mf_info->d_R);
    cudaFree(mf_info->d_test_COO);
    cudaFree(d_rand_state);
    cudaFree(d_e_group);
    cudaFree(d_sum_norms);
    cudaFree(d_sum_updated_val);
}
