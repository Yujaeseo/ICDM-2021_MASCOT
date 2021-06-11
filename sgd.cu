#include <iostream>
#include <string>
#include <boost/filesystem.hpp>
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
#include "model_init.h"
#include "preprocess_utils.h"
#include "sgd_kernel_128.h"
#include "sgd_kernel_64.h"
#include "test_kernel.h"
#include "rmse.h"
#include "precision_switching.h"
#include "statistics.h"
#include "fixed_point_sgd_kernel_128.h"
#include "fixed_point_sgd_kernel_128_cvpr.h"
#include "fixed_point_sgd_kernel_64.h"
#include "fixed_point_sgd_kernel_64_cvpr.h"
#define WRITE_FILE

using namespace std;

extern __global__ void init_rand_state(curandState*state, int size);


// void check_group_cnt(Mf_info* mf_info){
//     unsigned int* user_group_cnt = new unsigned int[mf_info->user_group_num]{};
//     unsigned int* item_group_cnt = new unsigned int[mf_info->item_group_num]{};
//     unsigned int* user_group_rating_cnt = new unsigned int[mf_info->user_group_num]{};
//     unsigned int* item_group_rating_cnt = new unsigned int[mf_info->item_group_num]{};
//     unsigned int* grid_rating_cnt = new unsigned int[mf_info->user_group_num * mf_info->item_group_num]{};

//     for (int i = 0; i < mf_info->n; i++) {
//         unsigned int user_group = mf_info->user_group_idx[mf_info->R[i].u];
//         unsigned int item_group = mf_info->item_group_idx[mf_info->R[i].i];
//         grid_rating_cnt[user_group * mf_info->user_group_num + item_group]++;
//         user_group_rating_cnt[user_group]++;
//         item_group_rating_cnt[item_group]++;
//     }

//     for (int i = 0; i < mf_info->max_user; i++){
//         user_group_cnt[mf_info->user_group_idx[mf_info->user2idx[i]]]++;
//     }
//     for (int i = 0; i < mf_info->max_item; i++){
//         item_group_cnt[mf_info->item_group_idx[mf_info->item2idx[i]]]++;
//     }

//     unsigned int total = 0;
//     cout << "\nCheck user grouping" << endl;
//     for (int i = 0; i < mf_info->user_group_num; i++){
//         total+=user_group_cnt[i];
//         cout << user_group_cnt[i] << " ";
//     }
//     cout << "*" << total << endl;
//     total = 0;
//     cout << "\nCheck item grouping" << endl;
//     for (int i = 0; i < mf_info->item_group_num; i++){
//         total+=item_group_cnt[i];
//         cout << item_group_cnt[i] << " ";
//     }
//     cout << "*" << total << endl;

//     cout << "\nCheck grid rating cnt" << endl;
//     total = 0;
//     for (int i = 0; i < mf_info->user_group_num * mf_info->item_group_num; i++){
//         total+= grid_rating_cnt[i];
//         cout << grid_rating_cnt[i] << " ";
//     }
//     cout << "*" << total << endl;
//     cout << "\n";
//     total = 0;
//     cout << "\nCheck user group rating num" << endl;
//     for (int i = 0; i < mf_info->user_group_num; i++){
//         total+=user_group_rating_cnt[i];
//         cout << user_group_rating_cnt[i] << " ";
//     }
//     cout << "*" << total << endl;
//     total = 0;
//     cout << "\nCheck item group rating num" << endl;
//     for (int i = 0; i < mf_info->item_group_num; i++){
//         total+=item_group_rating_cnt[i];
//         cout << item_group_rating_cnt[i] << " ";
//     }
//     cout << "*" << total << endl;
// }

void sgd_training_single(Mf_info *mf_info, SGD *sgd_info){

    size_t triplet_host_to_device_transfer_size = 0;

    size_t P_device_to_host_transfer_size = 0;
    size_t Q_device_to_host_transfer_size = 0;
    size_t total_device_to_host_transfer_size = 0;
    
    double triplet_host_to_device_transfer_time{};

    double P_device_to_host_transfer_time{};
    double Q_device_to_host_transfer_time{};
    double total_device_to_host_transfer_time{};

    curandState* rand_state;
    cudaMalloc(&rand_state, sizeof(curandState)*mf_info->params.num_workers);
    gpuErr(cudaPeekAtLastError());

    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(rand_state, mf_info->params.num_workers);
    gpuErr(cudaPeekAtLastError());
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    gpuErr(cudaPeekAtLastError());

    std::chrono::time_point<std::chrono::system_clock> triplet_host_to_device_start_time = std::chrono::system_clock::now();
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    triplet_host_to_device_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - triplet_host_to_device_start_time).count();
    triplet_host_to_device_transfer_size = sizeof(Node) * mf_info->n;

    cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    float* lr_decay_arr = new float[2048];

    for (int i = 0; i < mf_info->params.epoch + 4; i++){
        lr_decay_arr[i] = static_cast<float>(mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5))));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    cout << "Start FP32 SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;

    // // test rmse
    Node* test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    float rmse = 0; 

    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128)
        sgd_k128_kernel_hogwild_warp32_lrate<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        sgd_info->d_p,
                                                        sgd_info->d_q,
                                                        rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda
                                                        );

        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
   
        std::chrono::time_point<std::chrono::system_clock> P_device_to_host_transfer_start_time = std::chrono::system_clock::now();
        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        P_device_to_host_transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - P_device_to_host_transfer_start_time).count();
        P_device_to_host_transfer_size = sizeof(sgd_info->p[0]) * mf_info->max_user * mf_info->params.k;

        std::chrono::time_point<std::chrono::system_clock> Q_device_to_host_transfer_start_time = std::chrono::system_clock::now();
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        Q_device_to_host_transfer_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - Q_device_to_host_transfer_start_time).count();
        Q_device_to_host_transfer_size = sizeof(sgd_info->q[0]) * mf_info->max_item * mf_info->params.k;


        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);

        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    P_device_to_host_transfer_time = P_device_to_host_transfer_time / (double) mf_info->params.epoch;
    Q_device_to_host_transfer_time = Q_device_to_host_transfer_time / (double) mf_info->params.epoch;
    total_device_to_host_transfer_time = P_device_to_host_transfer_time + Q_device_to_host_transfer_time;
    total_device_to_host_transfer_size = P_device_to_host_transfer_size + Q_device_to_host_transfer_size;
    
    cout << "Triplet host to device transfer time : " << triplet_host_to_device_transfer_time << endl;
    cout << "P host to device transfer time       : " << P_device_to_host_transfer_time << endl;
    cout << "Q host to device transfer time       : " << Q_device_to_host_transfer_time << endl;
    cout << "Total host to device transfer time   : " << total_device_to_host_transfer_time << endl;
    cout << "Total execution time                 : " << sgd_update_execution_time << endl;
    cout << "Execution time(avg per epoch)        : " << sgd_update_execution_time / mf_info->params.epoch << endl;
    cout << "SGD FP32 update have been finished..." << endl;
}

void grouped_sgd_training_map_based_indexing(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
    //* Grouping methods
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);

    // check_group_cnt(mf_info);
    //* Generating index
    generate_map_idx_info(mf_info);
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    cpy2grouped_parameters_gpu(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;

    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;

    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    // for (int i = 0; i < mf_info->user_group_num; i++)
    //     mf_info->user_group_prec_info[i] = 1;
    // for (int i = 0; i < mf_info->item_group_num; i++)
    //     mf_info->item_group_prec_info[i] = 0;

    // cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    unsigned int *user_group_update_cnt;
    unsigned int *item_group_update_cnt;
    unsigned int *d_user_group_update_cnt;
    unsigned int *d_item_group_update_cnt;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_acc_user_group_error,
                                                    d_acc_item_group_error,
                                                    d_user_group_update_cnt,
                                                    d_item_group_update_cnt,
                                                    first_sample_rating_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        
        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        // if (e == mf_info->params.epoch-1){
            unsigned user_group_start_idx = 0;
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                if (mf_info->user_group_prec_info[i] == 0) {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
                else {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
            }
            gpuErr(cudaPeekAtLastError());

            unsigned item_group_start_idx = 0;
            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            for (int i = 0; i < mf_info->max_user; i++){
                unsigned int user_group = mf_info->user_index_info[i].g;
                unsigned int row = mf_info->user_index_info[i].v;
                if (mf_info->user_group_prec_info[user_group] == 0){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
                    }     
                }
                else {
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
                    }
                }
            }

            for (int i = 0; i < mf_info->max_item; i++){
                unsigned int item_group = mf_info->item_index_info[i].g;
                unsigned int row = mf_info->item_index_info[i].v;
                if (mf_info->item_group_prec_info[item_group] == 0){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
                    }
                }
                else{
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
                    }
                }
            }
        // }
        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        cout << "Transfer time " << error_transfer_time << endl;
        cout << "\n<User groups>\n";
        error_computation_start_time = std::chrono::system_clock::now();
        
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->user_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_user_group_error[i * block_num + j];
                    each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
                }
                if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
                // if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(mf_info->params.k) / (float)mf_info->n;
                else mf_info->user_group_error[i] = 0;
            }
            else{
                mf_info->user_group_error[i] = -1;
            }
            // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->user_group_error[i] << endl;
            cout << mf_info->user_group_error[i] << " ";

        }

        cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->item_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_item_group_error[i * block_num + j];
                    each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
                }
                if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
                // if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(mf_info->params.k) / (float)mf_info->n;
                else mf_info->item_group_error[i] = 0;
            }
            else{
                mf_info->item_group_error[i] = -1;
            }
            // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->item_group_error[i] << endl;
            cout << mf_info->item_group_error[i] << " ";

        }
        cout << "\n";
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        // if (e > 0) precision_switching_by_groups(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        // if (e == mf_info->params.epoch-1){
        double rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);

        // }

        // if (e == 0){    
        //     for (int i = 0; i < mf_info->user_group_num; i++)
        //         mf_info->user_group_prec_info[i] = 1;
        //     for (int i = 0; i < mf_info->item_group_num; i++)
        //         mf_info->item_group_prec_info[i] = 0;

        //     cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
        //     cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
        // }
    }

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl; 
}

void grouped_sgd_training_map_based_indexing_fp32_version(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
    //* Grouping methods
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);

    check_group_cnt(mf_info);
    //* Generating index
    generate_map_idx_info(mf_info);
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    cpy2grouped_parameters_gpu_float_version(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;

    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;

    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    // __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    // cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    // cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    unsigned int *user_group_update_cnt;
    unsigned int *item_group_update_cnt;
    unsigned int *d_user_group_update_cnt;
    unsigned int *d_item_group_update_cnt;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_fp32_version<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    mf_info->params.lambda,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    d_acc_user_group_error,
                                                    d_acc_item_group_error,
                                                    d_user_group_update_cnt,
                                                    d_item_group_update_cnt,
                                                    first_sample_rating_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        
    //     //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
    //     // if (e == mf_info->params.epoch-1){
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

        for (int i = 0; i < mf_info->max_user; i++){
            unsigned int user_group = mf_info->user_index_info[i].g;
            unsigned int row = mf_info->user_index_info[i].v;  
            for (int k = 0; k < mf_info->params.k; k++){
                sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
            }
        }

        for (int i = 0; i < mf_info->max_item; i++){
            unsigned int item_group = mf_info->item_index_info[i].g;
            unsigned int row = mf_info->item_index_info[i].v;
            for (int k = 0; k < mf_info->params.k; k++){
                sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
            }
        }
    //     // }
        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        // cout << "Transfer time " << error_transfer_time << endl;
        cout << "\n<User groups>\n";
        error_computation_start_time = std::chrono::system_clock::now();
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            for (int j = 0; j < block_num; j++){
                each_group_error_acc += acc_user_group_error[i * block_num + j];
                each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
            }
            if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            // if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(mf_info->params.k) / (float)mf_info->n;
            else mf_info->user_group_error[i] = 0;
            cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->user_group_error[i] << endl;
            // cout << mf_info->user_group_error[i] << " ";

        }

        cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            for (int j = 0; j < block_num; j++){
                each_group_error_acc += acc_item_group_error[i * block_num + j];
                each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
            }
            if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            // if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(mf_info->params.k) / (float)mf_info->n;
            else mf_info->item_group_error[i] = 0;
            cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->item_group_error[i] << endl;
            // cout << mf_info->item_group_error[i] << " ";

        }
        cout << "\n";
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
    //     std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
    //     if (e > 0) precision_switching_by_groups(mf_info, sgd_info);
    //     precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

    //     // if (e == mf_info->params.epoch-1){
        double rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
    //     // }

    //     // if (e == 0){    
    //     //     for (int i = 0; i < mf_info->user_group_num; i++)
    //     //         mf_info->user_group_prec_info[i] = 1;
    //     //     for (int i = 0; i < mf_info->item_group_num; i++)
    //     //         mf_info->item_group_prec_info[i] = 0;

    //     //     cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    //     //     cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    //     // }
    }

    // cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    // cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    // cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    // cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    // cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    // cout << "Total error computation time     : " << error_computation_time << endl;
    // cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    // cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    // cout << "Total parameters update     : " << sgd_update_execution_time << endl; 
}

// void grouped_sgd_training_map_based_indexing(Mf_info* mf_info, SGD* sgd_info){
//     //* Transfer rating triplets to GPU 
//     // random_shuffle(mf_info->R, mf_info->R + mf_info->n);

//     cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
//     cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
//     gpuErr(cudaPeekAtLastError());
    
//     //* Convert testset to COO format
//     mf_info->test_COO = test_set_preprocess(mf_info);
//     Node* d_test_COO;
//     cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
//     gpuErr(cudaPeekAtLastError());
//     cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

//     //* Histogram on GPU
//     user_item_rating_histogram(mf_info);
    
//     //* Grouping on CPU
//     mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
//     mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
//     //* Grouping methods
//     if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
//     else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
//     else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
//     else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);

//     check_group_cnt(mf_info);
//     //* Generating index
//     generate_map_idx_info(mf_info);
//     // for (int i = 0; i < mf_info->user_group_num; i++) cout << mf_info->user_group_size[i] << " ";
//     // cout << endl;
//     // for (int i = 0; i < mf_info->item_group_num; i++) cout << mf_info->item_group_size[i] << " ";
//     // cout << endl;
//     //* Initialize random states
//     curandState* d_rand_state;
//     cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
//     init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
//     cudaDeviceSynchronize();
//     gpuErr(cudaPeekAtLastError());

//     cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
//     cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
//     cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
//     cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

//     cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
//     cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
//     cpy2grouped_parameters_gpu(mf_info, sgd_info);


//     //!============================================ debug =====================================================
//     __half* temp_p = new __half[mf_info->max_user * mf_info->params.k];
//     __half* temp_q = new __half[mf_info->max_item * mf_info->params.k];
//     for (int i = 0; i < mf_info->max_user * mf_info->params.k; i++){
//         temp_p[i] = __float2half_rn(sgd_info->p[i]);
//     }
//     for (int i = 0; i < mf_info->max_item * mf_info->params.k; i++){
//         temp_q[i] = __float2half_rn(sgd_info->q[i]);        
//     }
//     unsigned user_group_start_idx = 0;
//     for (int i = 0; i < mf_info->user_group_num; i++){
//         unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
//         cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
//         gpuErr(cudaPeekAtLastError());
//     }
//     gpuErr(cudaPeekAtLastError());

//     unsigned item_group_start_idx = 0;
//     for (int i = 0; i < mf_info->item_group_num; i++){
//         unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
//         cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
//     }
//     gpuErr(cudaPeekAtLastError());

//     float p_acc = 0;
//     float q_acc = 0;
//     float p_minus_sum = 0;
//     float q_minus_sum = 0;

//     for (int i = 0; i < mf_info->max_user; i++){
//         unsigned int user_group = mf_info->user_index_info[i].g;
//         unsigned int row = mf_info->user_index_info[i].v;
//         for (int k = 0; k < mf_info->params.k; k++){
//             p_acc += __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
//             p_minus_sum +=  __half2float(temp_p[i*mf_info->params.k + k]);
//         }     
//     }
//     for (int i = 0; i < mf_info->max_item; i++){
//         unsigned int item_group = mf_info->item_index_info[i].g;
//         unsigned int row = mf_info->item_index_info[i].v;
//         for (int k = 0; k < mf_info->params.k; k++){
//             q_acc += __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
//             q_minus_sum +=  __half2float(temp_q[i*mf_info->params.k + k]);
//         }     
//     }

//     cout << "P acc : " << p_acc << endl;
//     cout << "Q acc : " << q_acc << endl;
//     cout << "P minus acc : " << p_minus_sum << endl;
//     cout << "Q minus acc : " << q_minus_sum << endl;
//     // return ;

//     float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
//     for (int i = 0; i < mf_info->params.epoch; i++){
//         lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
//     }
//     __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

//     for (int i = 0; i < mf_info->params.epoch; i++){
//         lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
//     }

//     int update_vector_size = 128;
//     int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
//     int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
//     unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;

//     cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
//     cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
//     cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
//     cout << "Start SGD update..." << endl;

//     double sgd_update_execution_time = 0;
//     unsigned int div = mf_info->params.thread_block_size/32;
    
//     // unsigned int error_kernel_work_groups = 2048;
//     unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
//     unsigned int group_error_size = error_kernel_work_groups;
//     unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
//     unsigned int seg_size = 32;
//     float* d_e_group;
//     cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

//     __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
//     cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
//     cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
//     double additional_data_transfer_time = 0;
//     std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

//     cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
//     cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
//     cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
//     cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
//     cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
//     cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
//     // for (int i = 0; i < mf_info->user_group_num; i++)
//     //     mf_info->user_group_prec_info[i] = 1;
//     // for (int i = 0; i < mf_info->item_group_num; i++)
//     //     mf_info->item_group_prec_info[i] = 0;

//     // cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
//     // cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);

//     cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
//     cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

//     unsigned int *acc_user_group_error;
//     unsigned int *acc_item_group_error;
//     unsigned int *d_acc_user_group_error;
//     unsigned int *d_acc_item_group_error;
//     unsigned int *user_group_update_cnt;
//     unsigned int *item_group_update_cnt;
//     unsigned int *d_user_group_update_cnt;
//     unsigned int *d_item_group_update_cnt;

//     unsigned int block_num = mf_info->params.num_workers/div;
//     cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
//     cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
//     cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
//     cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
//     cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
//     cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
//     cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
//     cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
//     additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    
//     cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
//     cout << "Time                             : " << additional_data_transfer_time << endl; 

//     double error_computation_time = 0;
//     double precision_switching_exec_time = 0;
//     size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
//     for (int e = 0; e < mf_info->params.epoch; e++){

//         std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
//         sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
//                                                     mf_info->d_R,
//                                                     mf_info->n,
//                                                     (void**)sgd_info->d_user_group_ptr,
//                                                     (void**)sgd_info->d_item_group_ptr,
//                                                     d_rand_state,
//                                                     lr_decay_arr_half[e],
//                                                     mf_info->params.k,
//                                                     1,
//                                                     e,
//                                                     update_count,
//                                                     update_vector_size,
//                                                     lambda_half,
//                                                     mf_info->d_user_index_info,
//                                                     mf_info->d_item_index_info,
//                                                     mf_info->d_user_group_prec_info,
//                                                     mf_info->d_item_group_prec_info,
//                                                     d_acc_user_group_error,
//                                                     d_acc_item_group_error,
//                                                     d_user_group_update_cnt,
//                                                     d_item_group_update_cnt,
//                                                     first_sample_rating_idx,
//                                                     mf_info->user_group_num,
//                                                     mf_info->item_group_num
//                                                     );
//         cudaDeviceSynchronize();
//         sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
//         gpuErr(cudaPeekAtLastError());    
        
//         //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
//         // if (e == mf_info->params.epoch-1){
//             unsigned user_group_start_idx = 0;
//             for (int i = 0; i < mf_info->user_group_num; i++){
//                 unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
//                 if (mf_info->user_group_prec_info[i] == 0) {
//                     cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
//                     gpuErr(cudaPeekAtLastError());
//                 }
//                 else {
//                     cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
//                     gpuErr(cudaPeekAtLastError());
//                 }
//             }
//             gpuErr(cudaPeekAtLastError());

//             unsigned item_group_start_idx = 0;
//             for (int i = 0; i < mf_info->item_group_num; i++){
//                 unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
//                 if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
//                 else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
//             }
//             gpuErr(cudaPeekAtLastError());

//             std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();
//             p_acc = 0;
//             q_acc = 0;
//             for (int i = 0; i < mf_info->max_user; i++){
//                 unsigned int user_group = mf_info->user_index_info[i].g;
//                 unsigned int row = mf_info->user_index_info[i].v;
//                 if (mf_info->user_group_prec_info[user_group] == 0){
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
//                         p_acc += sgd_info->p[i*mf_info->params.k + k];
//                     }     
//                 }
//                 else {
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
//                     }
//                 }
//             }

//             for (int i = 0; i < mf_info->max_item; i++){
//                 unsigned int item_group = mf_info->item_index_info[i].g;
//                 unsigned int row = mf_info->item_index_info[i].v;
//                 if (mf_info->item_group_prec_info[item_group] == 0){
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
//                         q_acc += sgd_info->q[i*mf_info->params.k + k];
//                     }
//                 }
//                 else{
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
//                     }
//                 }
//             }

//             cout << "p acc : " << p_acc << endl;
//             cout << "q acc : " << q_acc << endl;
//         // }
//         std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
//         cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
//         cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
//         cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
//         cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
//         double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
//         error_computation_time += error_transfer_time;
        
//         cout << "Transfer time " << error_transfer_time << endl;
//         cout << "\n<User groups>\n";
//         error_computation_start_time = std::chrono::system_clock::now();
//         for (int i = 0; i < mf_info->user_group_num; i++){
//             unsigned int each_group_error_acc = 0;
//             unsigned int each_group_cnt_acc = 0;
//             if (mf_info->user_group_prec_info[i] == 0){
//                 for (int j = 0; j < block_num; j++){
//                     each_group_error_acc += acc_user_group_error[i * block_num + j];
//                     each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
//                 }
//                 if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
//                 // if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(mf_info->params.k) / (float)mf_info->n;
//                 else mf_info->user_group_error[i] = 0;
//             }
//             else{
//                 mf_info->user_group_error[i] = -1;
//             }
//             // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->user_group_error[i] << endl;
//             cout << mf_info->user_group_error[i] << " ";

//         }

//         cout << "\n<Item groups>\n";
//         for (int i = 0; i < mf_info->item_group_num; i++){
//             unsigned int each_group_error_acc = 0;
//             unsigned int each_group_cnt_acc = 0;
//             if (mf_info->item_group_prec_info[i] == 0){
//                 for (int j = 0; j < block_num; j++){
//                     each_group_error_acc += acc_item_group_error[i * block_num + j];
//                     each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
//                 }
//                 if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
//                 // if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(mf_info->params.k) / (float)mf_info->n;
//                 else mf_info->item_group_error[i] = 0;
//             }
//             else{
//                 mf_info->item_group_error[i] = -1;
//             }
//             // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->item_group_error[i] << endl;
//             cout << mf_info->item_group_error[i] << " ";

//         }
//         cout << "\n";
//         error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
//         std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
//         // if (e > 0) precision_switching_by_groups(mf_info, sgd_info);
//         precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

//         // if (e == mf_info->params.epoch-1){
//         double rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
//         cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
//         // }

//         // if (e == 0){    
//         //     for (int i = 0; i < mf_info->user_group_num; i++)
//         //         mf_info->user_group_prec_info[i] = 1;
//         //     for (int i = 0; i < mf_info->item_group_num; i++)
//         //         mf_info->item_group_prec_info[i] = 0;

//         //     cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
//         //     cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
//         // }
//     }

//     cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
//     cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
//     cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
//     cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
//     cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
//     cout << "Total error computation time     : " << error_computation_time << endl;
//     cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
//     cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
//     cout << "Total parameters update     : " << sgd_update_execution_time << endl; 
// }

void grouped_sgd_training_comparison_based_indexing(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    // mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    // mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
    cudaMallocHost(&mf_info->user_group_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item_group_idx, sizeof(unsigned int) * mf_info->max_item);
    cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    //! Input to end_idx arr and group_idx arr
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);

    cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    // check_group_cnt(mf_info);
    
    //* GENERATE RECONSTRUTED INDEX
    mf_info->sorted_idx2user = new unsigned int[mf_info->max_user];
    mf_info->sorted_idx2item = new unsigned int[mf_info->max_item];
    cudaMallocHost(&mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);
    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    matrix_reconstruction(mf_info);

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    gpuErr(cudaPeekAtLastError());

    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    // for (int i = 0; i < mf_info->user_group_num; i++)
    //     mf_info->user_group_prec_info[i] = 1;
    // for (int i = 0; i < mf_info->item_group_num; i++)
    //     mf_info->item_group_prec_info[i] = 0;

    // cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    unsigned int *user_group_update_cnt;
    unsigned int *item_group_update_cnt;
    unsigned int *d_user_group_update_cnt;
    unsigned int *d_item_group_update_cnt;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    // cudaMemset(d_acc_user_group_error, 0, sizeof(float) * block_num * mf_info->user_group_num);
    // cudaMemset(d_acc_item_group_error, 0, sizeof(float) * block_num * mf_info->item_group_num);
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (5*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (2 * sizeof(unsigned int)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_range_based_indexing<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_acc_user_group_error,
                                                    d_acc_item_group_error,
                                                    d_user_group_update_cnt,
                                                    d_item_group_update_cnt,
                                                    first_sample_rating_idx,
                                                    mf_info->d_user_group_end_idx,
                                                    mf_info->d_item_group_end_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        // if (e == mf_info->params.epoch-1){
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                if (mf_info->user_group_prec_info[i] == 0) {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
                else {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
            }
            gpuErr(cudaPeekAtLastError());

            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            unsigned int user_group_start_idx = 0;
            unsigned int user_group_end_idx = 0;
            for (int g = 0; g < mf_info->user_group_num; g++){
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

            unsigned int item_group_start_idx = 0;
            unsigned int item_group_end_idx = 0;
            for (int g = 0; g < mf_info->item_group_num; g++){
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

            // for (int i = 0; i < mf_info->max_user; i++){
            //     unsigned int user_group = mf_info->user_index_info[i].g;
            //     unsigned int row = mf_info->user_index_info[i].v;
            //     if (mf_info->user_group_prec_info[user_group] == 0){
            //         for (int k = 0; k < mf_info->params.k; k++){
            //             sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
            //         }     
            //     }
            //     else {
            //         for (int k = 0; k < mf_info->params.k; k++){
            //             sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
            //         }
            //     }
            // }

            // for (int i = 0; i < mf_info->max_item; i++){
            //     unsigned int item_group = mf_info->item_index_info[i].g;
            //     unsigned int row = mf_info->item_index_info[i].v;
            //     if (mf_info->item_group_prec_info[item_group] == 0){
            //         for (int k = 0; k < mf_info->params.k; k++){
            //             sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
            //         }
            //     }
            //     else{
            //         for (int k = 0; k < mf_info->params.k; k++){
            //             sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
            //         }
            //     }
            // }
        // }
        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        cout << "Transfer time " << error_transfer_time << endl;
        cout << "\n<User groups>\n";
        error_computation_start_time = std::chrono::system_clock::now();
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->user_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_user_group_error[i * block_num + j];
                    each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
                }
                mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            }
            else{
                mf_info->user_group_error[i] = -1;
            }
            cout << mf_info->user_group_error[i] << " ";
        }

        cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->item_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_item_group_error[i * block_num + j];
                    each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
                }
                mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            }
            else{
                mf_info->item_group_error[i] = -1;
            }
            cout << mf_info->item_group_error[i] << " ";
        }
        cout << "\n";
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        // if (e == 0)
        // precision_switching_by_groups(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        // if (e == mf_info->params.epoch-1){
        double rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        // }

        // if (e == 0){    
        //     for (int i = 0; i < mf_info->user_group_num; i++)
        //         mf_info->user_group_prec_info[i] = 1;
        //     for (int i = 0; i < mf_info->item_group_num; i++)
        //         mf_info->item_group_prec_info[i] = 0;

        //     cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
        //     cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
        // }
    }

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl; 
}

void grouped_sgd_training_comparison_based_indexing_eval_indexing(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    // random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //! Input to end_idx arr and group_idx arr
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();
    
    //* GENERATE RECONSTRUTED INDEX
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    gpuErr(cudaPeekAtLastError());
    
    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int block_num = mf_info->params.num_workers/div;
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 
    cout << "*Adjusted user group num         : " << mf_info->user_group_num << endl;
    cout << "*Adjusted item group num         : " << mf_info->item_group_num << endl;
    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double rmse;
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (2 * sizeof(unsigned int)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_range_based_indexing_eval_only_idexing<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    mf_info->d_user_group_end_idx,
                                                    mf_info->d_item_group_end_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        // if (e == mf_info->params.epoch-1){
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                if (mf_info->user_group_prec_info[i] == 0) {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
                else {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
            }
            gpuErr(cudaPeekAtLastError());

            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            unsigned int user_group_start_idx = 0;
            unsigned int user_group_end_idx = 0;
            for (int g = 0; g < mf_info->user_group_num; g++){
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

            unsigned int item_group_start_idx = 0;
            unsigned int item_group_end_idx = 0;
            for (int g = 0; g < mf_info->item_group_num; g++){
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

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (sgd_update_execution_time)/1000 << endl;

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = 0;
    statistics_map["switching"] = 0;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = sgd_update_execution_time / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/indexing_naive_version/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
#endif
}


void grouped_sgd_training_map_based_grad_diversity(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);


    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    // check_group_cnt(mf_info);

    //* Generating index
    double generate_map_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> generate_map_start_point = std::chrono::system_clock::now();
    generate_map_idx_info(mf_info);
    generate_map_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - generate_map_start_point).count();

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    // sgd_info->user_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // sgd_info->user_group_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    // cpy2grouped_parameters(mf_info, sgd_info);
    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    cpy2grouped_parameters_gpu(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    // cout  << "User group num : " << mf_info->user_group_num;

    // cout  << "Item group num : " << mf_info->item_group_num;
    

    // mf_info->user_group_prec_info = new unsigned char[mf_info->user_group_num];
    // mf_info->item_group_prec_info = new unsigned char[mf_info->item_group_num];
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (sizeof(unsigned int)*mf_info->params.k*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    double rmse;
    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
    int start_idx = 5;
// #ifdef TEST
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_grad_sum_norm_p,
                                                    d_grad_sum_norm_q,
                                                    d_norm_sum_p,
                                                    d_norm_sum_q,
                                                    first_sample_rating_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        
        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            if (mf_info->user_group_prec_info[i] == 0) {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
            else {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

        for (int i = 0; i < mf_info->max_user; i++){
            unsigned int user_group = mf_info->user_index_info[i].g;
            unsigned int row = mf_info->user_index_info[i].v;
            if (mf_info->user_group_prec_info[user_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
                }     
            }
            else {
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        for (int i = 0; i < mf_info->max_item; i++){
            unsigned int item_group = mf_info->item_index_info[i].g;
            unsigned int row = mf_info->item_index_info[i].v;
            if (mf_info->item_group_prec_info[item_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
                }
            }
            else{
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        cout << "Transfer time " << error_transfer_time << endl;
        cout << "\n<User groups>\n";
        for (int i = 0; i < mf_info->user_group_num; i++){
            error_computation_start_time = std::chrono::system_clock::now();
            float each_group_grad_sum_norm_acc = 0;
            float each_group_norm_acc = 0;
            if (mf_info->user_group_prec_info[i] == 0){
                
                for (int k = 0; k < mf_info->params.k; k++){
                    each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                    grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                }
                for (int j = 0; j < block_num; j++){
                    // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                    each_group_norm_acc += norm_sum_p[i * block_num + j];
                }
                // cout << each_group_grad_sum_norm_acc << " ";
                mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
            }
            else{
                mf_info->user_group_error[i] = UINT_MAX;
            }
            error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
            //! log code
            if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e >= start_idx) user_switching_log[e].push_back(i);

            if (mf_info->user_group_error[i] == UINT_MAX) {
                cout << "-1" << " ";
                user_grad_diversity_log[e][i] = -1;
            }           
            else {
                cout << mf_info->user_group_error[i] << " ";
                user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
            }
        }

        cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            error_computation_start_time = std::chrono::system_clock::now();
            float each_group_grad_sum_norm_acc = 0;
            float each_group_norm_acc = 0;
            if (mf_info->item_group_prec_info[i] == 0){

                for (int k = 0; k < mf_info->params.k; k++){
                    each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                    grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                }

                for (int j = 0; j < block_num; j++){
                    // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                    each_group_norm_acc += norm_sum_q[i * block_num + j];
                }
                mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                // cout << each_group_grad_sum_norm_acc << " ";
            }
            else{
                mf_info->item_group_error[i] = UINT_MAX;
            }
            error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
            //! log code
            if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e >= start_idx) item_switching_log[e].push_back(i);
            
            if (mf_info->item_group_error[i] == UINT_MAX) {
                cout << "-1" << " ";
                item_grad_diversity_log[e][i] = -1;
            }            
            else {
                cout << mf_info->item_group_error[i] << " ";
                item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
            }
        }
        cout << "\n";
        // error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        // if (e >= start_idx) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
        precision_switching_start_point = std::chrono::system_clock::now();
        cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
    }

    // //! RMSE write code
    // string group_error_metric_output_file_path = string("./statistics/") + mf_info->out_file + ".txt";  
    // print_rmse(group_error_metric_output_file_path, rmse);
    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + generate_map_exec_time + cpy2grouped_parameters_exec_time;
    cout << "\n<Preprocessing time (micro sec)>" << endl;
    cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    cout << "Grouping                         : " << grouping_exec_time << endl;
    cout << "Generate map idx                 : " << generate_map_exec_time << endl;
    cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
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

    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    // string exec_rmse_output_file_path = string("./statistics/grouping/time_rmse/atomic_ver_time_rmse_") + mf_info->out_file + ".txt";  
    // string group_switching_log_output_file_path = string("./statistics/grouping/switching_log/atomic_ver_group_switching_log_") + mf_info->out_file + ".txt";
    // string group_diversity_log_output_file_path = string("./statistics/grouping/diversity_log/atomic_ver_group_diversity_log_") + mf_info->out_file + ".txt";
    
    // print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    // print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    // print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
// #endif
}

void grouped_sgd_training_map_based_grad_diversity_eval_indexing(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    // check_group_cnt(mf_info);

    //* Generating index
    double generate_map_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> generate_map_start_point = std::chrono::system_clock::now();
    generate_map_idx_info(mf_info);
    generate_map_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - generate_map_start_point).count();

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    // sgd_info->user_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // sgd_info->user_group_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    // cpy2grouped_parameters(mf_info, sgd_info);
    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    cpy2grouped_parameters_gpu(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    // cout  << "User group num : " << mf_info->user_group_num;

    // cout  << "Item group num : " << mf_info->item_group_num;
    

    // mf_info->user_group_prec_info = new unsigned char[mf_info->user_group_num];
    // mf_info->item_group_prec_info = new unsigned char[mf_info->item_group_num];
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    // cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    // cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    // unsigned int block_num = mf_info->params.num_workers/div;
    // cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    // cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    // cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    // cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    // cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    // cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    // cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    // cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (2*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    double rmse;
    // map<unsigned int, vector<unsigned int>> user_switching_log;
    // map<unsigned int, vector<unsigned int>> item_switching_log;
    // vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    // vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
// #ifdef TEST
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_eval_only_indexing<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        
        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            if (mf_info->user_group_prec_info[i] == 0) {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
            else {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

        for (int i = 0; i < mf_info->max_user; i++){
            unsigned int user_group = mf_info->user_index_info[i].g;
            unsigned int row = mf_info->user_index_info[i].v;
            if (mf_info->user_group_prec_info[user_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
                }     
            }
            else {
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        for (int i = 0; i < mf_info->max_item; i++){
            unsigned int item_group = mf_info->item_index_info[i].g;
            unsigned int row = mf_info->item_index_info[i].v;
            if (mf_info->item_group_prec_info[item_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
                }
            }
            else{
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        // std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        // cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        // cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        // double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        // error_computation_time += error_transfer_time;
        
        // cout << "Transfer time " << error_transfer_time << endl;
        // cout << "\n<User groups>\n";
        // for (int i = 0; i < mf_info->user_group_num; i++){
        //     error_computation_start_time = std::chrono::system_clock::now();
        //     float each_group_grad_sum_norm_acc = 0;
        //     float each_group_norm_acc = 0;
        //     if (mf_info->user_group_prec_info[i] == 0){
                
        //         for (int k = 0; k < mf_info->params.k; k++){
        //             each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
        //             grad_sum_norm_p[i*mf_info->params.k + k] = 0;
        //         }
        //         for (int j = 0; j < block_num; j++){
        //             // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
        //             each_group_norm_acc += norm_sum_p[i * block_num + j];
        //         }
        //         // cout << each_group_grad_sum_norm_acc << " ";
        //         mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
        //     }
        //     else{
        //         mf_info->user_group_error[i] = UINT_MAX;
        //     }
        //     error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
        //     //! log code
        //     if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e >= start_idx) user_switching_log[e].push_back(i);

        //     if (mf_info->user_group_error[i] == UINT_MAX) {
        //         cout << "-1" << " ";
        //         user_grad_diversity_log[e][i] = -1;
        //     }           
        //     else {
        //         cout << mf_info->user_group_error[i] << " ";
        //         user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
        //     }
        // }

        // cout << "\n<Item groups>\n";
        // for (int i = 0; i < mf_info->item_group_num; i++){
        //     error_computation_start_time = std::chrono::system_clock::now();
        //     float each_group_grad_sum_norm_acc = 0;
        //     float each_group_norm_acc = 0;
        //     if (mf_info->item_group_prec_info[i] == 0){

        //         for (int k = 0; k < mf_info->params.k; k++){
        //             each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
        //             grad_sum_norm_q[i*mf_info->params.k + k] = 0;
        //         }

        //         for (int j = 0; j < block_num; j++){
        //             // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
        //             each_group_norm_acc += norm_sum_q[i * block_num + j];
        //         }
        //         mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
        //         // cout << each_group_grad_sum_norm_acc << " ";
        //     }
        //     else{
        //         mf_info->item_group_error[i] = UINT_MAX;
        //     }
        //     error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
        //     //! log code
        //     if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e >= start_idx) item_switching_log[e].push_back(i);
            
        //     if (mf_info->item_group_error[i] == UINT_MAX) {
        //         cout << "-1" << " ";
        //         item_grad_diversity_log[e][i] = -1;
        //     }            
        //     else {
        //         cout << mf_info->item_group_error[i] << " ";
        //         item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
        //     }
        // }
        // cout << "\n";
        // // error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        // std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        // // if (e >= start_idx) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        // precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
        // precision_switching_start_point = std::chrono::system_clock::now();
        // cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        // precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
    }

    // //! RMSE write code
    // string group_error_metric_output_file_path = string("./statistics/") + mf_info->out_file + ".txt";  
    // print_rmse(group_error_metric_output_file_path, rmse);
    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + generate_map_exec_time + cpy2grouped_parameters_exec_time;
    cout << "\n<Preprocessing time (micro sec)>" << endl;
    cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    cout << "Grouping                         : " << grouping_exec_time << endl;
    cout << "Generate map idx                 : " << generate_map_exec_time << endl;
    cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
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

    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    // string exec_rmse_output_file_path = string("./statistics/grouping/time_rmse/atomic_ver_time_rmse_") + mf_info->out_file + ".txt";  
    // string group_switching_log_output_file_path = string("./statistics/grouping/switching_log/atomic_ver_group_switching_log_") + mf_info->out_file + ".txt";
    // string group_diversity_log_output_file_path = string("./statistics/grouping/diversity_log/atomic_ver_group_diversity_log_") + mf_info->out_file + ".txt";
    
    // print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    // print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    // print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
// #endif
}

void grouped_sgd_training_map_based_grad_diversity_partial_group(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);


    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);
    
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    check_group_cnt(mf_info);
    // cout << "hey" << endl;
    //* Generating index
    double generate_map_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> generate_map_start_point = std::chrono::system_clock::now();
    generate_map_idx_info(mf_info);
    generate_map_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - generate_map_start_point).count();
    // cout << "hey2" << endl;

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    // sgd_info->user_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // sgd_info->user_group_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // cout << "hey2.5" << endl;
    // cout << "User group num : " << mf_info->user_group_num << endl;
    // cout << "Item group num : " << mf_info->item_group_num << endl;

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    // cpy2grouped_parameters(mf_info, sgd_info);
    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    cpy2grouped_parameters_gpu(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    // mf_info->user_group_prec_info = new unsigned char[mf_info->user_group_num];
    // mf_info->item_group_prec_info = new unsigned char[mf_info->item_group_num];
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    // size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned int)*mf_info->params.k*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    double rmse;
    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
    int start_idx = 0;

    //! Additional code section
    unsigned int cached_user_group_num = 0;
    unsigned int cached_item_group_num = 0;
    unsigned int uncached_user_group_num = mf_info->user_group_num - cached_user_group_num;
    unsigned int uncached_item_group_num = mf_info->item_group_num - cached_item_group_num;

    float* d_user_group_sum_updated_val;
    float* d_item_group_sum_updated_val;
    
    cudaMalloc(&d_user_group_sum_updated_val, sizeof(float) * uncached_user_group_num * mf_info->params.k);
    cudaMalloc(&d_item_group_sum_updated_val, sizeof(float) * uncached_item_group_num * mf_info->params.k);

    //! initialize to zeros
    unsigned int num_groups = 10000;


    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (sizeof(unsigned int)*mf_info->params.k*(cached_user_group_num + cached_item_group_num)) + 
                             (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;

        if ((e >= start_idx) && (e % mf_info->interval == (start_idx % mf_info->interval))) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            initialize_float_array_to_val<<<num_groups, 512>>>(d_user_group_sum_updated_val, uncached_user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_groups, 512>>>(d_item_group_sum_updated_val, uncached_item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_grouped_cache<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_grad_sum_norm_p,
                                                    d_grad_sum_norm_q,
                                                    d_norm_sum_p,
                                                    d_norm_sum_q,
                                                    d_user_group_sum_updated_val,
                                                    d_item_group_sum_updated_val,
                                                    first_sample_rating_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num,
                                                    uncached_user_group_num,
                                                    uncached_item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());  

        // error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_p, d_user_group_sum_updated_val, uncached_user_group_num * mf_info->params.k);
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_q, d_item_group_sum_updated_val, uncached_item_group_num * mf_info->params.k);
            cudaDeviceSynchronize();
        }
        // double error_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        // error_computation_time += error_copy_time;

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        // std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

        for (int i = 0; i < mf_info->max_user; i++){
            unsigned int user_group = mf_info->user_index_info[i].g;
            unsigned int row = mf_info->user_index_info[i].v;
            if (mf_info->user_group_prec_info[user_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
                }     
            }
            else {
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        for (int i = 0; i < mf_info->max_item; i++){
            unsigned int item_group = mf_info->item_index_info[i].g;
            unsigned int row = mf_info->item_index_info[i].v;
            if (mf_info->item_group_prec_info[item_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
                }
            }
            else{
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        // cout << "Transfer time " << error_transfer_time << endl;
        if (error_check){
            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    // cout << each_group_grad_sum_norm_acc << " ";
                    mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                }
                else{
                    mf_info->user_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e >= start_idx) user_switching_log[e].push_back(i);

                if (mf_info->user_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    user_grad_diversity_log[e][i] = -1;
                }           
                else {
                    cout << mf_info->user_group_error[i] << " ";
                    user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
                }
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    // cout << each_group_grad_sum_norm_acc << " ";
                }
                else{
                    mf_info->item_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e >= start_idx) item_switching_log[e].push_back(i);
                
                if (mf_info->item_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    item_grad_diversity_log[e][i] = -1;
                }            
                else {
                    cout << mf_info->item_group_error[i] << " ";
                    item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
                }
            }
            cout << "\n";
        }
        // error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        // if (error_check) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
         // //! RMSE write code
#ifdef WRITE_FILE
        string group_error_metric_output_file_path = string("./New_statistics/Gradient diversity fp16/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif      
        precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check){
            cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        }
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
    }

    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + generate_map_exec_time + cpy2grouped_parameters_exec_time;
    cout << "\n<Preprocessing time (micro sec)>" << endl;
    cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    cout << "Grouping                         : " << grouping_exec_time << endl;
    cout << "Generate map idx                 : " << generate_map_exec_time << endl;
    cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
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
#ifdef WRITE_FILE

    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    // string exec_rmse_output_file_path = string("./New_statistics/Gradient diversity/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    // string group_switching_log_output_file_path = string("./statistics/grouping/switching_log/group_switching_log_") + mf_info->out_file + ".txt";
    string group_diversity_log_output_file_path = string("./New_statistics/Gradient diversity fp16/diversity_log/group_diversity_log_fp16") + mf_info->out_file + ".txt";
    
    // print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    // print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
#endif
}
__global__ void warm_up_gpu(){
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float ia, ib;
  ia = ib = 0.0f;
  ib += ia + tid; 
}

void grouped_sgd_training_comparison_based_grad_diversity_partial_group_naive_version(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
// #ifdef WRITE_FILE
//     check_group_cnt(mf_info);
// #endif
    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

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

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_sum_updated_val, sizeof(float) * mf_info->user_group_num * mf_info->params.k);
    cudaMalloc(&d_item_group_sum_updated_val, sizeof(float) * mf_info->item_group_num * mf_info->params.k);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    unsigned int* user_group_end_idx_shift = new unsigned int[mf_info->user_group_num + 1];
    unsigned int* item_group_end_idx_shift = new unsigned int[mf_info->item_group_num + 1];
    user_group_end_idx_shift[0] = -1;
    item_group_end_idx_shift[0] = -1;

    for (int i = 0; i < mf_info->user_group_num; i++){
        user_group_end_idx_shift[i+1] = mf_info->user_group_end_idx[i];
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        item_group_end_idx_shift[i+1] = mf_info->item_group_end_idx[i];
    }

    unsigned int* d_user_group_end_idx_shift;
    unsigned int* d_item_group_end_idx_shift;

    cudaMalloc(&d_user_group_end_idx_shift, sizeof(unsigned int) * (mf_info->user_group_num + 1));
    cudaMalloc(&d_item_group_end_idx_shift, sizeof(unsigned int) * (mf_info->item_group_num + 1));
    cudaMemcpy(d_user_group_end_idx_shift, user_group_end_idx_shift, sizeof(unsigned int) * (mf_info->user_group_num + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_item_group_end_idx_shift, item_group_end_idx_shift, sizeof(unsigned int) * (mf_info->item_group_num + 1), cudaMemcpyHostToDevice);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    cout << "NUM user groups : " << mf_info->user_group_num << endl;
    cout << "NUM item groups : " << mf_info->item_group_num << endl;


    // ! Test
    // unsigned int declared_user_group_num = mf_info->user_group_num + 32 - (mf_info->user_group_num%32);
    // unsigned int declared_item_group_num = mf_info->item_group_num + 32 - (mf_info->item_group_num%32);

    // unsigned int declared_user_group_num2 = (mf_info->user_group_num + 1) + 32 -((mf_info->user_group_num + 1)%32);
    // unsigned int declared_item_group_num2 = (mf_info->item_group_num + 1) + 32 -((mf_info->item_group_num + 1)%32);
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(declared_user_group_num + declared_item_group_num)) + 
    //                          (sizeof(unsigned int)*(declared_user_group_num2 + declared_item_group_num2)) + 
    //                          (sizeof(unsigned char)*(declared_user_group_num + declared_item_group_num));
    
    // ! Original
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (2*sizeof(unsigned int)) +
    //                          (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
   
    size_t user_cache_size_comp = mf_info->user_group_num > 62 ? sizeof(unsigned int) * (mf_info->user_group_num - 61) : 0;
    size_t item_cache_size_comp = mf_info->item_group_num > 62 ? sizeof(unsigned int) * (mf_info->item_group_num - 61) : 0;
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) +
                              user_cache_size_comp +
                              item_cache_size_comp + 
                              (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
        
    float* initial_user_group_error = new float[mf_info->user_group_num];
    float* initial_item_group_error = new float[mf_info->item_group_num];

    for (int i = 0; i <mf_info->user_group_num; i++) initial_user_group_error[i] = 1.0f;
    for (int i = 0; i <mf_info->item_group_num; i++) initial_item_group_error[i] = 1.0f;

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;
        //! interval 50 and start idx = 0일 때 오동작
        if ((e >= start_idx ) && (e % mf_info->interval == (start_idx % mf_info->interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }
        // //! Switching 없애기 
        // error_check = false;
        // first_sample_rating_idx = (update_count * update_vector_size) - 0;

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            initialize_float_array_to_val<<<num_groups, 512>>>(d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_groups, 512>>>(d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;
    
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128){
            sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_info_on_device_mem<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda,
                                                        // mf_info->d_user_group_end_idx,
                                                        // mf_info->d_item_group_end_idx,
                                                        d_user_group_end_idx_shift,
                                                        d_item_group_end_idx_shift,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_grad_sum_norm_p,
                                                        d_grad_sum_norm_q,
                                                        d_norm_sum_p,
                                                        d_norm_sum_q,
                                                        d_user_group_sum_updated_val,
                                                        d_item_group_sum_updated_val,
                                                        first_sample_rating_idx,
                                                        (int)mf_info->user_group_num,
                                                        (int)mf_info->item_group_num
                                                        );
        }
        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_reg_cache<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );
        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );

        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        // cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;
        gpuErr(cudaPeekAtLastError());  
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_p, d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k);
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_q, d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k);
            cudaDeviceSynchronize();
        }
        double error_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_copy_time;

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        // std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();
        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        if (error_check && e != mf_info->params.epoch - 1){

            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    // cout << each_group_grad_sum_norm_acc << " ";
                    mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->user_group_error[i] /= initial_user_group_error[i];
                }
                else{
                    mf_info->user_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e > start_idx) user_switching_log[e].push_back(i);

                if (mf_info->user_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    user_grad_diversity_log[e][i] = -1;
                }           
                else {
                    cout << mf_info->user_group_error[i] << " ";
                    user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
                }
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->item_group_error[i] /= initial_item_group_error[i];
                    // cout << each_group_grad_sum_norm_acc << " ";
                }
                else{
                    mf_info->item_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e > start_idx) item_switching_log[e].push_back(i);
                
                if (mf_info->item_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    item_grad_diversity_log[e][i] = -1;
                }            
                else {
                    cout << mf_info->item_group_error[i] << " ";
                    item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
                }
            }
            cout << "\n";

            if (e == start_idx){
                for (int i = 0; i < mf_info->user_group_num; i++){
                    initial_user_group_error[i] = mf_info->user_group_error[i];
                }
                for (int i = 0; i < mf_info->item_group_num; i++){
                    initial_item_group_error[i] = mf_info->item_group_error[i];
                }
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check && e > start_idx && e != mf_info->params.epoch - 1) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
        // string group_error_metric_output_file_path = string("./statistics/grouping/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/optimization vs naive/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check && e != mf_info->params.epoch - 1){
            cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        }
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
// #endif
    }

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

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/optimization vs naive/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    string group_switching_log_output_file_path = string("./New_statistics/optimization vs naive/switching_log/group_switching_log_") + mf_info->out_file + ".txt";
    string group_diversity_log_output_file_path = string("./New_statistics/optimization vs naive/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
#endif
}

void grouped_sgd_training_comparison_based_grad_diversity_partial_group_naive_version_device_mem(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
// #ifdef WRITE_FILE
//     check_group_cnt(mf_info);
// #endif
    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

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

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_sum_updated_val, sizeof(float) * mf_info->user_group_num * mf_info->params.k);
    cudaMalloc(&d_item_group_sum_updated_val, sizeof(float) * mf_info->item_group_num * mf_info->params.k);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    unsigned int* user_group_end_idx_shift = new unsigned int[mf_info->user_group_num + 1];
    unsigned int* item_group_end_idx_shift = new unsigned int[mf_info->item_group_num + 1];
    user_group_end_idx_shift[0] = -1;
    item_group_end_idx_shift[0] = -1;

    for (int i = 0; i < mf_info->user_group_num; i++){
        user_group_end_idx_shift[i+1] = mf_info->user_group_end_idx[i];
    }

    for (int i = 0; i < mf_info->item_group_num; i++){
        item_group_end_idx_shift[i+1] = mf_info->item_group_end_idx[i];
    }

    unsigned int* d_user_group_end_idx_shift;
    unsigned int* d_item_group_end_idx_shift;

    cudaMalloc(&d_user_group_end_idx_shift, sizeof(unsigned int) * (mf_info->user_group_num + 1));
    cudaMalloc(&d_item_group_end_idx_shift, sizeof(unsigned int) * (mf_info->item_group_num + 1));
    cudaMemcpy(d_user_group_end_idx_shift, user_group_end_idx_shift, sizeof(unsigned int) * (mf_info->user_group_num + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_item_group_end_idx_shift, item_group_end_idx_shift, sizeof(unsigned int) * (mf_info->item_group_num + 1), cudaMemcpyHostToDevice);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    cout << "NUM user groups : " << mf_info->user_group_num << endl;
    cout << "NUM item groups : " << mf_info->item_group_num << endl;


    // ! Test
    // unsigned int declared_user_group_num = mf_info->user_group_num + 32 - (mf_info->user_group_num%32);
    // unsigned int declared_item_group_num = mf_info->item_group_num + 32 - (mf_info->item_group_num%32);

    // unsigned int declared_user_group_num2 = (mf_info->user_group_num + 1) + 32 -((mf_info->user_group_num + 1)%32);
    // unsigned int declared_item_group_num2 = (mf_info->item_group_num + 1) + 32 -((mf_info->item_group_num + 1)%32);
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(declared_user_group_num + declared_item_group_num)) + 
    //                          (sizeof(unsigned int)*(declared_user_group_num2 + declared_item_group_num2)) + 
    //                          (sizeof(unsigned char)*(declared_user_group_num + declared_item_group_num));
    
    // ! Original
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (2*sizeof(unsigned int)) +
    //                          (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
   
    size_t user_cache_size_comp = mf_info->user_group_num > 62 ? sizeof(unsigned int) * (mf_info->user_group_num - 61) : 0;
    size_t item_cache_size_comp = mf_info->item_group_num > 62 ? sizeof(unsigned int) * (mf_info->item_group_num - 61) : 0;
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) +
                              user_cache_size_comp +
                              item_cache_size_comp + 
                              (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
        
    float* initial_user_group_error = new float[mf_info->user_group_num];
    float* initial_item_group_error = new float[mf_info->item_group_num];

    for (int i = 0; i <mf_info->user_group_num; i++) initial_user_group_error[i] = 1.0f;
    for (int i = 0; i <mf_info->item_group_num; i++) initial_item_group_error[i] = 1.0f;

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;
        //! interval 50 and start idx = 0일 때 오동작
        if ((e >= start_idx ) && (e % mf_info->interval == (start_idx % mf_info->interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }
        // //! Switching 없애기 
        // error_check = false;
        // first_sample_rating_idx = (update_count * update_vector_size) - 0;

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            initialize_float_array_to_val<<<num_groups, 512>>>(d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_groups, 512>>>(d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;
    
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128){
            sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_info_on_device_mem_search_order_opt<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count,
                                                        update_vector_size,
                                                        mf_info->params.lambda,
                                                        // mf_info->d_user_group_end_idx,
                                                        // mf_info->d_item_group_end_idx,
                                                        d_user_group_end_idx_shift,
                                                        d_item_group_end_idx_shift,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_grad_sum_norm_p,
                                                        d_grad_sum_norm_q,
                                                        d_norm_sum_p,
                                                        d_norm_sum_q,
                                                        d_user_group_sum_updated_val,
                                                        d_item_group_sum_updated_val,
                                                        first_sample_rating_idx,
                                                        (int)mf_info->user_group_num,
                                                        (int)mf_info->item_group_num
                                                        );
        }
        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_reg_cache<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );
        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );

        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        // cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;
        gpuErr(cudaPeekAtLastError());  
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_p, d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k);
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_q, d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k);
            cudaDeviceSynchronize();
        }
        double error_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_copy_time;

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        // std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();
        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        if (error_check && e != mf_info->params.epoch - 1){

            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    // cout << each_group_grad_sum_norm_acc << " ";
                    mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->user_group_error[i] /= initial_user_group_error[i];
                }
                else{
                    mf_info->user_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e > start_idx) user_switching_log[e].push_back(i);

                if (mf_info->user_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    user_grad_diversity_log[e][i] = -1;
                }           
                else {
                    cout << mf_info->user_group_error[i] << " ";
                    user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
                }
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->item_group_error[i] /= initial_item_group_error[i];
                    // cout << each_group_grad_sum_norm_acc << " ";
                }
                else{
                    mf_info->item_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e > start_idx) item_switching_log[e].push_back(i);
                
                if (mf_info->item_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    item_grad_diversity_log[e][i] = -1;
                }            
                else {
                    cout << mf_info->item_group_error[i] << " ";
                    item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
                }
            }
            cout << "\n";

            if (e == start_idx){
                for (int i = 0; i < mf_info->user_group_num; i++){
                    initial_user_group_error[i] = mf_info->user_group_error[i];
                }
                for (int i = 0; i < mf_info->item_group_num; i++){
                    initial_item_group_error[i] = mf_info->item_group_error[i];
                }
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check && e > start_idx && e != mf_info->params.epoch - 1) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
        // string group_error_metric_output_file_path = string("./statistics/grouping/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/optimization vs naive/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check && e != mf_info->params.epoch - 1){
            cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        }
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
// #endif
    }

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

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/optimization vs naive/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    string group_switching_log_output_file_path = string("./New_statistics/optimization vs naive/switching_log/group_switching_log_") + mf_info->out_file + ".txt";
    string group_diversity_log_output_file_path = string("./New_statistics/optimization vs naive/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
#endif
}

void grouped_sgd_training_comparison_based_grad_diversity_partial_group(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
// #ifdef WRITE_FILE
//     check_group_cnt(mf_info);
// #endif
    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

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
    
    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_sum_updated_val, sizeof(float) * mf_info->user_group_num * mf_info->params.k);
    cudaMalloc(&d_item_group_sum_updated_val, sizeof(float) * mf_info->item_group_num * mf_info->params.k);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    cout << "NUM user groups : " << mf_info->user_group_num << endl;
    cout << "NUM item groups : " << mf_info->item_group_num << endl;


    // ! Test
    // unsigned int declared_user_group_num = mf_info->user_group_num + 32 - (mf_info->user_group_num%32);
    // unsigned int declared_item_group_num = mf_info->item_group_num + 32 - (mf_info->item_group_num%32);

    // unsigned int declared_user_group_num2 = (mf_info->user_group_num + 1) + 32 -((mf_info->user_group_num + 1)%32);
    // unsigned int declared_item_group_num2 = (mf_info->item_group_num + 1) + 32 -((mf_info->item_group_num + 1)%32);
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(declared_user_group_num + declared_item_group_num)) + 
    //                          (sizeof(unsigned int)*(declared_user_group_num2 + declared_item_group_num2)) + 
    //                          (sizeof(unsigned char)*(declared_user_group_num + declared_item_group_num));
    
    // ! Original
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (2*sizeof(unsigned int)) +
    //                          (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
   
    size_t user_cache_size_comp = mf_info->user_group_num > 62 ? sizeof(unsigned int) * (mf_info->user_group_num - 61) : 0;
    size_t item_cache_size_comp = mf_info->item_group_num > 62 ? sizeof(unsigned int) * (mf_info->item_group_num - 61) : 0;
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) +
                              user_cache_size_comp +
                              item_cache_size_comp + 
                              (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
        
    float* initial_user_group_error = new float[mf_info->user_group_num];
    float* initial_item_group_error = new float[mf_info->item_group_num];

    for (int i = 0; i <mf_info->user_group_num; i++) initial_user_group_error[i] = 1.0f;
    for (int i = 0; i <mf_info->item_group_num; i++) initial_item_group_error[i] = 1.0f;

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;
        //! interval 50 and start idx = 0일 때 오동작
        if ((e >= start_idx ) && (e % mf_info->interval == (start_idx % mf_info->interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }
        // //! Switching 없애기 
        // error_check = false;
        // first_sample_rating_idx = (update_count * update_vector_size) - 0;

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            initialize_float_array_to_val<<<num_groups, 512>>>(d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_groups, 512>>>(d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;
    
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        if (mf_info->params.k == 128){
            sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_64reg_cache<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
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
                                                        d_user_group_sum_updated_val,
                                                        d_item_group_sum_updated_val,
                                                        first_sample_rating_idx,
                                                        (int)mf_info->user_group_num,
                                                        (int)mf_info->item_group_num
                                                        );
        }else if (mf_info->params.k == 64){
            sgd_k64_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_64reg_cache<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
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
                                                        d_user_group_sum_updated_val,
                                                        d_item_group_sum_updated_val,
                                                        first_sample_rating_idx,
                                                        (int)mf_info->user_group_num,
                                                        (int)mf_info->item_group_num
                                                        );
        }
        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_reg_cache<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );
        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );

        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        // cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;
        gpuErr(cudaPeekAtLastError());  
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_p, d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k);
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_q, d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k);
            cudaDeviceSynchronize();
        }
        double error_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_copy_time;

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        // std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();
        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        if (error_check && e != mf_info->params.epoch - 1){

            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    // cout << each_group_grad_sum_norm_acc << " ";
                    mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->user_group_error[i] /= initial_user_group_error[i];
                }
                else{
                    mf_info->user_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e > start_idx) user_switching_log[e].push_back(i);

                if (mf_info->user_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    user_grad_diversity_log[e][i] = -1;
                }           
                else {
                    cout << mf_info->user_group_error[i] << " ";
                    user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
                }
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->item_group_error[i] /= initial_item_group_error[i];
                    // cout << each_group_grad_sum_norm_acc << " ";
                }
                else{
                    mf_info->item_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e > start_idx) item_switching_log[e].push_back(i);
                
                if (mf_info->item_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    item_grad_diversity_log[e][i] = -1;
                }            
                else {
                    cout << mf_info->item_group_error[i] << " ";
                    item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
                }
            }
            cout << "\n";

            if (e == start_idx){
                for (int i = 0; i < mf_info->user_group_num; i++){
                    initial_user_group_error[i] = mf_info->user_group_error[i];
                }
                for (int i = 0; i < mf_info->item_group_num; i++){
                    initial_item_group_error[i] = mf_info->item_group_error[i];
                }
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check && e > start_idx && e != mf_info->params.epoch - 1) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
        // string group_error_metric_output_file_path = string("./statistics/grouping/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/fin_mod_ver_val_set/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check && e != mf_info->params.epoch - 1){
            cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        }
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
// #endif
    }

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

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/fin_mod_ver_val_set/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    string group_switching_log_output_file_path = string("./New_statistics/fin_mod_ver_val_set/switching_log/group_switching_log_") + mf_info->out_file + ".txt";
    string group_diversity_log_output_file_path = string("./New_statistics/fin_mod_ver_val_set/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
#endif
}

void grouped_sgd_training_comparison_based_grad_diversity_partial_group_cur_version_eval_indexing(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    // random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
// #ifdef WRITE_FILE
//     check_group_cnt(mf_info);
// #endif
    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    cout << "NUM user groups : " << mf_info->user_group_num << endl;
    cout << "NUM item groups : " << mf_info->item_group_num << endl;
   
    size_t user_cache_size_comp = mf_info->user_group_num > 62 ? sizeof(unsigned int) * (mf_info->user_group_num - 61) : 0;
    size_t item_cache_size_comp = mf_info->item_group_num > 62 ? sizeof(unsigned int) * (mf_info->item_group_num - 61) : 0;
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) +
                              user_cache_size_comp +
                              item_cache_size_comp + 
                              (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    __half lambda_half = __float2half_rn(mf_info->params.lambda);

    for (int e = 0; e < mf_info->params.epoch; e++){
    
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_64reg_cache_eval_indexing<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_group_end_idx,
                                                    mf_info->d_item_group_end_idx,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    (int)mf_info->user_group_num,
                                                    (int)mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        // cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;
        gpuErr(cudaPeekAtLastError());  

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        // std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();
        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << sgd_update_execution_time/1000 << endl;

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = 0;
    statistics_map["switching"] = 0;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = sgd_update_execution_time / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/indexing_reg_version/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
#endif
}

void grouped_sgd_training_comparison_based_grad_diversity_partial_group_time_check_per_area(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
// #ifdef WRITE_FILE
//     check_group_cnt(mf_info);
// #endif
    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

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
    
    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_sum_updated_val, sizeof(float) * mf_info->user_group_num * mf_info->params.k);
    cudaMalloc(&d_item_group_sum_updated_val, sizeof(float) * mf_info->item_group_num * mf_info->params.k);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    cout << "NUM user groups : " << mf_info->user_group_num << endl;
    cout << "NUM item groups : " << mf_info->item_group_num << endl;


    // ! Test
    // unsigned int declared_user_group_num = mf_info->user_group_num + 32 - (mf_info->user_group_num%32);
    // unsigned int declared_item_group_num = mf_info->item_group_num + 32 - (mf_info->item_group_num%32);

    // unsigned int declared_user_group_num2 = (mf_info->user_group_num + 1) + 32 -((mf_info->user_group_num + 1)%32);
    // unsigned int declared_item_group_num2 = (mf_info->item_group_num + 1) + 32 -((mf_info->item_group_num + 1)%32);
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(declared_user_group_num + declared_item_group_num)) + 
    //                          (sizeof(unsigned int)*(declared_user_group_num2 + declared_item_group_num2)) + 
    //                          (sizeof(unsigned char)*(declared_user_group_num + declared_item_group_num));
    
    // ! Original
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (2*sizeof(unsigned int)) +
    //                          (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
   
    size_t user_cache_size_comp = mf_info->user_group_num > 62 ? sizeof(unsigned int) * (mf_info->user_group_num - 61) : 0;
    size_t item_cache_size_comp = mf_info->item_group_num > 62 ? sizeof(unsigned int) * (mf_info->item_group_num - 61) : 0;
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) +
                              user_cache_size_comp +
                              item_cache_size_comp + 
                              (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
        
    float* initial_user_group_error = new float[mf_info->user_group_num];
    float* initial_item_group_error = new float[mf_info->item_group_num];

    for (int i = 0; i <mf_info->user_group_num; i++) initial_user_group_error[i] = 1.0f;
    for (int i = 0; i <mf_info->item_group_num; i++) initial_item_group_error[i] = 1.0f;

    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;
        //! interval 50 and start idx = 0일 때 오동작
        if ((e >= start_idx ) && (e % mf_info->interval == (start_idx % mf_info->interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }
        // //! Switching 없애기 
        // error_check = false;
        // first_sample_rating_idx = (update_count * update_vector_size) - 0;

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            initialize_float_array_to_val<<<num_groups, 512>>>(d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_groups, 512>>>(d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;
    
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_64reg_cache_time_check_per_area<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
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
                                                    d_user_group_sum_updated_val,
                                                    d_item_group_sum_updated_val,
                                                    first_sample_rating_idx,
                                                    (int)mf_info->user_group_num,
                                                    (int)mf_info->item_group_num
                                                    );

        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_reg_cache<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );
        // sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
        //                                             mf_info->d_R,
        //                                             mf_info->n,
        //                                             (void**)sgd_info->d_user_group_ptr,
        //                                             (void**)sgd_info->d_item_group_ptr,
        //                                             d_rand_state,
        //                                             lr_decay_arr[e],
        //                                             mf_info->params.k,
        //                                             1,
        //                                             e,
        //                                             update_count,
        //                                             update_vector_size,
        //                                             mf_info->params.lambda,
        //                                             mf_info->d_user_group_end_idx,
        //                                             mf_info->d_item_group_end_idx,
        //                                             mf_info->d_user_group_prec_info,
        //                                             mf_info->d_item_group_prec_info,
        //                                             d_grad_sum_norm_p,
        //                                             d_grad_sum_norm_q,
        //                                             d_norm_sum_p,
        //                                             d_norm_sum_q,
        //                                             d_user_group_sum_updated_val,
        //                                             d_item_group_sum_updated_val,
        //                                             first_sample_rating_idx,
        //                                             (int)mf_info->user_group_num,
        //                                             (int)mf_info->item_group_num
        //                                             );

        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        // cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;
        gpuErr(cudaPeekAtLastError());  
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_p, d_user_group_sum_updated_val, mf_info->user_group_num * mf_info->params.k);
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_q, d_item_group_sum_updated_val, mf_info->item_group_num * mf_info->params.k);
            cudaDeviceSynchronize();
        }
        double error_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_copy_time;

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        // std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();
        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        if (error_check && e != mf_info->params.epoch - 1){

            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    // cout << each_group_grad_sum_norm_acc << " ";
                    mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->user_group_error[i] /= initial_user_group_error[i];
                }
                else{
                    mf_info->user_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e > start_idx) user_switching_log[e].push_back(i);

                if (mf_info->user_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    user_grad_diversity_log[e][i] = -1;
                }           
                else {
                    cout << mf_info->user_group_error[i] << " ";
                    user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
                }
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    mf_info->item_group_error[i] /= initial_item_group_error[i];
                    // cout << each_group_grad_sum_norm_acc << " ";
                }
                else{
                    mf_info->item_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e > start_idx) item_switching_log[e].push_back(i);
                
                if (mf_info->item_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    item_grad_diversity_log[e][i] = -1;
                }            
                else {
                    cout << mf_info->item_group_error[i] << " ";
                    item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
                }
            }
            cout << "\n";

            if (e == start_idx){
                for (int i = 0; i < mf_info->user_group_num; i++){
                    initial_user_group_error[i] = mf_info->user_group_error[i];
                }
                for (int i = 0; i < mf_info->item_group_num; i++){
                    initial_item_group_error[i] = mf_info->item_group_error[i];
                }
            }
        }
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check && e > start_idx && e != mf_info->params.epoch - 1) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
        // string group_error_metric_output_file_path = string("./statistics/grouping/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/fin_mod_ver/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif
        error_computation_start_time = std::chrono::system_clock::now();
        if (error_check && e != mf_info->params.epoch - 1){
            cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        }
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
// #endif
    }

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

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/fin_mod_ver/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    string group_switching_log_output_file_path = string("./New_statistics/fin_mod_ver/switching_log/group_switching_log_") + mf_info->out_file + ".txt";
    string group_diversity_log_output_file_path = string("./New_statistics/fin_mod_ver/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
#endif
}


void grouped_sgd_training_grad_diversity_not_group_only_switching(Mf_info* mf_info, SGD* sgd_info){
    //* Random shuffle and transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    //! version == 12에서도 해당 부분 지우기
    // cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    // cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    float quantization_error = 0;
    //! 정보 하나로 통합하기
    // cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    // cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *sum_norms;
    float *d_sum_norms;
    float *sum_updated_val;
    float *d_sum_updated_val;
    
    unsigned int block_num = mf_info->params.num_workers/div;
    //! 전부 하나로 통합
    cudaMalloc(&d_sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMallocHost(&sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMalloc(&d_sum_updated_val, sizeof(float) * mf_info->params.k);
    cudaMallocHost(&sum_updated_val, sizeof(float) * mf_info->params.k);

    double precision_switching_and_error_comp_execution_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    //! 추후에 커널 코드 작성 시 수정하기
    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (2*sizeof(unsigned int)) +
                             (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    // map<unsigned int, vector<unsigned int>> user_switching_log;
    // map<unsigned int, vector<unsigned int>> item_switching_log;
    // vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    // vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));

    float initial_error = 1; 
    // float* initial_user_group_error = new float[mf_info->user_group_num];
    // float* initial_item_group_error = new float[mf_info->item_group_num];
    //! 시간 더해야 함... 추후 실험에서
    cudaMalloc(&sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k);
    // for (int i = 0; i <mf_info->user_group_num; i++) initial_user_group_error[i] = 1.0f;
    // for (int i = 0; i <mf_info->item_group_num; i++) initial_item_group_error[i] = 1.0f;
    void* p = (void*)(sgd_info->d_half_p);
    void* q = (void*)(sgd_info->d_half_q);
    unsigned char cur_precision = 0;//# 0 => half, 1 => single

    vector<vector<double>> grad_diversity_log(mf_info->params.epoch, vector<double>(1, 0));
    map<unsigned int, vector<unsigned int>> switching_log;
    int switching_point = 0;
    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;
        if (!cur_precision &&(e >= start_idx ) && (e % mf_info->interval == (start_idx % mf_info->interval)) && mf_info->params.epoch - 1 != e) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }
        // 초기화하기
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grad_diversity_not_grouped_only_switching<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                            mf_info->d_R,
                            mf_info->n,
                            (void*)p,
                            (void*)q,
                            d_rand_state,
                            lr_decay_arr[e],
                            mf_info->params.k,
                            1,
                            e,
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
        cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;

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

        std::chrono::time_point<std::chrono::system_clock> precision_switching_and_error_comp_start_time = std::chrono::system_clock::now();
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

            quantization_error = norm_acc/(float)sum_norms_acc;
            quantization_error = quantization_error/(float)initial_error;
            
            cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
            cout << "Quantization error : " << quantization_error << endl;
            
            if (e == start_idx) initial_error = quantization_error;
            if (e > start_idx) grad_diversity_log[e][0] = quantization_error;

            //! Switching
            if (e > start_idx && quantization_error < mf_info->error_threshold) {
                transition_params_half2float(mf_info, sgd_info);
                cur_precision = 1;
                p = (void*)(sgd_info->d_p);
                q = (void*)(sgd_info->d_q);
                cudaFree(sgd_info->d_half_p);
                cudaFree(sgd_info->d_half_q);
                switching_point = e;
            }
        }
        
        if(cur_precision && e != switching_point) grad_diversity_log[e][0] = -1;

        precision_switching_and_error_comp_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_and_error_comp_start_time).count();
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/only_switching_val_set/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif

    }
#ifdef WRITE_FILE

    map<string, double> statistics_map;

    statistics_map["preprocess"] = 0.0;
    statistics_map["switching"] = (precision_switching_and_error_comp_execution_time) / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (precision_switching_and_error_comp_execution_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/only_switching_val_set/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    string group_diversity_log_output_file_path = string("./New_statistics/only_switching_val_set/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";

    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    print_grad_diversity_log_not_grouping(group_diversity_log_output_file_path, grad_diversity_log);
#endif

    cout << "Total error comp & precision switching time     : " << precision_switching_and_error_comp_execution_time << endl;
    cout << "Parameters update per epoch                     : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update                         : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                               : " << precision_switching_and_error_comp_execution_time + sgd_update_execution_time << endl;
}

//! Average of user and item grad diversity version
// void grouped_sgd_training_grad_diversity_not_group_only_switching(Mf_info* mf_info, SGD* sgd_info){
//     //* Random shuffle and transfer rating triplets to GPU 
//     random_shuffle(mf_info->R, mf_info->R + mf_info->n);

//     cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
//     cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
//     gpuErr(cudaPeekAtLastError());
    
//     //* Convert testset to COO format
//     mf_info->test_COO = test_set_preprocess(mf_info);
//     cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
//     gpuErr(cudaPeekAtLastError());
//     cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
//     //* Initialize random states
//     curandState* d_rand_state;
//     cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
//     init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
//     cudaDeviceSynchronize();
//     gpuErr(cudaPeekAtLastError());

//     //* Learning rate initialization
//     float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
//     for (int i = 0; i < mf_info->params.epoch; i++){
//         lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
//     }
    
//     int update_vector_size = 128;
//     int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
//     int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
//     unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
//     cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
//     cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
//     cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
//     cout << "Start SGD update..." << endl;
    
//     unsigned int div = mf_info->params.thread_block_size/32;
    
//     unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
//     unsigned int group_error_size = error_kernel_work_groups;
//     unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
//     unsigned int seg_size = 32;
//     float* d_e_group;
//     cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

//     //! version == 12에서도 해당 부분 지우기
//     // cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
//     // cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

//     double additional_info_init_exec_time = 0;
//     std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
//     float quantization_error = 0;
//     //! 정보 하나로 통합하기
//     // cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
//     // cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

//     float *sum_norms;
//     float *sum_norms_item; 
    
//     float *d_sum_norms;
//     float *d_sum_norms_item;

//     float *sum_updated_val;
//     float *sum_updated_val_item;

//     float *d_sum_updated_val;
//     float *d_sum_updated_val_item;

//     unsigned int block_num = mf_info->params.num_workers/div;
//     //! 전부 하나로 통합
//     cudaMalloc(&d_sum_norms, sizeof(float) * mf_info->params.num_workers);
//     cudaMalloc(&d_sum_norms_item, sizeof(float) * mf_info->params.num_workers);

//     cudaMallocHost(&sum_norms, sizeof(float) * mf_info->params.num_workers);
//     cudaMallocHost(&sum_norms_item, sizeof(float) * mf_info->params.num_workers);

//     cudaMalloc(&d_sum_updated_val, sizeof(float) * mf_info->params.k);
//     cudaMalloc(&d_sum_updated_val_item, sizeof(float) * mf_info->params.k);

//     cudaMallocHost(&sum_updated_val, sizeof(float) * mf_info->params.k);
//     cudaMallocHost(&sum_updated_val_item, sizeof(float) * mf_info->params.k);

//     double precision_switching_and_error_comp_execution_time = 0;
//     double sgd_update_execution_time = 0;
//     double rmse;    
//     int start_idx = 5;
//     unsigned int num_groups = 10000;

//     //! 추후에 커널 코드 작성 시 수정하기
//     size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
//                              (2*sizeof(unsigned int)) +
//                              (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
//                              (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
//     // map<unsigned int, vector<unsigned int>> user_switching_log;
//     // map<unsigned int, vector<unsigned int>> item_switching_log;
//     // vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
//     // vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));

//     float initial_error = 1; 
//     // float* initial_user_group_error = new float[mf_info->user_group_num];
//     // float* initial_item_group_error = new float[mf_info->item_group_num];
//     //! 시간 더해야 함... 추후 실험에서
//     cudaMalloc(&sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k);
//     cudaMalloc(&sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k);
//     // for (int i = 0; i <mf_info->user_group_num; i++) initial_user_group_error[i] = 1.0f;
//     // for (int i = 0; i <mf_info->item_group_num; i++) initial_item_group_error[i] = 1.0f;
//     void* p = (void*)(sgd_info->d_half_p);
//     void* q = (void*)(sgd_info->d_half_q);
//     unsigned char cur_precision = 0;//# 0 => half, 1 => single

//     vector<vector<double>> grad_diversity_log(mf_info->params.epoch, vector<double>(1, 0));
//     map<unsigned int, vector<unsigned int>> switching_log;
//     int switching_point = 0;
//     for (int e = 0; e < mf_info->params.epoch; e++){
//         bool error_check = false;
//         first_sample_rating_idx = (update_count * update_vector_size) - 0;
//         if (!cur_precision &&(e >= start_idx ) && (e % mf_info->interval == (start_idx % mf_info->interval)) && mf_info->params.epoch - 1 != e) {
//             error_check = true;
//             first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
//         }
//         // 초기화하기
//         std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
//         sgd_k128_kernel_hogwild_warp32_lrate_fp16_grad_diversity_not_grouped_only_switching<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
//                             mf_info->d_R,
//                             mf_info->n,
//                             (void*)p,
//                             (void*)q,
//                             d_rand_state,
//                             lr_decay_arr[e],
//                             mf_info->params.k,
//                             1,
//                             e,
//                             update_count,
//                             update_vector_size,
//                             mf_info->params.lambda,
//                             cur_precision,
//                             first_sample_rating_idx,
//                             d_sum_updated_val,
//                             d_sum_norms,
//                             d_sum_updated_val_item,
//                             d_sum_norms_item
//                         );
//         cudaDeviceSynchronize();
//         double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
//         sgd_update_execution_time += sgd_update_time_per_epoch;
//         cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;

//         gpuErr(cudaPeekAtLastError());
//         if (!cur_precision){
//             cudaMemcpy(sgd_info->half_p, sgd_info->d_half_p, sizeof(half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
//             cudaMemcpy(sgd_info->half_q, sgd_info->d_half_q, sizeof(half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);

//             transform_feature_vector_half2float(sgd_info->half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
//             transform_feature_vector_half2float(sgd_info->half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
//         }else{
//             cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
//             cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
//         }

//         std::chrono::time_point<std::chrono::system_clock> precision_switching_and_error_comp_start_time = std::chrono::system_clock::now();
//         if (error_check && e != mf_info->params.epoch - 1){

//             cudaMemcpy(sum_norms, d_sum_norms, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
//             cudaMemcpy(sum_updated_val, d_sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);

//             cudaMemcpy(sum_norms_item, d_sum_norms_item, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
//             cudaMemcpy(sum_updated_val_item, d_sum_updated_val_item, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);

//             float sum_norms_acc = 0;
//             float sum_norms_acc_item = 0;
//             float norm_acc = 0 ;
//             float norm_acc_item = 0 ;

//             for (int w = 0; w < mf_info->params.num_workers; w++){
//                 norm_acc += sum_norms[w];
//                 norm_acc_item += sum_norms_item[w];
//             }

//             for (int k = 0; k < mf_info->params.k; k++){
//                 sum_norms_acc += powf(sum_updated_val[k],2);
//                 sum_norms_acc_item += powf(sum_updated_val_item[k],2);
//                 sum_updated_val[k] = 0.0f;
//                 sum_updated_val_item[k] = 0.0f;
//             }

//             quantization_error = ((norm_acc/(float)sum_norms_acc) + (norm_acc_item/(float)sum_norms_acc_item))/2.0f;
//             quantization_error = quantization_error/(float)initial_error;
            
//             cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
//             cudaMemcpy(d_sum_updated_val_item, sum_updated_val_item, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);

//             cout << "Quantization error : " << quantization_error << endl;
            
//             if (e == start_idx) initial_error = quantization_error;
//             if (e > start_idx) grad_diversity_log[e][0] = quantization_error;

//             //! Switching
//             if (e > start_idx && quantization_error < mf_info->error_threshold) {
//                 transition_params_half2float(mf_info, sgd_info);
//                 cur_precision = 1;
//                 p = (void*)(sgd_info->d_p);
//                 q = (void*)(sgd_info->d_q);
//                 cudaFree(sgd_info->d_half_p);
//                 cudaFree(sgd_info->d_half_q);
//                 switching_point = e;
//             }
//         }
        
//         if(cur_precision && e != switching_point) grad_diversity_log[e][0] = -1;

//         precision_switching_and_error_comp_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_and_error_comp_start_time).count();
//         rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
//         cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

// #ifdef WRITE_FILE
//         //! RMSE write code
//         string group_error_metric_output_file_path = string("./New_statistics/only_switching/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
//         print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
// #endif

//     }
// #ifdef WRITE_FILE

//     map<string, double> statistics_map;

//     statistics_map["preprocess"] = 0.0;
//     statistics_map["switching"] = (precision_switching_and_error_comp_execution_time) / 1000;
//     statistics_map["update"] = sgd_update_execution_time / 1000;
//     statistics_map["total"] = (precision_switching_and_error_comp_execution_time + sgd_update_execution_time) / 1000;
//     statistics_map["rmse"] = rmse;

//     string exec_rmse_output_file_path = string("./New_statistics/only_switching/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
//     string group_diversity_log_output_file_path = string("./New_statistics/only_switching/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";

//     print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
//     print_grad_diversity_log_not_grouping(group_diversity_log_output_file_path, grad_diversity_log);
// #endif

//     cout << "Total error comp & precision switching time     : " << precision_switching_and_error_comp_execution_time << endl;
//     cout << "Parameters update per epoch                     : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
//     cout << "Total parameters update                         : " << sgd_update_execution_time << endl; 
//     cout << "Total MF time(ms)                               : " << precision_switching_and_error_comp_execution_time + sgd_update_execution_time << endl;
// }

void grouped_sgd_training_grad_diversity_not_switching_only_grouping(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    cout << "NUM user groups : " << mf_info->user_group_num << endl;
    cout << "NUM item groups : " << mf_info->item_group_num << endl;

    double sgd_update_execution_time = 0;
    double rmse;    
    unsigned int num_groups = 10000;
    // ! 커널 수정 후 바꾸기
    // size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (2*sizeof(unsigned int)) +
    //                          (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    size_t user_cache_size_comp = mf_info->user_group_num > 62 ? sizeof(unsigned int) * (mf_info->user_group_num - 61) : 0;
    size_t item_cache_size_comp = mf_info->item_group_num > 62 ? sizeof(unsigned int) * (mf_info->item_group_num - 61) : 0;
    size_t shared_mem_size = (2*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) +
                              user_cache_size_comp +
                              item_cache_size_comp + 
                              (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
    
    unsigned int user_fp32_tail_groups_num = mf_info->fp32_user_last_n_group;
    unsigned int item_fp32_tail_groups_num = mf_info->fp32_item_last_n_group;
    
    if (mf_info->user_group_num < user_fp32_tail_groups_num) user_fp32_tail_groups_num = mf_info->user_group_num;
    if (mf_info->item_group_num < item_fp32_tail_groups_num) item_fp32_tail_groups_num = mf_info->item_group_num;
    
    for (int i = 0; i < mf_info->user_group_num; i++) {
        if (i >= mf_info->user_group_num - user_fp32_tail_groups_num) mf_info->user_group_error[i] = mf_info->error_threshold - 1.0f;
        else mf_info->user_group_error[i] = mf_info->error_threshold + 1.0f;
    }

    for (int i = 0; i < mf_info->item_group_num; i++) {
        if (i >= mf_info->item_group_num - item_fp32_tail_groups_num) mf_info->item_group_error[i] = mf_info->error_threshold - 1.0f;
        else mf_info->item_group_error[i] = mf_info->error_threshold + 1.0f;
    }

    precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
    
    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grad_diversity_not_switching_only_grouping<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    mf_info->params.lambda,
                                                    mf_info->d_user_group_end_idx,
                                                    mf_info->d_item_group_end_idx,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());  

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/fin_only_grouping/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif
    }

    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + reconst_exec_time + cpy2grouped_parameters_exec_time + additional_info_init_exec_time;

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time;
    statistics_map["switching"] = 0.0;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;
    statistics_map["total_user_groups"] = mf_info->user_group_num; 
    statistics_map["total_item_groups"] = mf_info->item_group_num;
    statistics_map["fp32_user_groups"] = user_fp32_tail_groups_num;
    statistics_map["fp32_item_groups"] = item_fp32_tail_groups_num;

    string exec_rmse_output_file_path = string("./New_statistics/fin_only_grouping/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    print_exec_time_and_rmse_group_only_version(exec_rmse_output_file_path, statistics_map);
#endif

    cout << "\n<Preprocessing time (micro sec)> " << endl;
    cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    cout << "Grouping                         : " << grouping_exec_time << endl;
    cout << "Matrix reconstruction            : " << reconst_exec_time << endl;
    cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
    cout << "Additional info init             : " << additional_info_init_exec_time << endl;
    cout << "Total preprocessing time         : " << preprocess_exec_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (preprocess_exec_time + sgd_update_execution_time)/1000 << endl;

}

void grouped_sgd_training_random_select_user_item(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    mf_info->user_group_num = 2;
    mf_info->item_group_num = 2;

    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_shuffle(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    // // //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    split_user_item_ratio(mf_info);
    // // if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    // // else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    //* MAT RECONSTRUCTION ON DEVICE
    // //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    cout << "NUM user groups : " << mf_info->user_group_num << endl;
    cout << "NUM item groups : " << mf_info->item_group_num << endl;

    double sgd_update_execution_time = 0;
    double rmse;    
    unsigned int num_groups = 10000;
    
    // ! 커널 수정 후 바꾸기
    // size_t shared_mem_size = 2*sizeof(unsigned int) * (mf_info->user_group_num + mf_info->item_group_num);
    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
    
    // unsigned int user_fp32_tail_groups_num = mf_info->fp32_user_last_n_group;
    // unsigned int item_fp32_tail_groups_num = mf_info->fp32_item_last_n_group;
    
    // if (mf_info->user_group_num < user_fp32_tail_groups_num) user_fp32_tail_groups_num = mf_info->user_group_num;
    // if (mf_info->item_group_num < item_fp32_tail_groups_num) item_fp32_tail_groups_num = mf_info->item_group_num;
    
    // for (int i = 0; i < mf_info->user_group_num; i++) {
    //     if (i >= mf_info->user_group_num - user_fp32_tail_groups_num) mf_info->user_group_error[i] = mf_info->error_threshold - 1.0f;
    //     else mf_info->user_group_error[i] = mf_info->error_threshold + 1.0f;
    // }

    // for (int i = 0; i < mf_info->item_group_num; i++) {
    //     if (i >= mf_info->item_group_num - item_fp32_tail_groups_num) mf_info->item_group_error[i] = mf_info->error_threshold - 1.0f;
    //     else mf_info->item_group_error[i] = mf_info->error_threshold + 1.0f;
    // }

    mf_info->user_group_error[0] = mf_info->error_threshold + 1.0f;
    mf_info->item_group_error[0] = mf_info->error_threshold + 1.0f;

    mf_info->user_group_error[1] = mf_info->error_threshold - 1.0f;
    mf_info->item_group_error[1] = mf_info->error_threshold - 1.0f;

    precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
    
    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_random_fp16<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    mf_info->params.lambda,
                                                    mf_info->d_user_group_end_idx,
                                                    mf_info->d_item_group_end_idx,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num,
                                                    mf_info->fp16_user_num,
                                                    mf_info->fp16_item_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());  

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/half_ratio/rmse_per_epoch/fp32comp_rmse_per_epoch_ur_") + to_string(mf_info->fp16_user_ratio) + "_ir_" + to_string(mf_info->fp16_item_ratio) + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif
    }

    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + reconst_exec_time + cpy2grouped_parameters_exec_time + additional_info_init_exec_time;

#ifdef WRITE_FILE
    map<string, double> statistics_map;

    statistics_map["preprocess"] = 0;
    statistics_map["switching"] = 0.0;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = sgd_update_execution_time / 1000;
    statistics_map["rmse"] = rmse;
    statistics_map["total_user_groups"] = mf_info->fp16_user_num; 
    statistics_map["total_item_groups"] = mf_info->fp16_item_num;
    statistics_map["fp32_user_groups"] = 0;
    statistics_map["fp32_item_groups"] = 0;

    string exec_rmse_output_file_path = string("./New_statistics/half_ratio/time_rmse/fp32comp_time_rmse_ur_")+ to_string(mf_info->fp16_user_ratio) + "_ir_" + to_string(mf_info->fp16_item_ratio) + mf_info->out_file + ".txt";
    print_exec_time_and_rmse_group_only_version(exec_rmse_output_file_path, statistics_map);
#endif

    // cout << "\n<Preprocessing time (micro sec)> " << endl;
    // cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    // cout << "Grouping                         : " << grouping_exec_time << endl;
    // cout << "Matrix reconstruction            : " << reconst_exec_time << endl;
    // cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
    // cout << "Additional info init             : " << additional_info_init_exec_time << endl;
    // cout << "Total preprocessing time         : " << preprocess_exec_time << endl;
    // cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time / 1000<< endl; 
    // cout << "Total MF time(ms)                : " << (preprocess_exec_time + sgd_update_execution_time)/1000 << endl;

}


void grouped_sgd_training_comparison_based_grad_diversity_partial_group_timing_overhead(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();
    cout << "*************Histogram exec time : " << rating_histogram_execution_time << endl;

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size_ret_end_idx(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict_ret_end_idx(mf_info);

    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();
    cout << "*************Grouping on cpu exec time : " << grouping_exec_time << endl;
    
    // cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    // cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    // cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    double reconst_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> reconst_start_point = std::chrono::system_clock::now();
    matrix_reconstruction(mf_info);
    reconst_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - reconst_start_point).count();
    cout << "Mat reconstruction exec time : " << reconst_exec_time << endl;
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    // sgd_info->user_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // sgd_info->user_group_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // cout << "hey2.5" << endl;
    // cout << "User group num : " << mf_info->user_group_num << endl;
    // cout << "Item group num : " << mf_info->item_group_num << endl;

    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();

    //* Group info allocation (host side)
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Group info allocation (device side)
    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    //* Copy grouped parameter from cpu to device's memory    
    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    cout << "********** cpy2grouped_params exec time : " << cpy2grouped_parameters_exec_time << endl;
    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    // mf_info->user_group_prec_info = new unsigned char[mf_info->user_group_num];
    // mf_info->item_group_prec_info = new unsigned char[mf_info->item_group_num];
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    // size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned int)*mf_info->params.k*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    double rmse;
    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
    int start_idx = 0;

    //! Additional code section
    unsigned int cached_user_group_num = 0;
    unsigned int cached_item_group_num = 0;
    unsigned int uncached_user_group_num = mf_info->user_group_num - cached_user_group_num;
    unsigned int uncached_item_group_num = mf_info->item_group_num - cached_item_group_num;

    float* d_user_group_sum_updated_val;
    float* d_item_group_sum_updated_val;
    
    cudaMalloc(&d_user_group_sum_updated_val, sizeof(float) * uncached_user_group_num * mf_info->params.k);
    cudaMalloc(&d_item_group_sum_updated_val, sizeof(float) * uncached_item_group_num * mf_info->params.k);

    //! Initialize to zeros
    unsigned int num_groups = 10000;


    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (2*sizeof(unsigned int)) +
                             (sizeof(unsigned int)*mf_info->params.k*(cached_user_group_num + cached_item_group_num)) + 
                             (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    
    //! For timing

    int clock_rate;
    cudaDeviceGetAttribute(&clock_rate, cudaDevAttrClockRate, 0);
    cout << "**Clock rate : " << clock_rate << endl;

    long long int *time_per_block_arr;
    long long int *d_time_per_block_arr;

    time_per_block_arr = new long long int[block_num * 2];
    cudaMalloc(&d_time_per_block_arr, sizeof(long long int) * block_num * 2);
    
    for (int e = 0; e < mf_info->params.epoch; e++){
        bool error_check = false;
        first_sample_rating_idx = (update_count * update_vector_size) - 0;

        if ((e >= start_idx) && (e % mf_info->interval == (start_idx % mf_info->interval))) {
            error_check = true;
            first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
        }

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            for (int i = 0; i < block_num * 2; i++) time_per_block_arr[i] = (long long int)0;
            cudaMemcpy(d_time_per_block_arr, time_per_block_arr, sizeof(long long int) * block_num * 2, cudaMemcpyHostToDevice);

            initialize_float_array_to_val<<<num_groups, 512>>>(d_user_group_sum_updated_val, uncached_user_group_num * mf_info->params.k, 0.0f);
            initialize_float_array_to_val<<<num_groups, 512>>>(d_item_group_sum_updated_val, uncached_item_group_num * mf_info->params.k, 0.0f);
            cudaDeviceSynchronize();
        }

        double error_init_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_init_time;

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_timing_overhead<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    // mf_info->d_user_index_info,
                                                    // mf_info->d_item_index_info,
                                                    mf_info->d_user_group_end_idx,
                                                    mf_info->d_item_group_end_idx,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_grad_sum_norm_p,
                                                    d_grad_sum_norm_q,
                                                    d_norm_sum_p,
                                                    d_norm_sum_q,
                                                    d_user_group_sum_updated_val,
                                                    d_item_group_sum_updated_val,
                                                    first_sample_rating_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num,
                                                    uncached_user_group_num,
                                                    uncached_item_group_num,
                                                    d_time_per_block_arr
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());  

        // error_computation_start_time = std::chrono::system_clock::now();
        if (error_check){
            double max_overhead = -1;
            cudaMemcpy(time_per_block_arr, d_time_per_block_arr, sizeof(long long int) * block_num * 2, cudaMemcpyDeviceToHost);
            for (int i = block_num; i < block_num * 2; i++){
                double overhead_ms = ((double)time_per_block_arr[i] / ((double)clock_rate * 1000.0))*1000.0;
                max_overhead = overhead_ms > max_overhead ? overhead_ms : max_overhead;
                // cout << i << "," << ((double)time_per_block_arr[i] / ((double)clock_rate * 1000.0))*1000.0 << endl;
                // cout << time_per_block_arr[i] << " ";
            }
            cout << "Overhead time : " << max_overhead << endl;
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_p, d_user_group_sum_updated_val, uncached_user_group_num * mf_info->params.k);
            cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_q, d_item_group_sum_updated_val, uncached_item_group_num * mf_info->params.k);
            cudaDeviceSynchronize();
        }
        // double error_copy_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        // error_computation_time += error_copy_time;

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
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
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        // std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();
        user_group_start_idx = 0;
        unsigned int user_group_end_idx = 0;
        for (int g = 0; g < mf_info->user_group_num; g++){
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
        for (int g = 0; g < mf_info->item_group_num; g++){
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
        if (error_check){
            cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        }
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        // cout << "Transfer time " << error_transfer_time << endl;
        if (error_check){
            cout << "\n<User groups>\n";
            for (int i = 0; i < mf_info->user_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    
                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                        grad_sum_norm_p[i*mf_info->params.k + k] = 0;
                    }
                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                        each_group_norm_acc += norm_sum_p[i * block_num + j];
                    }
                    // cout << each_group_grad_sum_norm_acc << " ";
                    mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                }
                else{
                    mf_info->user_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e >= start_idx) user_switching_log[e].push_back(i);

                if (mf_info->user_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    user_grad_diversity_log[e][i] = -1;
                }           
                else {
                    cout << mf_info->user_group_error[i] << " ";
                    user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
                }
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                error_computation_start_time = std::chrono::system_clock::now();
                float each_group_grad_sum_norm_acc = 0;
                float each_group_norm_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){

                    for (int k = 0; k < mf_info->params.k; k++){
                        each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                        grad_sum_norm_q[i*mf_info->params.k + k] = 0;
                    }

                    for (int j = 0; j < block_num; j++){
                        // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                        each_group_norm_acc += norm_sum_q[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
                    // cout << each_group_grad_sum_norm_acc << " ";
                }
                else{
                    mf_info->item_group_error[i] = UINT_MAX;
                }
                error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
                
                //! log code
                if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e >= start_idx) item_switching_log[e].push_back(i);
                
                if (mf_info->item_group_error[i] == UINT_MAX) {
                    cout << "-1" << " ";
                    item_grad_diversity_log[e][i] = -1;
                }            
                else {
                    cout << mf_info->item_group_error[i] << " ";
                    item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
                }
            }
            cout << "\n";
        }
        // error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
         // //! RMSE write code
        string group_error_metric_output_file_path = string("./statistics/grouping/rmse_per_epoch/grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
        
        precision_switching_start_point = std::chrono::system_clock::now();
        if (error_check){
            cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        }
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
// #endif
    }

    // double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + generate_map_exec_time + cpy2grouped_parameters_exec_time;
    // cout << "\n<Preprocessing time (micro sec)>" << endl;
    // cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    // cout << "Grouping                         : " << grouping_exec_time << endl;
    // cout << "Generate map idx                 : " << generate_map_exec_time << endl;
    // cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
    // cout << "Total preprocessing time         : " << preprocess_exec_time << endl;
    // cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    // cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    // cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    // cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    // cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    // cout << "Total error computation time     : " << error_computation_time << endl;
    // cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    // cout << "Total MF time(ms)                : " << (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time)/1000 << endl;

    // map<string, double> statistics_map;

    // statistics_map["preprocess"] = preprocess_exec_time / 1000;
    // statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    // statistics_map["update"] = sgd_update_execution_time / 1000;
    // statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    // statistics_map["rmse"] = rmse;

    // string exec_rmse_output_file_path = string("./statistics/grouping/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    // string group_switching_log_output_file_path = string("./statistics/grouping/switching_log/group_switching_log_") + mf_info->out_file + ".txt";
    // string group_diversity_log_output_file_path = string("./statistics/grouping/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";
    
    // print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    // print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    // print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
}


void grouped_sgd_training_map_based_grad_diversity_partial_group_fp32_version(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);


    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    check_group_cnt(mf_info);
    // cout << "hey" << endl;
    //* Generating index
    double generate_map_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> generate_map_start_point = std::chrono::system_clock::now();
    generate_map_idx_info(mf_info);
    generate_map_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - generate_map_start_point).count();
    // cout << "hey2" << endl;

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    // sgd_info->user_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // sgd_info->user_group_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // cout << "hey2.5" << endl;
    // cout << "User group num : " << mf_info->user_group_num << endl;
    // cout << "Item group num : " << mf_info->item_group_num << endl;

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    // cpy2grouped_parameters(mf_info, sgd_info);
    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    // cout << "hey3" << endl;

    cpy2grouped_parameters_gpu_float_version(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();
    // cout << "hey4" << endl;

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    // cout  << "User group num : " << mf_info->user_group_num;

    // cout  << "Item group num : " << mf_info->item_group_num;
    

    // mf_info->user_group_prec_info = new unsigned char[mf_info->user_group_num];
    // mf_info->item_group_prec_info = new unsigned char[mf_info->item_group_num];
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    // size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned int)*mf_info->params.k*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    double rmse;
    map<unsigned int, vector<unsigned int>> user_switching_log;
    map<unsigned int, vector<unsigned int>> item_switching_log;
    vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
    
    //! Additional code section
    unsigned int cached_user_group_num = 0;
    unsigned int cached_item_group_num = 0;
    unsigned int uncached_user_group_num = mf_info->user_group_num - cached_user_group_num;
    unsigned int uncached_item_group_num = mf_info->item_group_num - cached_item_group_num;

    float* d_user_group_sum_updated_val;
    float* d_item_group_sum_updated_val;
    
    cudaMalloc(&d_user_group_sum_updated_val, sizeof(float) * uncached_user_group_num * mf_info->params.k);
    cudaMalloc(&d_item_group_sum_updated_val, sizeof(float) * uncached_item_group_num * mf_info->params.k);

    //! initialize to zeros
    unsigned int num_groups = 10000;


    size_t shared_mem_size = (3*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (sizeof(unsigned int)*mf_info->params.k*(cached_user_group_num + cached_item_group_num)) + 
                             (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));

    for (int e = 0; e < mf_info->params.epoch; e++){
        initialize_float_array_to_val<<<num_groups, 512>>>(d_user_group_sum_updated_val, uncached_user_group_num * mf_info->params.k, 0.0f);
        initialize_float_array_to_val<<<num_groups, 512>>>(d_item_group_sum_updated_val, uncached_item_group_num * mf_info->params.k, 0.0f);
        cudaDeviceSynchronize();
        
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_grouped_cache_fp32_verison<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_grad_sum_norm_p,
                                                    d_grad_sum_norm_q,
                                                    d_norm_sum_p,
                                                    d_norm_sum_q,
                                                    d_user_group_sum_updated_val,
                                                    d_item_group_sum_updated_val,
                                                    first_sample_rating_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num,
                                                    uncached_user_group_num,
                                                    uncached_item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());  
 
        cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_p, d_user_group_sum_updated_val, uncached_user_group_num * mf_info->params.k);
        cpyfp32_arr2fp32_arr<<<num_groups, 512>>>(d_grad_sum_norm_q, d_item_group_sum_updated_val, uncached_item_group_num * mf_info->params.k);
        cudaDeviceSynchronize();

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            gpuErr(cudaPeekAtLastError());
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

        for (int i = 0; i < mf_info->max_user; i++){
            unsigned int user_group = mf_info->user_index_info[i].g;
            unsigned int row = mf_info->user_index_info[i].v;
            for (int k = 0; k < mf_info->params.k; k++){
                sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
            }
        }

        for (int i = 0; i < mf_info->max_item; i++){
            unsigned int item_group = mf_info->item_index_info[i].g;
            unsigned int row = mf_info->item_index_info[i].v;
            for (int k = 0; k < mf_info->params.k; k++){
                sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
            }
        }

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        // cout << "\n<User groups>\n";
        error_computation_start_time = std::chrono::system_clock::now();
        for (int i = 0; i < mf_info->user_group_num; i++){
            float each_group_grad_sum_norm_acc = 0;
            float each_group_norm_acc = 0;
            
            for (int k = 0; k < mf_info->params.k; k++){
                each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                grad_sum_norm_p[i*mf_info->params.k + k] = 0;
            }
            for (int j = 0; j < block_num; j++){
                // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                each_group_norm_acc += norm_sum_p[i * block_num + j];
            }
            // cout << each_group_grad_sum_norm_acc << " ";
            mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
            // cout << each_group_norm_acc << " " << each_group_grad_sum_norm_acc;
            // cout << mf_info->user_group_error[i] << " ";
            user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
        }

        // cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            float each_group_grad_sum_norm_acc = 0;
            float each_group_norm_acc = 0;

            for (int k = 0; k < mf_info->params.k; k++){
                each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                grad_sum_norm_q[i*mf_info->params.k + k] = 0;
            }

            for (int j = 0; j < block_num; j++){
                // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                each_group_norm_acc += norm_sum_q[i * block_num + j];
            }
            mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
            // cout << each_group_grad_sum_norm_acc << " ";

            // cout << mf_info->item_group_error[i] << " ";
            item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
        }
        // cout << "\n";
        // error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
    
        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/Gradient diversity/rmse_per_epoch/grouping_rmse_per_epoch_fp32") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif

        //! Group error write code
        cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
    }

#ifdef WRITE_FILE
    string group_diversity_log_output_file_path = string("./New_statistics/Gradient diversity/diversity_log/group_diversity_log_fp32") + mf_info->out_file + ".txt";
    print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
#endif

    // //! RMSE write code
    // string group_error_metric_output_file_path = string("./statistics/") + mf_info->out_file + ".txt";  
    // print_rmse(group_error_metric_output_file_path, rmse);
    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + generate_map_exec_time + cpy2grouped_parameters_exec_time;
    cout << "\n<Preprocessing time (micro sec)>" << endl;
    cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    cout << "Grouping                         : " << grouping_exec_time << endl;
    cout << "Generate map idx                 : " << generate_map_exec_time << endl;
    cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
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

    cout << "User groups" << endl;
    for (int e = 0; e < user_grad_diversity_log.size(); e++){
        for (int g = 0; g < user_grad_diversity_log[e].size(); g++){
            cout << user_grad_diversity_log[e][g] << "\t";
        }
        cout << endl;
    }
    cout << "\nItem groups" << endl;
    for (int e = 0; e < item_grad_diversity_log.size(); e++){
        for (int g = 0; g < item_grad_diversity_log[e].size(); g++){
            cout << item_grad_diversity_log[e][g] << "\t";
        }
        cout << endl;
    }
    cout << "\n\n";

    // map<string, double> statistics_map;

    // statistics_map["preprocess"] = preprocess_exec_time / 1000;
    // statistics_map["switching"] = (precision_switching_exec_time + error_computation_time) / 1000;
    // statistics_map["update"] = sgd_update_execution_time / 1000;
    // statistics_map["total"] = (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    // statistics_map["rmse"] = rmse;

    // string exec_rmse_output_file_path = string("./statistics/grouping/time_rmse/time_rmse_") + mf_info->out_file + ".txt";  
    // string group_switching_log_output_file_path = string("./statistics/grouping/switching_log/group_switching_log_") + mf_info->out_file + ".txt";
    // string group_diversity_log_output_file_path = string("./statistics/grouping/diversity_log/group_diversity_log_") + mf_info->out_file + ".txt";
    
    // print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    // print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    // print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
}

void grouped_sgd_training_map_based_grad_diversity_fp32_version(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);

    //* Grouping methods
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);
    
    // check_group_cnt(mf_info);

    //* Generating index
    generate_map_idx_info(mf_info);

    // sgd_info->user_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // sgd_info->user_group_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    // cpy2grouped_parameters(mf_info, sgd_info);
    cpy2grouped_parameters_gpu_float_version(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    // size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned int)*mf_info->params.k*(mf_info->user_group_num + mf_info->item_group_num)) + 
    //                          (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    size_t shared_mem_size = (sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num));

    double rmse;

    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_fp32_version<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (float**)sgd_info->d_user_group_ptr,
                                                    (float**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    mf_info->params.lambda,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    d_grad_sum_norm_p,
                                                    d_grad_sum_norm_q,
                                                    d_norm_sum_p,
                                                    d_norm_sum_q,
                                                    first_sample_rating_idx,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        
        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            gpuErr(cudaPeekAtLastError());
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

        for (int i = 0; i < mf_info->max_user; i++){
            unsigned int user_group = mf_info->user_index_info[i].g;
            unsigned int row = mf_info->user_index_info[i].v;
            for (int k = 0; k < mf_info->params.k; k++){
                sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
            }
        }

        for (int i = 0; i < mf_info->max_item; i++){
            unsigned int item_group = mf_info->item_index_info[i].g;
            unsigned int row = mf_info->item_index_info[i].v;
            for (int k = 0; k < mf_info->params.k; k++){
                sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
            }
        }

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        cout << "Transfer time " << error_transfer_time << endl;
        cout << "\n<User groups>\n";
        error_computation_start_time = std::chrono::system_clock::now();
        for (int i = 0; i < mf_info->user_group_num; i++){
            float each_group_grad_sum_norm_acc = 0;
            float each_group_norm_acc = 0;
                
            for (int k = 0; k < mf_info->params.k; k++){
                each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
                grad_sum_norm_p[i*mf_info->params.k + k] = 0;
            }
            for (int j = 0; j < block_num; j++){
                // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
                each_group_norm_acc += norm_sum_p[i * block_num + j];
            }
            // cout << each_group_grad_sum_norm_acc << " ";
            mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
            // cout << each_group_norm_acc << " " << each_group_grad_sum_norm_acc;
            if (mf_info->user_group_error[i] == UINT_MAX) cout << "-1" << " ";            
            else cout << mf_info->user_group_error[i] << " ";
        }

        cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            float each_group_grad_sum_norm_acc = 0;
            float each_group_norm_acc = 0;

            for (int k = 0; k < mf_info->params.k; k++){
                each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
                grad_sum_norm_q[i*mf_info->params.k + k] = 0;
            }

            for (int j = 0; j < block_num; j++){
                // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
                each_group_norm_acc += norm_sum_q[i * block_num + j];
            }
            mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
            // cout << each_group_grad_sum_norm_acc << " ";

            if (mf_info->item_group_error[i] == UINT_MAX) cout << "-1" << " ";            
            else cout << mf_info->item_group_error[i] << " ";
        }
        cout << "\n";
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
    
        cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
        cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
    }

    //! RMSE write code
    // string group_error_metric_output_file_path = string("./statistics/") + mf_info->out_file + ".txt";  
    // print_rmse(group_error_metric_output_file_path, rmse);

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl; 
}

void grouped_sgd_training_division_based_indexing(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    // mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    // mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
    cudaMallocHost(&mf_info->user_group_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item_group_idx, sizeof(unsigned int) * mf_info->max_item);
    // cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    // cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    //! Input to end_idx arr and group_idx arr
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) cout << "Grouping number error" << endl;
    else if (mf_info->grouping_method == 3) cout << "Grouping number error" << endl;
    else if (mf_info->grouping_method == 4) cout << "Grouping number error" << endl;

    // cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    // cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    // cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    check_group_cnt(mf_info);
    
    //* GENERATE RECONSTRUTED INDEX
    mf_info->sorted_idx2user = new unsigned int[mf_info->max_user];
    mf_info->sorted_idx2item = new unsigned int[mf_info->max_item];
    cudaMallocHost(&mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);
    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    matrix_reconstruction(mf_info);

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    gpuErr(cudaPeekAtLastError());

    // cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_gpu_for_division_indexing(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    // // for (int i = 0; i < mf_info->user_group_num; i++)
    // //     mf_info->user_group_prec_info[i] = 1;
    // // for (int i = 0; i < mf_info->item_group_num; i++)
    // //     mf_info->item_group_prec_info[i] = 0;

    // // cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // // cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    unsigned int *user_group_update_cnt;
    unsigned int *item_group_update_cnt;
    unsigned int *d_user_group_update_cnt;
    unsigned int *d_item_group_update_cnt;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    // // cudaMemset(d_acc_user_group_error, 0, sizeof(float) * block_num * mf_info->user_group_num);
    // // cudaMemset(d_acc_item_group_error, 0, sizeof(float) * block_num * mf_info->item_group_num);
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 
    cout << (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num) << "=======" << endl;    
    cout << (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num) << "=======" << endl;
    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_division_based_indexing<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_acc_user_group_error,
                                                    d_acc_item_group_error,
                                                    d_user_group_update_cnt,
                                                    d_item_group_update_cnt,
                                                    first_sample_rating_idx,
                                                    (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num),
                                                    (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num),
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
    //     //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
    //     // if (e == mf_info->params.epoch-1){
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                if (mf_info->user_group_prec_info[i] == 0) {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
                else {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
            }
            gpuErr(cudaPeekAtLastError());

            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            unsigned int user_group_start_idx = 0;
            unsigned int user_group_end_idx = 0;
            for (int g = 0; g < mf_info->user_group_num; g++){
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

            unsigned int item_group_start_idx = 0;
            unsigned int item_group_end_idx = 0;
            for (int g = 0; g < mf_info->item_group_num; g++){
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

    //         // for (int i = 0; i < mf_info->max_user; i++){
    //         //     unsigned int user_group = mf_info->user_index_info[i].g;
    //         //     unsigned int row = mf_info->user_index_info[i].v;
    //         //     if (mf_info->user_group_prec_info[user_group] == 0){
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
    //         //         }     
    //         //     }
    //         //     else {
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
    //         //         }
    //         //     }
    //         // }

    //         // for (int i = 0; i < mf_info->max_item; i++){
    //         //     unsigned int item_group = mf_info->item_index_info[i].g;
    //         //     unsigned int row = mf_info->item_index_info[i].v;
    //         //     if (mf_info->item_group_prec_info[item_group] == 0){
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
    //         //         }
    //         //     }
    //         //     else{
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
    //         //         }
    //         //     }
    //         // }
    //     // }
        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        cout << "Transfer time " << error_transfer_time << endl;
        cout << "\n<User groups>\n";
        error_computation_start_time = std::chrono::system_clock::now();
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->user_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_user_group_error[i * block_num + j];
                    each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
                }
                mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            }
            else{
                mf_info->user_group_error[i] = -1;
            }
            cout << mf_info->user_group_error[i] << " ";
        }

        cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->item_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_item_group_error[i * block_num + j];
                    each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
                }
                mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            }
            else{
                mf_info->item_group_error[i] = -1;
            }
            cout << mf_info->item_group_error[i] << " ";
        }
        cout << "\n";
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        if (e > 0)
        precision_switching_by_groups(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        // if (e == mf_info->params.epoch-1){
        double rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    //     // }

    //     // if (e == 0){    
    //     //     for (int i = 0; i < mf_info->user_group_num; i++)
    //     //         mf_info->user_group_prec_info[i] = 1;
    //     //     for (int i = 0; i < mf_info->item_group_num; i++)
    //     //         mf_info->item_group_prec_info[i] = 0;

    //     //     cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    //     //     cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    //     // }
    }

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl;
}

void grouped_sgd_training_division_based_indexing_fp32_version(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    // mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    // mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
    cudaMallocHost(&mf_info->user_group_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item_group_idx, sizeof(unsigned int) * mf_info->max_item);
    // cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    // cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    //! Input to end_idx arr and group_idx arr
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) cout << "Grouping number error" << endl;
    else if (mf_info->grouping_method == 3) cout << "Grouping number error" << endl;
    else if (mf_info->grouping_method == 4) cout << "Grouping number error" << endl;

    // cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    // cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    // cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    check_group_cnt(mf_info);
    
    //* GENERATE RECONSTRUTED INDEX
    mf_info->sorted_idx2user = new unsigned int[mf_info->max_user];
    mf_info->sorted_idx2item = new unsigned int[mf_info->max_item];
    cudaMallocHost(&mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);
    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    matrix_reconstruction(mf_info);

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    gpuErr(cudaPeekAtLastError());

    // cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_gpu_for_division_indexing_float_version(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    // // for (int i = 0; i < mf_info->user_group_num; i++)
    // //     mf_info->user_group_prec_info[i] = 1;
    // // for (int i = 0; i < mf_info->item_group_num; i++)
    // //     mf_info->item_group_prec_info[i] = 0;

    // // cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // // cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    unsigned int *user_group_update_cnt;
    unsigned int *item_group_update_cnt;
    unsigned int *d_user_group_update_cnt;
    unsigned int *d_item_group_update_cnt;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    // // cudaMemset(d_acc_user_group_error, 0, sizeof(float) * block_num * mf_info->user_group_num);
    // // cudaMemset(d_acc_item_group_error, 0, sizeof(float) * block_num * mf_info->item_group_num);
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 
    cout << (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num) << "=======" << endl;    
    cout << (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num) << "=======" << endl;
    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));    
    
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_division_based_indexing_fp32_version<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    mf_info->params.lambda,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    d_acc_user_group_error,
                                                    d_acc_item_group_error,
                                                    d_user_group_update_cnt,
                                                    d_item_group_update_cnt,
                                                    first_sample_rating_idx,
                                                    (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num),
                                                    (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num),
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
    //     //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
    //     // if (e == mf_info->params.epoch-1){
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
            gpuErr(cudaPeekAtLastError());

            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            unsigned int user_group_start_idx = 0;
            unsigned int user_group_end_idx = 0;
            for (int g = 0; g < mf_info->user_group_num; g++){
                user_group_end_idx += mf_info->user_group_size[g];
                unsigned int user_idx_in_group = 0;
                for (int u = user_group_start_idx; u < user_group_end_idx; u++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[u * mf_info->params.k + k] = (((float*)sgd_info->user_group_ptr[g])[user_idx_in_group * mf_info->params.k + k]);
                    }
                    user_idx_in_group += 1;
                }
                user_group_start_idx = user_group_end_idx;
            }

            unsigned int item_group_start_idx = 0;
            unsigned int item_group_end_idx = 0;
            for (int g = 0; g < mf_info->item_group_num; g++){
                item_group_end_idx += mf_info->item_group_size[g];
                unsigned int item_idx_in_group = 0;
                for (int i = item_group_start_idx; i < item_group_end_idx; i++){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i * mf_info->params.k + k] = (((float*)sgd_info->item_group_ptr[g])[item_idx_in_group * mf_info->params.k + k]);
                    }
                    item_idx_in_group += 1;
                }
                item_group_start_idx = item_group_end_idx;
            }

    // //         // for (int i = 0; i < mf_info->max_user; i++){
    // //         //     unsigned int user_group = mf_info->user_index_info[i].g;
    // //         //     unsigned int row = mf_info->user_index_info[i].v;
    // //         //     if (mf_info->user_group_prec_info[user_group] == 0){
    // //         //         for (int k = 0; k < mf_info->params.k; k++){
    // //         //             sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
    // //         //         }     
    // //         //     }
    // //         //     else {
    // //         //         for (int k = 0; k < mf_info->params.k; k++){
    // //         //             sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
    // //         //         }
    // //         //     }
    // //         // }

    // //         // for (int i = 0; i < mf_info->max_item; i++){
    // //         //     unsigned int item_group = mf_info->item_index_info[i].g;
    // //         //     unsigned int row = mf_info->item_index_info[i].v;
    // //         //     if (mf_info->item_group_prec_info[item_group] == 0){
    // //         //         for (int k = 0; k < mf_info->params.k; k++){
    // //         //             sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
    // //         //         }
    // //         //     }
    // //         //     else{
    // //         //         for (int k = 0; k < mf_info->params.k; k++){
    // //         //             sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
    // //         //         }
    // //         //     }
    // //         // }
    // //     // }
        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        error_computation_time += error_transfer_time;
        
        cout << "Transfer time " << error_transfer_time << endl;
        cout << "\n<User groups>\n";
        error_computation_start_time = std::chrono::system_clock::now();
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->user_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_user_group_error[i * block_num + j];
                    each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
                }
                mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            }
            else{
                mf_info->user_group_error[i] = -1;
            }
            cout << mf_info->user_group_error[i] << " ";
        }

        cout << "\n<Item groups>\n";
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int each_group_error_acc = 0;
            unsigned int each_group_cnt_acc = 0;
            if (mf_info->item_group_prec_info[i] == 0){
                for (int j = 0; j < block_num; j++){
                    each_group_error_acc += acc_item_group_error[i * block_num + j];
                    each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
                }
                mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
            }
            else{
                mf_info->item_group_error[i] = -1;
            }
            cout << mf_info->item_group_error[i] << " ";
        }
        cout << "\n";
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        // if (e > 0)
        // precision_switching_by_groups(mf_info, sgd_info);
        precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        // if (e == mf_info->params.epoch-1){
        double rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    // //     // }

    // //     // if (e == 0){    
    // //     //     for (int i = 0; i < mf_info->user_group_num; i++)
    // //     //         mf_info->user_group_prec_info[i] = 1;
    // //     //     for (int i = 0; i < mf_info->item_group_num; i++)
    // //     //         mf_info->item_group_prec_info[i] = 0;

    // //     //     cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // //     //     cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    // //     // }
    }

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl;
}

void grouped_sgd_training_division_based_indexing_iteration(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    // mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    // mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
    cudaMallocHost(&mf_info->user_group_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item_group_idx, sizeof(unsigned int) * mf_info->max_item);
    // cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    // cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    //! Input to end_idx arr and group_idx arr
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) cout << "Grouping number error" << endl;
    else if (mf_info->grouping_method == 3) cout << "Grouping number error" << endl;
    else if (mf_info->grouping_method == 4) cout << "Grouping number error" << endl;

    // cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    // cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    // cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    check_group_cnt(mf_info);
    
    //* GENERATE RECONSTRUTED INDEX
    mf_info->sorted_idx2user = new unsigned int[mf_info->max_user];
    mf_info->sorted_idx2item = new unsigned int[mf_info->max_item];
    cudaMallocHost(&mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);
    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    matrix_reconstruction(mf_info);

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    gpuErr(cudaPeekAtLastError());

    // cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);
    cpy2grouped_parameters_gpu_for_division_indexing(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    unsigned int div = mf_info->params.thread_block_size/32;

    // // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    // // for (int i = 0; i < mf_info->user_group_num; i++)
    // //     mf_info->user_group_prec_info[i] = 1;
    // // for (int i = 0; i < mf_info->item_group_num; i++)
    // //     mf_info->item_group_prec_info[i] = 0;

    // // cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // // cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    unsigned int *user_group_update_cnt;
    unsigned int *item_group_update_cnt;
    unsigned int *d_user_group_update_cnt;
    unsigned int *d_item_group_update_cnt;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    // // cudaMemset(d_acc_user_group_error, 0, sizeof(float) * block_num * mf_info->user_group_num);
    // // cudaMemset(d_acc_item_group_error, 0, sizeof(float) * block_num * mf_info->item_group_num);
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 
    cout << (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num) << "=======" << endl;    
    cout << (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num) << "=======" << endl;
    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));    
    
    unsigned int iteration_set_num = 4;
    double sgd_update_execution_time = 0;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int update_count_per_iteration = ceil(update_count/(float)iteration_set_num);
    int update_count_last_iteration = update_count - (update_count_per_iteration * (iteration_set_num-1));

    cout << "Total update cnt                 : " << ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size)) << endl;
    // cout << "Update cnt per iteration set     : " << update_count << endl;
    // cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    // cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    // cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;    
    for (int e = 0; e < mf_info->params.epoch; e++){

        for (int it = 0; it < iteration_set_num; it++){
            int update_count_tmp;
            if (it == iteration_set_num - 1) update_count_tmp = update_count_last_iteration;
            else update_count_tmp = update_count_per_iteration;
            cout << "Update cnt per iteration set     : " << update_count_tmp << endl;
            int sample_ratings_num = (float)(update_count_tmp * update_vector_size) * mf_info->sample_ratio;
            unsigned int first_sample_rating_idx = (update_count_tmp * update_vector_size) - sample_ratings_num;
            
            std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
            sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_division_based_indexing<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr_half[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count_tmp,
                                                        update_vector_size,
                                                        lambda_half,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_acc_user_group_error,
                                                        d_acc_item_group_error,
                                                        d_user_group_update_cnt,
                                                        d_item_group_update_cnt,
                                                        first_sample_rating_idx,
                                                        (unsigned int)ceil(mf_info->max_user/(float)mf_info->user_group_num),
                                                        (unsigned int)ceil(mf_info->max_item/(float)mf_info->item_group_num),
                                                        mf_info->user_group_num,
                                                        mf_info->item_group_num
                                                        );
            cudaDeviceSynchronize();
            sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
            gpuErr(cudaPeekAtLastError()); 
            std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
            cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            error_computation_time += error_transfer_time;
            
            cout << "Transfer time " << error_transfer_time << endl;
            cout << "\n<User groups>\n";
            error_computation_start_time = std::chrono::system_clock::now();
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int each_group_error_acc = 0;
                unsigned int each_group_cnt_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    for (int j = 0; j < block_num; j++){
                        each_group_error_acc += acc_user_group_error[i * block_num + j];
                        each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
                    }
                    mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
                }
                else{
                    mf_info->user_group_error[i] = -1;
                }
                cout << mf_info->user_group_error[i] << " ";
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int each_group_error_acc = 0;
                unsigned int each_group_cnt_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){
                    for (int j = 0; j < block_num; j++){
                        each_group_error_acc += acc_item_group_error[i * block_num + j];
                        each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
                }
                else{
                    mf_info->item_group_error[i] = -1;
                }
                cout << mf_info->item_group_error[i] << " ";
            }
            cout << "\n";
            error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
            std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
            // if (e > 10)
            precision_switching_by_groups(mf_info, sgd_info);
            precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
        }

    //     //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
    //     // if (e == mf_info->params.epoch-1){
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                if (mf_info->user_group_prec_info[i] == 0) {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
                else {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
            }
            gpuErr(cudaPeekAtLastError());

            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            unsigned int user_group_start_idx = 0;
            unsigned int user_group_end_idx = 0;
            for (int g = 0; g < mf_info->user_group_num; g++){
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

            unsigned int item_group_start_idx = 0;
            unsigned int item_group_end_idx = 0;
            for (int g = 0; g < mf_info->item_group_num; g++){
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

    //         // for (int i = 0; i < mf_info->max_user; i++){
    //         //     unsigned int user_group = mf_info->user_index_info[i].g;
    //         //     unsigned int row = mf_info->user_index_info[i].v;
    //         //     if (mf_info->user_group_prec_info[user_group] == 0){
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
    //         //         }     
    //         //     }
    //         //     else {
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
    //         //         }
    //         //     }
    //         // }

    //         // for (int i = 0; i < mf_info->max_item; i++){
    //         //     unsigned int item_group = mf_info->item_index_info[i].g;
    //         //     unsigned int row = mf_info->item_index_info[i].v;
    //         //     if (mf_info->item_group_prec_info[item_group] == 0){
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
    //         //         }
    //         //     }
    //         //     else{
    //         //         for (int k = 0; k < mf_info->params.k; k++){
    //         //             sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
    //         //         }
    //         //     }
    //         // }
    //     // }

        // if (e == mf_info->params.epoch-1){
        double rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    //     // }

    //     // if (e == 0){    
    //     //     for (int i = 0; i < mf_info->user_group_num; i++)
    //     //         mf_info->user_group_prec_info[i] = 1;
    //     //     for (int i = 0; i < mf_info->item_group_num; i++)
    //     //         mf_info->item_group_prec_info[i] = 0;

    //     //     cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    //     //     cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);
    //     // }
    }

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl;
}

void grouped_sgd_training_comparison_based_indexing_iteration(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    user_item_rating_histogram(mf_info);
    
    //* Grouping on CPU
    // mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    // mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    
    cudaMallocHost(&mf_info->user_group_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item_group_idx, sizeof(unsigned int) * mf_info->max_item);
    cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    //! Input to end_idx arr and group_idx arr
    split_group_based_equal_size_ret_end_idx(mf_info);

    cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    check_group_cnt(mf_info);
    
    //* GENERATE RECONSTRUTED INDEX
    mf_info->sorted_idx2user = new unsigned int[mf_info->max_user];
    mf_info->sorted_idx2item = new unsigned int[mf_info->max_item];
    cudaMallocHost(&mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);
    //* MAT RECONSTRUCTION ON DEVICE
    //! Input to entity2sorted_idx, sorted_idx2entity, group_size
    matrix_reconstruction(mf_info);

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    gpuErr(cudaPeekAtLastError());

    cpy2grouped_parameters_gpu_for_comparison_indexing(mf_info, sgd_info);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    unsigned int iteration_set_num = 1;
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int update_count_per_iteration = ceil(update_count/(float)iteration_set_num);
    int update_count_last_iteration = update_count - (update_count_per_iteration * (iteration_set_num-1));
    cout << "Total update cnt                 : " << ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size)) << endl;

    // int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    // unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    // cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    // cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    // cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    double additional_data_transfer_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_data_transfer_start_time = std::chrono::system_clock::now();

    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    // for (int i = 0; i < mf_info->user_group_num; i++)
    //     mf_info->user_group_prec_info[i] = 1;
    // for (int i = 0; i < mf_info->item_group_num; i++)
    //     mf_info->item_group_prec_info[i] = 0;

    // cudaMemcpy(mf_info->d_user_group_prec_info, mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    // cudaMemcpy(mf_info->d_item_group_prec_info, mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    unsigned int *user_group_update_cnt;
    unsigned int *item_group_update_cnt;
    unsigned int *d_user_group_update_cnt;
    unsigned int *d_item_group_update_cnt;

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMalloc(&d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMalloc(&d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    cudaMallocHost(&user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num);
    cudaMallocHost(&item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num);
    additional_data_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_data_transfer_start_time).count();
    // cudaMemset(d_acc_user_group_error, 0, sizeof(float) * block_num * mf_info->user_group_num);
    // cudaMemset(d_acc_item_group_error, 0, sizeof(float) * block_num * mf_info->item_group_num);
    
    cout << "\n<User & item additional data transfer time (micro sec)>" << endl;
    cout << "Time                             : " << additional_data_transfer_time << endl; 

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (5*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (2 * sizeof(unsigned int)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
    for (int e = 0; e < mf_info->params.epoch; e++){

        for (int it = 0; it < iteration_set_num; it++){
            int update_count_tmp;
            if (it == iteration_set_num - 1) update_count_tmp = update_count_last_iteration;
            else update_count_tmp = update_count_per_iteration;
            cout << "Update cnt per iteration set     : " << update_count_tmp << endl;
            int sample_ratings_num = (float)(update_count_tmp * update_vector_size) * mf_info->sample_ratio;
            unsigned int first_sample_rating_idx = (update_count_tmp * update_vector_size) - sample_ratings_num;
            std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
            sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_range_based_indexing<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr_half[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count_tmp,
                                                        update_vector_size,
                                                        lambda_half,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_acc_user_group_error,
                                                        d_acc_item_group_error,
                                                        d_user_group_update_cnt,
                                                        d_item_group_update_cnt,
                                                        first_sample_rating_idx,
                                                        mf_info->d_user_group_end_idx,
                                                        mf_info->d_item_group_end_idx,
                                                        mf_info->user_group_num,
                                                        mf_info->item_group_num
                                                        );
            cudaDeviceSynchronize();
            sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
            gpuErr(cudaPeekAtLastError());  
            std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
            cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(user_group_update_cnt, d_user_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(item_group_update_cnt, d_item_group_update_cnt, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            error_computation_time += error_transfer_time;
            
            cout << "Transfer time " << error_transfer_time << endl;
            cout << "\n<User groups>\n";
            error_computation_start_time = std::chrono::system_clock::now();
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int each_group_error_acc = 0;
                unsigned int each_group_cnt_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    for (int j = 0; j < block_num; j++){
                        each_group_error_acc += acc_user_group_error[i * block_num + j];
                        each_group_cnt_acc += user_group_update_cnt[i * block_num + j];
                    }
                    mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
                }
                else{
                    mf_info->user_group_error[i] = -1;
                }
                cout << mf_info->user_group_error[i] << " ";
            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int each_group_error_acc = 0;
                unsigned int each_group_cnt_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){
                    for (int j = 0; j < block_num; j++){
                        each_group_error_acc += acc_item_group_error[i * block_num + j];
                        each_group_cnt_acc += item_group_update_cnt[i * block_num + j];
                    }
                    mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
                }
                else{
                    mf_info->item_group_error[i] = -1;
                }
                cout << mf_info->item_group_error[i] << " ";
            }
            cout << "\n";
            error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
            std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
            // if (e == 0)
            precision_switching_by_groups(mf_info, sgd_info);
            precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
        }

        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        // if (e == mf_info->params.epoch-1){
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                if (mf_info->user_group_prec_info[i] == 0) {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
                else {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
            }
            gpuErr(cudaPeekAtLastError());

            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            unsigned int user_group_start_idx = 0;
            unsigned int user_group_end_idx = 0;
            for (int g = 0; g < mf_info->user_group_num; g++){
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

            unsigned int item_group_start_idx = 0;
            unsigned int item_group_end_idx = 0;
            for (int g = 0; g < mf_info->item_group_num; g++){
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

          
        double rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    cout << "Total error computation time     : " << error_computation_time << endl;
    cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl; 
}

void grouped_sgd_training_map_based_indexing_check_gradient(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }
    
    // //* Grouping on CPU
    // mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    // mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    // unsigned int error_kernel_work_groups = 2048;
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
 
    // cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    // cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

    __half* half_p = (__half*)malloc(sizeof(__half) * mf_info->max_user * mf_info->params.k); 
    __half* half_q = (__half*)malloc(sizeof(__half) * mf_info->max_item * mf_info->params.k);
    __half* d_half_p;
    __half* d_half_q;

    cudaMalloc(&d_half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k);    
    cudaMalloc(&d_half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k);
    
    conversion_features_half((short*)half_p, sgd_info->p ,mf_info->max_user, mf_info->params.k);
    conversion_features_half((short*)half_q, sgd_info->q ,mf_info->max_item, mf_info->params.k);
    
    cudaMemcpy(d_half_p, half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_half_q, half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);   
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
    __half lambda_half = __float2half_rn(mf_info->params.lambda);

    float* grad_mean_array_user;
    float* grad_mean_array_item;
    float* d_grad_mean_array_user;
    float* d_grad_mean_array_item;
    unsigned int* update_cnt_array_user;
    unsigned int* update_cnt_array_item;
    unsigned int* d_update_cnt_array_user;
    unsigned int* d_update_cnt_array_item;
    
    cudaMallocHost(&update_cnt_array_user, sizeof(unsigned int)*mf_info->max_user);
    cudaMallocHost(&update_cnt_array_item, sizeof(unsigned int)*mf_info->max_item);
    cudaMallocHost(&grad_mean_array_user, sizeof(float)*mf_info->max_user);
    cudaMallocHost(&grad_mean_array_item, sizeof(float)*mf_info->max_item);

    cudaMalloc(&d_update_cnt_array_user, sizeof(unsigned int)*mf_info->max_user);    
    cudaMalloc(&d_update_cnt_array_item, sizeof(unsigned int)*mf_info->max_item);
    cudaMalloc(&d_grad_mean_array_user, sizeof(float)*mf_info->max_user);    
    cudaMalloc(&d_grad_mean_array_item, sizeof(float)*mf_info->max_item);
    unsigned int block_num = mf_info->params.num_workers/div;
    unsigned int *acc_user_group_error;
    unsigned int *acc_item_group_error;
    unsigned int *d_acc_user_group_error;
    unsigned int *d_acc_item_group_error;
    // unsigned int* count_num_of_ratings_user = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    // unsigned int* count_num_of_ratings_item = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    // pair<float, unsigned int>* mean_grad_user = (pair<float, unsigned int>*)malloc(sizeof(pair<float, unsigned int>) * mf_info->max_user);
    // pair<float, unsigned int>* mean_grad_item = (pair<float, unsigned int>*)malloc(sizeof(pair<float, unsigned int>) * mf_info->max_item);
    // for (int i = 0; i < mf_info->max_user; i++) count_num_of_ratings_user[i] = 0;
    // for (int i = 0; i < mf_info->max_item; i++) count_num_of_ratings_item[i] = 0;
    size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    double sgd_update_execution_time = 0;
    for (int e = 0; e < mf_info->params.epoch; e++){
        if (e < 20){
            std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
            half_sgd_k128_kernel_hogwild_warp32_lrate<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        d_half_p,
                                                        d_half_q,
                                                        d_rand_state,
                                                        lr_decay_arr_half[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count,
                                                        update_vector_size,
                                                        lambda_half
                                                        );
            cudaDeviceSynchronize();
            sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();

            gpuErr(cudaPeekAtLastError());
            cudaMemcpy(half_p, d_half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
            cudaMemcpy(half_q, d_half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
            
            transform_feature_vector_half2float((short*)half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
            transform_feature_vector_half2float((short*)half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
        }else if (e == 20){
            for (int i = 0; i < mf_info->max_user; i++) {grad_mean_array_user[i] = 0.f; update_cnt_array_user[i] = 0;}
            for (int i = 0; i < mf_info->max_item; i++) {grad_mean_array_item[i] = 0.f; update_cnt_array_item[i] = 0;}

            cudaMemcpy(d_update_cnt_array_user, update_cnt_array_user, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyHostToDevice);
            cudaMemcpy(d_update_cnt_array_item, update_cnt_array_item, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_mean_array_user, grad_mean_array_user, sizeof(float) * mf_info->max_user, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_mean_array_item, grad_mean_array_item, sizeof(float) * mf_info->max_item, cudaMemcpyHostToDevice);
            std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
            half_sgd_k128_kernel_hogwild_warp32_lrate_grad_mean_user_item<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                            mf_info->d_R,
                                                            mf_info->n,
                                                            d_half_p,
                                                            d_half_q,
                                                            d_rand_state,
                                                            lr_decay_arr_half[e],
                                                            mf_info->params.k,
                                                            1,
                                                            e,
                                                            update_count,
                                                            update_vector_size,
                                                            lambda_half,
                                                            d_grad_mean_array_user,
                                                            d_grad_mean_array_item,
                                                            d_update_cnt_array_user,
                                                            d_update_cnt_array_item,
                                                            0
                                                            );
            cudaDeviceSynchronize();
            sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();

            cudaMemcpy(update_cnt_array_user, d_update_cnt_array_user, sizeof(unsigned int)*mf_info->max_user, cudaMemcpyDeviceToHost);
            cudaMemcpy(update_cnt_array_item, d_update_cnt_array_item, sizeof(unsigned int)*mf_info->max_item, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_mean_array_user, d_grad_mean_array_user, sizeof(float)*mf_info->max_user, cudaMemcpyDeviceToHost);
            cudaMemcpy(grad_mean_array_item, d_grad_mean_array_item, sizeof(float)*mf_info->max_item, cudaMemcpyDeviceToHost); 
            
            for (int i = 0; i < mf_info->max_user; i++) {grad_mean_array_user[i] = grad_mean_array_user[i] / ((float)update_cnt_array_user[i] * mf_info->params.k);}
            for (int i = 0; i < mf_info->max_item; i++) {grad_mean_array_item[i] = grad_mean_array_item[i] / ((float)update_cnt_array_item[i] * mf_info->params.k);}
            
            cudaMemcpy(d_grad_mean_array_user, grad_mean_array_user, sizeof(float) * mf_info->max_user, cudaMemcpyHostToDevice);
            cudaMemcpy(d_grad_mean_array_item, grad_mean_array_item, sizeof(float) * mf_info->max_item, cudaMemcpyHostToDevice);
            cudaMallocHost(&mf_info->user2idx, sizeof(unsigned int) * mf_info->max_user);
            cudaMallocHost(&mf_info->item2idx, sizeof(unsigned int) * mf_info->max_item);
            cudaMalloc(&mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user);
            cudaMalloc(&mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item);
    
            unsigned int num_groups = 10000;

            init_idx_arr<<<num_groups, 512>>>(mf_info->d_user2idx, mf_info->max_user);
            init_idx_arr<<<num_groups, 512>>>(mf_info->d_item2idx, mf_info->max_item);
            cudaDeviceSynchronize();
            cudaMemcpy(mf_info->user2idx, mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
            cudaMemcpy(mf_info->item2idx, mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);

            thrust::device_ptr<float> thrust_d_user2grad(d_grad_mean_array_user);
            thrust::device_ptr<float> thrust_d_item2grad(d_grad_mean_array_item);
            thrust::device_ptr<unsigned int> thrust_d_user2idx(mf_info->d_user2idx);
            thrust::device_ptr<unsigned int> thrust_d_item2idx(mf_info->d_item2idx);
            thrust::sort_by_key(thrust_d_user2grad, thrust_d_user2grad + mf_info->max_user, thrust_d_user2idx, thrust::greater<float>());
            thrust::sort_by_key(thrust_d_item2grad, thrust_d_item2grad + mf_info->max_item, thrust_d_item2idx, thrust::greater<float>());
            cudaDeviceSynchronize();

            // cudaMemcpy(d_grad_mean_array_user, grad_mean_array_user, sizeof(float) * mf_info->max_user, cudaMemcpyDeviceToHost);
            // cudaMemcpy(d_grad_mean_array_item, grad_mean_array_item, sizeof(float) * mf_info->max_item, cudaMemcpyDeviceToHost);
            cudaMemcpy(mf_info->user2idx, mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
            cudaMemcpy(mf_info->item2idx, mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);

            //* Grouping on CPU
            mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
            mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
            
            //* Grouping methods
            split_group_based_equal_size(mf_info);
            check_group_cnt(mf_info);

            generate_map_idx_info(mf_info);
            // check_group_cnt(mf_info);

            cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
            cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
            cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
            cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

            cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
            cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
            gpuErr(cudaPeekAtLastError());

            cudaMemcpy(half_p, d_half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
            cudaMemcpy(half_q, d_half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);        
            transform_feature_vector_half2float((short*)half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
            transform_feature_vector_half2float((short*)half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);

            cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
            cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);

            cpy2grouped_parameters_gpu(mf_info, sgd_info);

            cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
            cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
            cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
            cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
            cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
            cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
            cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
            cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);
            cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
            cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
            cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
            cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);

            // transform_feature_vector_half2float((short*)half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
            // transform_feature_vector_half2float((short*)half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
            
            for (int i = 0; i < mf_info->max_user; i++) {update_cnt_array_user[i] = 0;}
            for (int i = 0; i < mf_info->max_item; i++) {update_cnt_array_item[i] = 0;}

            cudaMemcpy(d_update_cnt_array_user, update_cnt_array_user, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyHostToDevice);
            cudaMemcpy(d_update_cnt_array_item, update_cnt_array_item, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyHostToDevice);
        }else {
            std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();

            sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                        mf_info->d_R,
                                                        mf_info->n,
                                                        (void**)sgd_info->d_user_group_ptr,
                                                        (void**)sgd_info->d_item_group_ptr,
                                                        d_rand_state,
                                                        lr_decay_arr_half[e],
                                                        mf_info->params.k,
                                                        1,
                                                        e,
                                                        update_count,
                                                        update_vector_size,
                                                        lambda_half,
                                                        mf_info->d_user_index_info,
                                                        mf_info->d_item_index_info,
                                                        mf_info->d_user_group_prec_info,
                                                        mf_info->d_item_group_prec_info,
                                                        d_acc_user_group_error,
                                                        d_acc_item_group_error,
                                                        d_update_cnt_array_user,
                                                        d_update_cnt_array_item,
                                                        first_sample_rating_idx,
                                                        mf_info->user_group_num,
                                                        mf_info->item_group_num
                                                        );
            cudaDeviceSynchronize();
            sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();

            //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
            unsigned user_group_start_idx = 0;
            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
                if (mf_info->user_group_prec_info[i] == 0) {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
                else {
                    cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                    gpuErr(cudaPeekAtLastError());
                }
            }
            gpuErr(cudaPeekAtLastError());

            unsigned item_group_start_idx = 0;
            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
                if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
            }
            gpuErr(cudaPeekAtLastError());

            std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

            for (int i = 0; i < mf_info->max_user; i++){
                unsigned int user_group = mf_info->user_index_info[i].g;
                unsigned int row = mf_info->user_index_info[i].v;
                if (mf_info->user_group_prec_info[user_group] == 0){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
                    }     
                }
                else {
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
                    }
                }
            }

            for (int i = 0; i < mf_info->max_item; i++){
                unsigned int item_group = mf_info->item_index_info[i].g;
                unsigned int row = mf_info->item_index_info[i].v;
                if (mf_info->item_group_prec_info[item_group] == 0){
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
                    }
                }
                else{
                    for (int k = 0; k < mf_info->params.k; k++){
                        sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
                    }
                }
            }
        // }
            std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
            cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(update_cnt_array_user, d_update_cnt_array_user, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
            cudaMemcpy(update_cnt_array_item, d_update_cnt_array_item, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
            double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            error_computation_time += error_transfer_time;
            
            cout << "Transfer time " << error_transfer_time << endl;
            cout << "\n<User groups>\n";
            error_computation_start_time = std::chrono::system_clock::now();

            for (int i = 0; i < mf_info->user_group_num; i++){
                unsigned int each_group_error_acc = 0;
                unsigned int each_group_cnt_acc = 0;
                if (mf_info->user_group_prec_info[i] == 0){
                    for (int j = 0; j < block_num; j++){
                        each_group_error_acc += acc_user_group_error[i * block_num + j];
                        each_group_cnt_acc += update_cnt_array_user[i * block_num + j];
                    }
                    // if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
                    if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = (each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k))*log2((each_group_cnt_acc/(float)mf_info->n) + 2.0f);
                    else mf_info->user_group_error[i] = 0;
                }
                else{
                    mf_info->user_group_error[i] = -1;
                }
                // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->user_group_error[i] << endl;
                cout << mf_info->user_group_error[i] << " ";

            }

            cout << "\n<Item groups>\n";
            for (int i = 0; i < mf_info->item_group_num; i++){
                unsigned int each_group_error_acc = 0;
                unsigned int each_group_cnt_acc = 0;
                if (mf_info->item_group_prec_info[i] == 0){
                    for (int j = 0; j < block_num; j++){
                        each_group_error_acc += acc_item_group_error[i * block_num + j];
                        each_group_cnt_acc += update_cnt_array_item[i * block_num + j];
                    }
                    // if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k); 
                    if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = (each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k))*log2((each_group_cnt_acc/(float)mf_info->n) + 2.0f);

                    else mf_info->item_group_error[i] = 0;
                }
                else{
                    mf_info->item_group_error[i] = -1;
                }
                // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->item_group_error[i] << endl;
                cout << mf_info->item_group_error[i] << " ";
            }
            cout << "\n";
            error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
            std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
            precision_switching_by_groups(mf_info, sgd_info);
            precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
        }
        float rmse = 0;
        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }
    cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update     : " << sgd_update_execution_time << endl; 

}

// void grouped_sgd_training_map_based_indexing_check_gradient(Mf_info* mf_info, SGD* sgd_info){
//     //* Transfer rating triplets to GPU 
//     // random_shuffle(mf_info->R, mf_info->R + mf_info->n);

//     cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
//     cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
//     gpuErr(cudaPeekAtLastError());
    
//     //* Convert testset to COO format
//     mf_info->test_COO = test_set_preprocess(mf_info);
//     Node* d_test_COO;
//     cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
//     gpuErr(cudaPeekAtLastError());
//     cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

//     float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
//     for (int i = 0; i < mf_info->params.epoch; i++){
//         lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
//     }
//     __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

//     for (int i = 0; i < mf_info->params.epoch; i++){
//         lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
//     }
    
//     // //* Grouping on CPU
//     // mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
//     // mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);

//     int update_vector_size = 128;
//     int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
//     int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
//     unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
//     unsigned int div = mf_info->params.thread_block_size/32;
    
//     // unsigned int error_kernel_work_groups = 2048;
//     unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
//     unsigned int group_error_size = error_kernel_work_groups;
//     unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
//     unsigned int seg_size = 32;
//     float* d_e_group;
//     cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
 
//     // cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
//     // cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);

//     __half* half_p = (__half*)malloc(sizeof(__half) * mf_info->max_user * mf_info->params.k); 
//     __half* half_q = (__half*)malloc(sizeof(__half) * mf_info->max_item * mf_info->params.k);
//     __half* d_half_p;
//     __half* d_half_q;

//     cudaMalloc(&d_half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k);    
//     cudaMalloc(&d_half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k);
    
//     conversion_features_half((short*)half_p, sgd_info->p ,mf_info->max_user, mf_info->params.k);
//     conversion_features_half((short*)half_q, sgd_info->q ,mf_info->max_item, mf_info->params.k);
    
//     //! ======================================== debug ================================================
//     float p_acc = 0;
//     float q_acc = 0;
//     for (int i = 0; i < mf_info->max_user * mf_info->params.k; i++){
//         p_acc += __half2float(half_p[i]);
//     }

//     for (int i = 0; i < mf_info->max_item * mf_info->params.k; i++){
//         q_acc += __half2float(half_q[i]);
//     }
//     cout << "p acc : " << p_acc << endl;
//     cout << "q acc : " << q_acc << endl;

//     // return;



//     cudaMemcpy(d_half_p, half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_half_q, half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);   
//     //* Initialize random states
//     curandState* d_rand_state;
//     cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
//     init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
//     cudaDeviceSynchronize();
//     gpuErr(cudaPeekAtLastError());
//     __half lambda_half = __float2half_rn(mf_info->params.lambda);

//     float* grad_mean_array_user;
//     float* grad_mean_array_item;
//     float* d_grad_mean_array_user;
//     float* d_grad_mean_array_item;
//     unsigned int* update_cnt_array_user;
//     unsigned int* update_cnt_array_item;
//     unsigned int* d_update_cnt_array_user;
//     unsigned int* d_update_cnt_array_item;
    
//     cudaMallocHost(&update_cnt_array_user, sizeof(unsigned int)*mf_info->max_user);
//     cudaMallocHost(&update_cnt_array_item, sizeof(unsigned int)*mf_info->max_item);
//     cudaMallocHost(&grad_mean_array_user, sizeof(float)*mf_info->max_user);
//     cudaMallocHost(&grad_mean_array_item, sizeof(float)*mf_info->max_item);

//     cudaMalloc(&d_update_cnt_array_user, sizeof(unsigned int)*mf_info->max_user);    
//     cudaMalloc(&d_update_cnt_array_item, sizeof(unsigned int)*mf_info->max_item);
//     cudaMalloc(&d_grad_mean_array_user, sizeof(float)*mf_info->max_user);    
//     cudaMalloc(&d_grad_mean_array_item, sizeof(float)*mf_info->max_item);
//     unsigned int block_num = mf_info->params.num_workers/div;
//     unsigned int *acc_user_group_error;
//     unsigned int *acc_item_group_error;
//     unsigned int *d_acc_user_group_error;
//     unsigned int *d_acc_item_group_error;
//     // unsigned int* count_num_of_ratings_user = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
//     // unsigned int* count_num_of_ratings_item = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
//     // pair<float, unsigned int>* mean_grad_user = (pair<float, unsigned int>*)malloc(sizeof(pair<float, unsigned int>) * mf_info->max_user);
//     // pair<float, unsigned int>* mean_grad_item = (pair<float, unsigned int>*)malloc(sizeof(pair<float, unsigned int>) * mf_info->max_item);
//     // for (int i = 0; i < mf_info->max_user; i++) count_num_of_ratings_user[i] = 0;
//     // for (int i = 0; i < mf_info->max_item; i++) count_num_of_ratings_item[i] = 0;
//     size_t shared_mem_size = (4*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + (sizeof(unsigned char) * (mf_info->user_group_num + mf_info->item_group_num));
//     double error_computation_time = 0;
//     double precision_switching_exec_time = 0;
//     double sgd_update_execution_time = 0;
//     for (int e = 0; e < mf_info->params.epoch; e++){
//         if (e < 20){
//             std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
//             half_sgd_k128_kernel_hogwild_warp32_lrate<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
//                                                         mf_info->d_R,
//                                                         mf_info->n,
//                                                         d_half_p,
//                                                         d_half_q,
//                                                         d_rand_state,
//                                                         lr_decay_arr_half[e],
//                                                         mf_info->params.k,
//                                                         1,
//                                                         e,
//                                                         update_count,
//                                                         update_vector_size,
//                                                         lambda_half
//                                                         );
//             cudaDeviceSynchronize();
//             sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();

//             gpuErr(cudaPeekAtLastError());
//             cudaMemcpy(half_p, d_half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
//             cudaMemcpy(half_q, d_half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
            
//             transform_feature_vector_half2float((short*)half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
//             transform_feature_vector_half2float((short*)half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);

//             p_acc = 0;
//             q_acc = 0;
//             for (int i = 0; i < mf_info->max_user * mf_info->params.k; i++){
//                 p_acc += __half2float(half_p[i]);
//             }

//             for (int i = 0; i < mf_info->max_item * mf_info->params.k; i++){
//                 q_acc += __half2float(half_q[i]);
//             }

//             cout << "p acc : " << p_acc << endl;
//             cout << "q acc : " << q_acc << endl;
//         }else if (e == 20){
//             for (int i = 0; i < mf_info->max_user; i++) {grad_mean_array_user[i] = 0.f; update_cnt_array_user[i] = 0;}
//             for (int i = 0; i < mf_info->max_item; i++) {grad_mean_array_item[i] = 0.f; update_cnt_array_item[i] = 0;}

//             cudaMemcpy(d_update_cnt_array_user, update_cnt_array_user, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyHostToDevice);
//             cudaMemcpy(d_update_cnt_array_item, update_cnt_array_item, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyHostToDevice);
//             cudaMemcpy(d_grad_mean_array_user, grad_mean_array_user, sizeof(float) * mf_info->max_user, cudaMemcpyHostToDevice);
//             cudaMemcpy(d_grad_mean_array_item, grad_mean_array_item, sizeof(float) * mf_info->max_item, cudaMemcpyHostToDevice);
//             std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
//             half_sgd_k128_kernel_hogwild_warp32_lrate_grad_mean_user_item<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
//                                                             mf_info->d_R,
//                                                             mf_info->n,
//                                                             d_half_p,
//                                                             d_half_q,
//                                                             d_rand_state,
//                                                             lr_decay_arr_half[e],
//                                                             mf_info->params.k,
//                                                             1,
//                                                             e,
//                                                             update_count,
//                                                             update_vector_size,
//                                                             lambda_half,
//                                                             d_grad_mean_array_user,
//                                                             d_grad_mean_array_item,
//                                                             d_update_cnt_array_user,
//                                                             d_update_cnt_array_item,
//                                                             0
//                                                             );
//             cudaDeviceSynchronize();
//             sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();

//             cudaMemcpy(update_cnt_array_user, d_update_cnt_array_user, sizeof(unsigned int)*mf_info->max_user, cudaMemcpyDeviceToHost);
//             cudaMemcpy(update_cnt_array_item, d_update_cnt_array_item, sizeof(unsigned int)*mf_info->max_item, cudaMemcpyDeviceToHost);
//             cudaMemcpy(grad_mean_array_user, d_grad_mean_array_user, sizeof(float)*mf_info->max_user, cudaMemcpyDeviceToHost);
//             cudaMemcpy(grad_mean_array_item, d_grad_mean_array_item, sizeof(float)*mf_info->max_item, cudaMemcpyDeviceToHost); 
            
//             for (int i = 0; i < mf_info->max_user; i++) {grad_mean_array_user[i] = grad_mean_array_user[i] / ((float)update_cnt_array_user[i] * mf_info->params.k);}
//             for (int i = 0; i < mf_info->max_item; i++) {grad_mean_array_item[i] = grad_mean_array_item[i] / ((float)update_cnt_array_item[i] * mf_info->params.k);}
            
//             cudaMemcpy(d_grad_mean_array_user, grad_mean_array_user, sizeof(float) * mf_info->max_user, cudaMemcpyHostToDevice);
//             cudaMemcpy(d_grad_mean_array_item, grad_mean_array_item, sizeof(float) * mf_info->max_item, cudaMemcpyHostToDevice);
//             cudaMallocHost(&mf_info->user2idx, sizeof(unsigned int) * mf_info->max_user);
//             cudaMallocHost(&mf_info->item2idx, sizeof(unsigned int) * mf_info->max_item);
//             cudaMalloc(&mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user);
//             cudaMalloc(&mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item);
    
//             unsigned int num_groups = 10000;

//             init_idx_arr<<<num_groups, 512>>>(mf_info->d_user2idx, mf_info->max_user);
//             init_idx_arr<<<num_groups, 512>>>(mf_info->d_item2idx, mf_info->max_item);
//             cudaDeviceSynchronize();
//             cudaMemcpy(mf_info->user2idx, mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
//             cudaMemcpy(mf_info->item2idx, mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);

//             thrust::device_ptr<float> thrust_d_user2grad(d_grad_mean_array_user);
//             thrust::device_ptr<float> thrust_d_item2grad(d_grad_mean_array_item);
//             thrust::device_ptr<unsigned int> thrust_d_user2idx(mf_info->d_user2idx);
//             thrust::device_ptr<unsigned int> thrust_d_item2idx(mf_info->d_item2idx);
//             thrust::sort_by_key(thrust_d_user2grad, thrust_d_user2grad + mf_info->max_user, thrust_d_user2idx, thrust::greater<float>());
//             thrust::sort_by_key(thrust_d_item2grad, thrust_d_item2grad + mf_info->max_item, thrust_d_item2idx, thrust::greater<float>());
//             cudaDeviceSynchronize();

//             // cudaMemcpy(d_grad_mean_array_user, grad_mean_array_user, sizeof(float) * mf_info->max_user, cudaMemcpyDeviceToHost);
//             // cudaMemcpy(d_grad_mean_array_item, grad_mean_array_item, sizeof(float) * mf_info->max_item, cudaMemcpyDeviceToHost);
//             cudaMemcpy(mf_info->user2idx, mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
//             cudaMemcpy(mf_info->item2idx, mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);

//             //* Grouping on CPU
//             mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
//             mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
            
//             //* Grouping methods
//             split_group_based_equal_size(mf_info);
//             check_group_cnt(mf_info);

//             generate_map_idx_info(mf_info);
//             // check_group_cnt(mf_info);

//             cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
//             cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
//             cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
//             cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

//             cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
//             cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
//             gpuErr(cudaPeekAtLastError());

//             cudaMemcpy(half_p, d_half_p, sizeof(__half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
//             cudaMemcpy(half_q, d_half_q, sizeof(__half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);        
//             transform_feature_vector_half2float((short*)half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
//             transform_feature_vector_half2float((short*)half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);

//             cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
//             cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);

//             cpy2grouped_parameters_gpu(mf_info, sgd_info);

//             cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
//             cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
//             cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
//             cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
//             cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
//             cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
//             cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
//             cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);
//             cudaMalloc(&d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
//             cudaMalloc(&d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);
//             cudaMallocHost(&acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num);
//             cudaMallocHost(&acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num);

//             // transform_feature_vector_half2float((short*)half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
//             // transform_feature_vector_half2float((short*)half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
            
//             for (int i = 0; i < mf_info->max_user; i++) {update_cnt_array_user[i] = 0;}
//             for (int i = 0; i < mf_info->max_item; i++) {update_cnt_array_item[i] = 0;}

//             cudaMemcpy(d_update_cnt_array_user, update_cnt_array_user, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyHostToDevice);
//             cudaMemcpy(d_update_cnt_array_item, update_cnt_array_item, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyHostToDevice);
//         }else {
//             std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();

//             sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
//                                                         mf_info->d_R,
//                                                         mf_info->n,
//                                                         (void**)sgd_info->d_user_group_ptr,
//                                                         (void**)sgd_info->d_item_group_ptr,
//                                                         d_rand_state,
//                                                         lr_decay_arr_half[e],
//                                                         mf_info->params.k,
//                                                         1,
//                                                         e,
//                                                         update_count,
//                                                         update_vector_size,
//                                                         lambda_half,
//                                                         mf_info->d_user_index_info,
//                                                         mf_info->d_item_index_info,
//                                                         mf_info->d_user_group_prec_info,
//                                                         mf_info->d_item_group_prec_info,
//                                                         d_acc_user_group_error,
//                                                         d_acc_item_group_error,
//                                                         d_update_cnt_array_user,
//                                                         d_update_cnt_array_item,
//                                                         first_sample_rating_idx,
//                                                         mf_info->user_group_num,
//                                                         mf_info->item_group_num
//                                                         );
//             cudaDeviceSynchronize();
//             sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();

//             //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
//             unsigned user_group_start_idx = 0;
//             for (int i = 0; i < mf_info->user_group_num; i++){
//                 unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
//                 if (mf_info->user_group_prec_info[i] == 0) {
//                     cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
//                     gpuErr(cudaPeekAtLastError());
//                 }
//                 else {
//                     cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
//                     gpuErr(cudaPeekAtLastError());
//                 }
//             }
//             gpuErr(cudaPeekAtLastError());

//             unsigned item_group_start_idx = 0;
//             for (int i = 0; i < mf_info->item_group_num; i++){
//                 unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
//                 if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
//                 else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
//             }
//             gpuErr(cudaPeekAtLastError());

//             std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

//             for (int i = 0; i < mf_info->max_user; i++){
//                 unsigned int user_group = mf_info->user_index_info[i].g;
//                 unsigned int row = mf_info->user_index_info[i].v;
//                 if (mf_info->user_group_prec_info[user_group] == 0){
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
//                     }     
//                 }
//                 else {
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
//                     }
//                 }
//             }

//             for (int i = 0; i < mf_info->max_item; i++){
//                 unsigned int item_group = mf_info->item_index_info[i].g;
//                 unsigned int row = mf_info->item_index_info[i].v;
//                 if (mf_info->item_group_prec_info[item_group] == 0){
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
//                     }
//                 }
//                 else{
//                     for (int k = 0; k < mf_info->params.k; k++){
//                         sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
//                     }
//                 }
//             }
//         // }
//             std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
//             cudaMemcpy(acc_user_group_error, d_acc_user_group_error, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
//             cudaMemcpy(acc_item_group_error, d_acc_item_group_error, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
//             cudaMemcpy(update_cnt_array_user, d_update_cnt_array_user, sizeof(unsigned int) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
//             cudaMemcpy(update_cnt_array_item, d_update_cnt_array_item, sizeof(unsigned int) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
//             double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
//             error_computation_time += error_transfer_time;
            
//             cout << "Transfer time " << error_transfer_time << endl;
//             cout << "\n<User groups>\n";
//             error_computation_start_time = std::chrono::system_clock::now();

//             for (int i = 0; i < mf_info->user_group_num; i++){
//                 unsigned int each_group_error_acc = 0;
//                 unsigned int each_group_cnt_acc = 0;
//                 if (mf_info->user_group_prec_info[i] == 0){
//                     for (int j = 0; j < block_num; j++){
//                         each_group_error_acc += acc_user_group_error[i * block_num + j];
//                         each_group_cnt_acc += update_cnt_array_user[i * block_num + j];
//                     }
//                     // if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k);
//                     if (each_group_cnt_acc != 0) mf_info->user_group_error[i] = (each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k))*log2((each_group_cnt_acc/(float)mf_info->n) + 2.0f);
//                     else mf_info->user_group_error[i] = 0;
//                 }
//                 else{
//                     mf_info->user_group_error[i] = -1;
//                 }
//                 // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->user_group_error[i] << endl;
//                 cout << mf_info->user_group_error[i] << " ";

//             }

//             cout << "\n<Item groups>\n";
//             for (int i = 0; i < mf_info->item_group_num; i++){
//                 unsigned int each_group_error_acc = 0;
//                 unsigned int each_group_cnt_acc = 0;
//                 if (mf_info->item_group_prec_info[i] == 0){
//                     for (int j = 0; j < block_num; j++){
//                         each_group_error_acc += acc_item_group_error[i * block_num + j];
//                         each_group_cnt_acc += update_cnt_array_item[i * block_num + j];
//                     }
//                     // if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k); 
//                     if (each_group_cnt_acc != 0) mf_info->item_group_error[i] = (each_group_error_acc/(float)(each_group_cnt_acc * mf_info->params.k))*log2((each_group_cnt_acc/(float)mf_info->n) + 2.0f);

//                     else mf_info->item_group_error[i] = 0;
//                 }
//                 else{
//                     mf_info->item_group_error[i] = -1;
//                 }
//                 // cout << "Group " << i + 1 << " : " << each_group_cnt_acc << "\t" << mf_info->item_group_error[i] << endl;
//                 cout << mf_info->item_group_error[i] << " ";
//             }
//             cout << "\n";
//             error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
//             std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
//             precision_switching_by_groups(mf_info, sgd_info);
//             precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
//         }
//         float rmse = 0;
//         rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
//         cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
//     }
//     cout << "Parameters update per epoch : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
//     cout << "Total parameters update     : " << sgd_update_execution_time << endl; 

// }


void grouped_sgd_training_map_based_eval_compute_precision(Mf_info* mf_info, SGD* sgd_info){
    //* Transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    Node* d_test_COO;
    cudaMalloc(&d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);

    //* Histogram on GPU
    double rating_histogram_execution_time = 0;
    std::chrono::time_point<std::chrono::system_clock> rating_histogram_start_point = std::chrono::system_clock::now();
    user_item_rating_histogram(mf_info);
    rating_histogram_execution_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - rating_histogram_start_point).count();

    //* Grouping on CPU
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);

    //* Grouping methods
    double grouping_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    if (mf_info->grouping_method == 1) split_group_based_equal_size(mf_info);
    else if (mf_info->grouping_method == 2) split_group_based_rating_num(mf_info);
    else if (mf_info->grouping_method == 3) split_group_based_rating_num_exp(mf_info);
    else if (mf_info->grouping_method == 4) split_group_based_equal_size_not_strict(mf_info);
    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();

    // check_group_cnt(mf_info);

    //* Generating index
    double generate_map_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> generate_map_start_point = std::chrono::system_clock::now();
    generate_map_idx_info(mf_info);
    generate_map_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - generate_map_start_point).count();

    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    // sgd_info->user_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_d_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    // sgd_info->user_group_ptr = (void**)malloc(sizeof(void*) * mf_info->user_group_num);
    // sgd_info->item_group_ptr = (void**)malloc(sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_d_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_d_ptr, sizeof(void*) * mf_info->item_group_num);
    cudaMallocHost(&sgd_info->user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMallocHost(&sgd_info->item_group_ptr, sizeof(void*) * mf_info->item_group_num);

    cudaMalloc(&sgd_info->d_user_group_ptr, sizeof(void*) * mf_info->user_group_num);
    cudaMalloc(&sgd_info->d_item_group_ptr, sizeof(void*) * mf_info->item_group_num);
    
    // cpy2grouped_parameters(mf_info, sgd_info);
    double cpy2grouped_parameters_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> cpy2grouped_parameters_start_point = std::chrono::system_clock::now();
    cpy2grouped_parameters_gpu(mf_info, sgd_info);
    cpy2grouped_parameters_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - cpy2grouped_parameters_start_point).count();

    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    __half* lr_decay_arr_half = (__half*)malloc(sizeof(__half)*mf_info->params.epoch);

    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr_half[i] = __float2half_rn(lr_decay_arr[i]);
    }

    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    double sgd_update_execution_time = 0;
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    __half lambda_half = __float2half_rn(mf_info->params.lambda);
    
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
    
    // cout  << "User group num : " << mf_info->user_group_num;

    // cout  << "Item group num : " << mf_info->item_group_num;
    

    // mf_info->user_group_prec_info = new unsigned char[mf_info->user_group_num];
    // mf_info->item_group_prec_info = new unsigned char[mf_info->item_group_num];
    cudaMallocHost(&mf_info->user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMalloc(&mf_info->d_user_group_prec_info, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_prec_info, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMemset(mf_info->d_user_group_prec_info, 0, sizeof(unsigned char) * mf_info->user_group_num);
    cudaMemset(mf_info->d_item_group_prec_info, 0, sizeof(unsigned char) * mf_info->item_group_num);
    cudaMallocHost(&mf_info->user_group_error, sizeof(float) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_error, sizeof(float) * mf_info->item_group_num);

    float *grad_sum_norm_p;
    float *grad_sum_norm_q;
    float *d_grad_sum_norm_p;
    float *d_grad_sum_norm_q;
    float *norm_sum_p;
    float *norm_sum_q;
    float *d_norm_sum_p;
    float *d_norm_sum_q;

    // unsigned int block_num = mf_info->params.num_workers/div;
    // cudaMalloc(&d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    // cudaMalloc(&d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    // cudaMallocHost(&grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num);
    // cudaMallocHost(&grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num);
    // cudaMalloc(&d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    // cudaMalloc(&d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);
    // cudaMallocHost(&norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num);
    // cudaMallocHost(&norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num);

    double error_computation_time = 0;
    double precision_switching_exec_time = 0;
    size_t shared_mem_size = (2*sizeof(unsigned int)*(mf_info->user_group_num + mf_info->item_group_num)) + 
                             (sizeof(unsigned char)*(mf_info->user_group_num + mf_info->item_group_num));
    double rmse;
    // map<unsigned int, vector<unsigned int>> user_switching_log;
    // map<unsigned int, vector<unsigned int>> item_switching_log;
    // vector<vector<double>> user_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->user_group_num, 0));
    // vector<vector<double>> item_grad_diversity_log(mf_info->params.epoch, vector<double>(mf_info->item_group_num, 0));
// #ifdef TEST

    for (int i = 0; i < mf_info->user_group_num; i++) mf_info->user_group_error[i] = mf_info->error_threshold + 1;
    for (int i = 0; i < mf_info->item_group_num; i++) mf_info->item_group_error[i] = mf_info->error_threshold - 1;
    
    if (mf_info->user_group_error[0] < mf_info->error_threshold) cout << "User FP32" << endl;
    if (mf_info->item_group_error[0] < mf_info->error_threshold) cout << "Item FP32" << endl;

    precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
    for (int e = 0; e < mf_info->params.epoch; e++){

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_eval_eval_compute_precision<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size, shared_mem_size>>>(
                                                    mf_info->d_R,
                                                    mf_info->n,
                                                    (void**)sgd_info->d_user_group_ptr,
                                                    (void**)sgd_info->d_item_group_ptr,
                                                    d_rand_state,
                                                    lr_decay_arr_half[e],
                                                    mf_info->params.k,
                                                    1,
                                                    e,
                                                    update_count,
                                                    update_vector_size,
                                                    lambda_half,
                                                    mf_info->d_user_index_info,
                                                    mf_info->d_item_index_info,
                                                    mf_info->d_user_group_prec_info,
                                                    mf_info->d_item_group_prec_info,
                                                    mf_info->user_group_num,
                                                    mf_info->item_group_num
                                                    );
        cudaDeviceSynchronize();
        sgd_update_execution_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        gpuErr(cudaPeekAtLastError());    
        
        //* TRANSFER GROUPED PARAMETER FROM GPU TO HOST
        unsigned user_group_start_idx = 0;
        for (int i = 0; i < mf_info->user_group_num; i++){
            unsigned int group_params_size = mf_info->user_group_size[i] * mf_info->params.k;
            if (mf_info->user_group_prec_info[i] == 0) {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
            else {
                cudaMemcpy(sgd_info->user_group_ptr[i], sgd_info->user_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
                gpuErr(cudaPeekAtLastError());
            }
        }
        gpuErr(cudaPeekAtLastError());

        unsigned item_group_start_idx = 0;
        for (int i = 0; i < mf_info->item_group_num; i++){
            unsigned int group_params_size = mf_info->item_group_size[i] * mf_info->params.k; 
            if (mf_info->item_group_prec_info[i] == 0) cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(__half) * group_params_size, cudaMemcpyDeviceToHost);
            else cudaMemcpy(sgd_info->item_group_ptr[i], sgd_info->item_group_d_ptr[i], sizeof(float) * group_params_size, cudaMemcpyDeviceToHost);
        }
        gpuErr(cudaPeekAtLastError());

        std::chrono::time_point<std::chrono::system_clock> group_params_to_origin_start_time = std::chrono::system_clock::now();

        for (int i = 0; i < mf_info->max_user; i++){
            unsigned int user_group = mf_info->user_index_info[i].g;
            unsigned int row = mf_info->user_index_info[i].v;
            if (mf_info->user_group_prec_info[user_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k]);
                }     
            }
            else {
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->p[i*mf_info->params.k + k] =  ((float*)(sgd_info->user_group_ptr[user_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        for (int i = 0; i < mf_info->max_item; i++){
            unsigned int item_group = mf_info->item_index_info[i].g;
            unsigned int row = mf_info->item_index_info[i].v;
            if (mf_info->item_group_prec_info[item_group] == 0){
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  __half2float(((__half*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k]);
                }
            }
            else{
                for (int k = 0; k < mf_info->params.k; k++){
                    sgd_info->q[i*mf_info->params.k + k] =  ((float*)(sgd_info->item_group_ptr[item_group]))[(row * mf_info->params.k) + k];
                }
            }
        }

        // std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        // cudaMemcpy(grad_sum_norm_p, d_grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        // cudaMemcpy(grad_sum_norm_q, d_grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_sum_p, d_norm_sum_p, sizeof(float) * block_num * mf_info->user_group_num, cudaMemcpyDeviceToHost);
        // cudaMemcpy(norm_sum_q, d_norm_sum_q, sizeof(float) * block_num * mf_info->item_group_num, cudaMemcpyDeviceToHost);
        // double error_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        // error_computation_time += error_transfer_time;
        
        // cout << "Transfer time " << error_transfer_time << endl;
        // cout << "\n<User groups>\n";
        // for (int i = 0; i < mf_info->user_group_num; i++){
        //     error_computation_start_time = std::chrono::system_clock::now();
        //     float each_group_grad_sum_norm_acc = 0;
        //     float each_group_norm_acc = 0;
        //     if (mf_info->user_group_prec_info[i] == 0){
                
        //         for (int k = 0; k < mf_info->params.k; k++){
        //             each_group_grad_sum_norm_acc+=powf(grad_sum_norm_p[i*mf_info->params.k + k],2);
        //             grad_sum_norm_p[i*mf_info->params.k + k] = 0;
        //         }
        //         for (int j = 0; j < block_num; j++){
        //             // each_group_grad_sum_norm_acc += grad_sum_norm_p[i * block_num + j];
        //             each_group_norm_acc += norm_sum_p[i * block_num + j];
        //         }
        //         // cout << each_group_grad_sum_norm_acc << " ";
        //         mf_info->user_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
        //     }
        //     else{
        //         mf_info->user_group_error[i] = UINT_MAX;
        //     }
        //     error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
        //     //! log code
        //     if (mf_info->user_group_prec_info[i] == 0 && mf_info->user_group_error[i] < mf_info->error_threshold && e >= start_idx) user_switching_log[e].push_back(i);

        //     if (mf_info->user_group_error[i] == UINT_MAX) {
        //         cout << "-1" << " ";
        //         user_grad_diversity_log[e][i] = -1;
        //     }           
        //     else {
        //         cout << mf_info->user_group_error[i] << " ";
        //         user_grad_diversity_log[e][i] = mf_info->user_group_error[i];
        //     }
        // }

        // cout << "\n<Item groups>\n";
        // for (int i = 0; i < mf_info->item_group_num; i++){
        //     error_computation_start_time = std::chrono::system_clock::now();
        //     float each_group_grad_sum_norm_acc = 0;
        //     float each_group_norm_acc = 0;
        //     if (mf_info->item_group_prec_info[i] == 0){

        //         for (int k = 0; k < mf_info->params.k; k++){
        //             each_group_grad_sum_norm_acc+=powf(grad_sum_norm_q[i*mf_info->params.k + k],2);
        //             grad_sum_norm_q[i*mf_info->params.k + k] = 0;
        //         }

        //         for (int j = 0; j < block_num; j++){
        //             // each_group_grad_sum_norm_acc += grad_sum_norm_q[i * block_num + j];
        //             each_group_norm_acc += norm_sum_q[i * block_num + j];
        //         }
        //         mf_info->item_group_error[i] = each_group_norm_acc/(float)(each_group_grad_sum_norm_acc);
        //         // cout << each_group_grad_sum_norm_acc << " ";
        //     }
        //     else{
        //         mf_info->item_group_error[i] = UINT_MAX;
        //     }
        //     error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            
        //     //! log code
        //     if (mf_info->item_group_prec_info[i] == 0 && mf_info->item_group_error[i] < mf_info->error_threshold && e >= start_idx) item_switching_log[e].push_back(i);
            
        //     if (mf_info->item_group_error[i] == UINT_MAX) {
        //         cout << "-1" << " ";
        //         item_grad_diversity_log[e][i] = -1;
        //     }            
        //     else {
        //         cout << mf_info->item_group_error[i] << " ";
        //         item_grad_diversity_log[e][i] = mf_info->item_group_error[i];
        //     }
        // }
        // cout << "\n";
        // // error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
        
        // std::chrono::time_point<std::chrono::system_clock> precision_switching_start_point = std::chrono::system_clock::now();
        // if (e >= start_idx) precision_switching_by_groups_grad_diversity(mf_info, sgd_info);
        // precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();

        rmse = gpu_test_rmse(mf_info, sgd_info, d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        if (mf_info->params.epoch-1 == e) cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        //! Group error write code
        // string group_error_metric_output_file_path = string("./statistics/") + "User_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->user_group_error, rmse, mf_info->user_group_num, e+1, mf_info->params.epoch);
        // group_error_metric_output_file_path = string("./statistics/") + "Item_" + mf_info->out_file + ".txt";  
        // print_group_error_val(group_error_metric_output_file_path, mf_info->item_group_error, rmse, mf_info->item_group_num, e+1, mf_info->params.epoch);
        // precision_switching_start_point = std::chrono::system_clock::now();
        // cudaMemcpy(d_grad_sum_norm_p, grad_sum_norm_p, sizeof(float) * mf_info->params.k * mf_info->user_group_num, cudaMemcpyHostToDevice);
        // cudaMemcpy(d_grad_sum_norm_q, grad_sum_norm_q, sizeof(float) * mf_info->params.k * mf_info->item_group_num, cudaMemcpyHostToDevice);
        // precision_switching_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - precision_switching_start_point).count();
    }


    // //! RMSE write code
    // string group_error_metric_output_file_path = string("./statistics/") + mf_info->out_file + ".txt";  
    // print_rmse(group_error_metric_output_file_path, rmse);
    double preprocess_exec_time = rating_histogram_execution_time + grouping_exec_time + generate_map_exec_time + cpy2grouped_parameters_exec_time;
    cout << "\n<Preprocessing time (micro sec)>" << endl;
    // cout << "Rating histogram                 : " << rating_histogram_execution_time << endl;
    // cout << "Grouping                         : " << grouping_exec_time << endl;
    // cout << "Generate map idx                 : " << generate_map_exec_time << endl;
    // cout << "Copy to grouped params           : " << cpy2grouped_parameters_exec_time << endl;
    // cout << "Total preprocessing time         : " << preprocess_exec_time << endl;
    // cout << "\n<User & item parameter copy exec time (micro sec)>" << endl;
    // cout << "Precision switching              : " << precision_switching_exec_time / mf_info->params.epoch << endl; 
    // cout << "Total precision switching time   : " << precision_switching_exec_time << endl;
    // cout << "\n<User & item group error comp exec time (micro sec)>" << endl;
    // cout << "Error computation time           : " << error_computation_time / mf_info->params.epoch << endl; 
    // cout << "Total error computation time     : " << error_computation_time << endl;
    // cout << "\n<User & item parameter update exec time (micro sec)>" << endl;
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    // cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    // cout << "Total MF time(ms)                : " << (preprocess_exec_time + precision_switching_exec_time + error_computation_time + sgd_update_execution_time)/1000 << endl;

    string exec_rmse_output_file_path = string("./New_statistics/comp_precision_yahoo/rmse_time_for_comp_precision") + mf_info->out_file;  

    if (mf_info->user_group_error[0] < mf_info->error_threshold) {
        cout << "User FP32" << endl;
        exec_rmse_output_file_path = exec_rmse_output_file_path + "user_fp32";
    }else{
        exec_rmse_output_file_path = exec_rmse_output_file_path + "user_fp16";
    }

    if (mf_info->item_group_error[0] < mf_info->error_threshold) {
        cout << "Item FP32" << endl;
        exec_rmse_output_file_path = exec_rmse_output_file_path + "item_fp32";
    }else{
        exec_rmse_output_file_path = exec_rmse_output_file_path + "item_fp16";
    }
    exec_rmse_output_file_path = exec_rmse_output_file_path + ".txt";
    map<string, double> statistics_map;

    statistics_map["preprocess"] = 0 / 1000;
    statistics_map["switching"] = (0) / 1000;
    statistics_map["update"] = sgd_update_execution_time / mf_info->params.epoch / 1000;
    statistics_map["total"] = sgd_update_execution_time / mf_info->params.epoch / 1000;
    statistics_map["rmse"] = rmse;

    // string group_switching_log_output_file_path = string("./statistics/grouping/switching_log/atomic_ver_group_switching_log_") + mf_info->out_file + ".txt";
    // string group_diversity_log_output_file_path = string("./statistics/grouping/diversity_log/atomic_ver_group_diversity_log_") + mf_info->out_file + ".txt";
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
    // print_group_switching_log(group_switching_log_output_file_path, user_switching_log, item_switching_log);
    // print_grad_diversity_log(group_diversity_log_output_file_path, user_grad_diversity_log, item_grad_diversity_log);
// #endif
}

//! SVD CUDA에 있는 것으로 돌리기
void mixed_precision_training_mf(Mf_info* mf_info, SGD* sgd_info){
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    random_shuffle(mf_info->R, mf_info->R + mf_info->n);
    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);

    //* Mastser copy of user(item) embedding
    cudaMalloc(&(sgd_info->d_p), sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&(sgd_info->d_q), sizeof(float) * mf_info->max_item * mf_info->params.k);
        
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    // transform_feature_vector_half2float((short*)sgd_info->half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
    // transform_feature_vector_half2float((short*)sgd_info->half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
    // cudaMemcpy(sgd_info->d_p, sgd_info->p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyHostToDevice);
    // cudaMemcpy(sgd_info->d_q, sgd_info->q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyHostToDevice);
    
    transition_params_half2float(mf_info, sgd_info);

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

    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);  

    //* Scaling factor
    float scaling_factor = 1000;
    __half scaled_lambda = __float2half_rn(mf_info->params.lambda * scaling_factor);
    double rmse;
    double sgd_update_execution_time = 0;

    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        rev_origin_sgd_k128_kernel_hogwild_warp32_lrate_loss_scaling<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                                mf_info->d_R,
                                                mf_info->n,
                                                sgd_info->d_p,
                                                sgd_info->d_q,
                                                d_rand_state,
                                                lr_decay_arr[e],
                                                mf_info->params.k,
                                                1,
                                                e,
                                                update_count,
                                                update_vector_size,
                                                scaled_lambda,
                                                scaling_factor
                                                );
        cudaDeviceSynchronize();
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        cout << "Time per epoch : " << sgd_update_time_per_epoch << endl;

        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        // rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
    }

    cout << "Parameters update per epoch                     : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update                         : " << sgd_update_execution_time / 1000 << endl; 

}

//! ICML20 Muppet version
void muppet_training_mf(Mf_info* mf_info, SGD* sgd_info){
    //* Random shuffle and transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    double additional_info_init_exec_time = 0;
    double sgd_update_execution_time = 0;
    double error_computation_time = 0;

    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    float quantization_error = 0;
    float *sum_norms;
    float *d_sum_norms;
    float *sum_updated_val;
    float *d_sum_updated_val;

    float *sum_norms_epoch = new float[mf_info->params.epoch];
    float *sum_updated_val_epoch = new float[mf_info->params.k * mf_info->params.epoch];
    float *gradient_diversity_per_epoch = new float[mf_info->params.epoch];
    float *sum_updated_val_acc = new float[mf_info->params.k];

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMallocHost(&sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMalloc(&d_sum_updated_val, sizeof(float) * mf_info->params.k);
    cudaMallocHost(&sum_updated_val, sizeof(float) * mf_info->params.k);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();


    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;
    float initial_error = 1; 
    unsigned char cur_precision = 0;
    int resolution_size = 3;
    float max_grad_diversity = -1.f;
    float alpha  = 1.0f;
    float beta = 1.5f;
    float lambda_for_threshold = 0.3f;
    int gamma = 1;
    int violation_times = 0;

    vector<float> gradient_diversity_per_epoch_log;
    vector<float> gradient_diversity_ratio_per_epoch_log;
    vector<float> error_threshold_per_epoch_log;
    vector<float> rmse_per_epoch_log;
    vector<int> cur_precision_per_epoch_log;
    vector<int> violation_times_log;
    unsigned char bit_width_set[5] = {8, 12, 14 ,16, 32};
    int switched_epoch = -1;
    int precision_idx;
    if (mf_info->is_yahoo) precision_idx = 1;
    else precision_idx = 0;

    for (int e = 0; e < mf_info->params.epoch; e++){
        float decaying_threshold = alpha + (beta * exp(-1*lambda_for_threshold*(e)));
        error_threshold_per_epoch_log.push_back(decaying_threshold);
        cur_precision_per_epoch_log.push_back(bit_width_set[precision_idx]);

        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        //! clean device mem (not neccesary)
        for (int k = 0; k < mf_info->params.k; k++) sum_updated_val[k] = 0;
        cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        
        
        //! Round to nearest
        if (mf_info->params.k == 128){
            sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                1,
                                e,
                                update_count,
                                update_vector_size,
                                mf_info->params.lambda,
                                cur_precision,
                                first_sample_rating_idx,
                                d_sum_updated_val,
                                d_sum_norms,
                                (unsigned char)bit_width_set[precision_idx],
                                (unsigned char)bit_width_set[precision_idx]
                            );
        }else if (mf_info->params.k == 64){
            sgd_k64_kernel_hogwild_warp32_lrate_grad_diversity_muppet<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                1,
                                e,
                                update_count,
                                update_vector_size,
                                mf_info->params.lambda,
                                cur_precision,
                                first_sample_rating_idx,
                                d_sum_updated_val,
                                d_sum_norms,
                                (unsigned char)bit_width_set[precision_idx],
                                (unsigned char)bit_width_set[precision_idx]
                            );
        }
        //! Stochastic rounding version
        // sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet_stochastic_rounding<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
        //                     mf_info->d_R,
        //                     mf_info->n,
        //                     sgd_info->d_p,
        //                     sgd_info->d_q,
        //                     d_rand_state,
        //                     lr_decay_arr[e],
        //                     mf_info->params.k,
        //                     1,
        //                     e,
        //                     update_count,
        //                     update_vector_size,
        //                     mf_info->params.lambda,
        //                     cur_precision,
        //                     first_sample_rating_idx,
        //                     d_sum_updated_val,
        //                     d_sum_norms,
        //                     (unsigned char)bit_width_set[precision_idx],
        //                     (unsigned char)bit_width_set[precision_idx]
        //                 );

        cudaDeviceSynchronize(); 
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        cout << "per epoch " << sgd_update_time_per_epoch << endl;
        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        rmse_per_epoch_log.push_back(rmse);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        violation_times_log.push_back(violation_times);

#ifdef WRITE_FILE
        //! RMSE write code
        // string group_error_metric_output_file_path = string("./New_statistics/Muppet_icml/rmse_per_epoch_stochastic_rounding/icml_stochastic_rounding_rmse_per_epoch_") + mf_info->out_file + ".txt"; 
        string group_error_metric_output_file_path = string("./New_statistics/Muppet_icml/rmse_per_epoch/icml_rmse_per_epoch_") + mf_info->out_file + ".txt";   
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
#endif

        if (bit_width_set[precision_idx] == (unsigned char)32){
            gradient_diversity_per_epoch_log.push_back(0);
            gradient_diversity_ratio_per_epoch_log.push_back(0);     
            continue;
        };

        error_computation_start_time = std::chrono::system_clock::now();
        //! Get grad from device to compute diversity by epoch
        cudaMemcpy(sum_norms, d_sum_norms, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
        cudaMemcpy(sum_updated_val, d_sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);
        
        //! Save
        float norm_acc = 0 ;
        
        for (int w = 0; w < mf_info->params.num_workers; w++) norm_acc += sum_norms[w];
        sum_norms_epoch[e] = norm_acc;
        for (int k = 0; k < mf_info->params.k; k++) {
            sum_updated_val_acc[k] = 0;
            sum_updated_val_epoch[e * mf_info->params.k + k] = sum_updated_val[k];
        }
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();

        //! Compute metric
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
            gradient_diversity_per_epoch_log.push_back(grad_diversity_this_epoch);
            gradient_diversity_ratio_per_epoch_log.push_back(grad_ratio);
        }else{
            gradient_diversity_per_epoch_log.push_back(0);
            gradient_diversity_ratio_per_epoch_log.push_back(1);           
        }
        // cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < mf_info->params.epoch; i++){
        cout << "Epoch " << setw(3) << left << (i + 1)
             << "  RMSE " << setw(10) << left << fixed << setprecision(6) << rmse_per_epoch_log[i]
             << "  Grad_div " << setw(10) << left << fixed << setprecision(6) << gradient_diversity_per_epoch_log[i] 
             << "  Grad_ratio " << setw(10) << left << fixed  << setprecision(6) << gradient_diversity_ratio_per_epoch_log[i] 
             << "  Threshold " << setw(10) << left << fixed  << setprecision(6) << error_threshold_per_epoch_log[i] 
             << "  Cur_bitwidth " << setw(2) << left << cur_precision_per_epoch_log[i]
             << "  Violation times " << setw(1) << left << violation_times_log[i]
             << endl;
    }

    cout << "Error computation time           : " << error_computation_time << endl; 
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (error_computation_time + sgd_update_execution_time)/1000 << endl;
    
#ifdef WRITE_FILE
    double preprocess_exec_time = additional_info_init_exec_time;    
    map<string, double> statistics_map;

    statistics_map["preprocess"] = preprocess_exec_time / 1000;
    statistics_map["switching"] = error_computation_time / 1000;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = (preprocess_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
    statistics_map["rmse"] = rmse;

    // string exec_rmse_output_file_path = string("./New_statistics/Muppet_icml/time_rmse_stochastic_rounding/icml_stochastic_rounding_time_rmse_") + mf_info->out_file + ".txt";  
    string exec_rmse_output_file_path = string("./New_statistics/Muppet_icml/time_rmse/icml_time_rmse_") + mf_info->out_file + ".txt";  

    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
#endif
}


//! Average of user, item gradient diversity version
// void muppet_training_mf(Mf_info* mf_info, SGD* sgd_info){
//     //* Random shuffle and transfer rating triplets to GPU 
//     random_shuffle(mf_info->R, mf_info->R + mf_info->n);

//     cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
//     cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
//     gpuErr(cudaPeekAtLastError());
    
//     //* Convert testset to COO format
//     mf_info->test_COO = test_set_preprocess(mf_info);
//     cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
//     gpuErr(cudaPeekAtLastError());
//     cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
//     //* Initialize random states
//     curandState* d_rand_state;
//     cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
//     init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
//     cudaDeviceSynchronize();
//     gpuErr(cudaPeekAtLastError());

//     //* Learning rate initialization
//     float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
//     for (int i = 0; i < mf_info->params.epoch; i++){
//         lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
//     }
    
//     int update_vector_size = 128;
//     int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
//     int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
//     unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
//     cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
//     cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
//     cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
//     cout << "Start SGD update..." << endl;
    
//     unsigned int div = mf_info->params.thread_block_size/32;
    
//     unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
//     unsigned int group_error_size = error_kernel_work_groups;
//     unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
//     unsigned int seg_size = 32;
//     float* d_e_group;
//     cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

//     double additional_info_init_exec_time = 0;
//     double sgd_update_execution_time = 0;
//     double error_computation_time = 0;

//     std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
//     float quantization_error = 0;
//     float *sum_norms;
//     float *sum_norms_item;

//     float *d_sum_norms;
//     float *d_sum_norms_item;

//     float *sum_updated_val;
//     float *sum_updated_val_item;

//     float *d_sum_updated_val;
//     float *d_sum_updated_val_item;

//     float *sum_norms_epoch = new float[mf_info->params.epoch];
//     float *sum_norms_epoch_item = new float[mf_info->params.epoch];

//     float *sum_updated_val_epoch = new float[mf_info->params.k * mf_info->params.epoch];
//     float *sum_updated_val_epoch_item = new float[mf_info->params.k * mf_info->params.epoch];

//     float *gradient_diversity_per_epoch = new float[mf_info->params.epoch];
    
//     float *sum_updated_val_acc = new float[mf_info->params.k];
//     float *sum_updated_val_acc_item = new float[mf_info->params.k];

//     unsigned int block_num = mf_info->params.num_workers/div;
//     cudaMalloc(&d_sum_norms, sizeof(float) * mf_info->params.num_workers);
//     cudaMalloc(&d_sum_norms_item, sizeof(float) * mf_info->params.num_workers);

//     cudaMallocHost(&sum_norms, sizeof(float) * mf_info->params.num_workers);
//     cudaMallocHost(&sum_norms_item, sizeof(float) * mf_info->params.num_workers);

//     cudaMalloc(&d_sum_updated_val, sizeof(float) * mf_info->params.k);
//     cudaMalloc(&d_sum_updated_val_item, sizeof(float) * mf_info->params.k);

//     cudaMallocHost(&sum_updated_val, sizeof(float) * mf_info->params.k);
//     cudaMallocHost(&sum_updated_val_item, sizeof(float) * mf_info->params.k);

//     additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

//     double rmse;    
//     int start_idx = 5;
//     unsigned int num_groups = 10000;
//     float initial_error = 1; 
//     unsigned char cur_precision = 0;
//     int resolution_size = 3;
//     float max_grad_diversity = -1.f;
//     float alpha  = 1.0f;
//     float beta = 1.5f;
//     float lambda_for_threshold = 0.3f;
//     int gamma = 1;
//     int violation_times = 0;

//     vector<float> gradient_diversity_per_epoch_log;
//     vector<float> gradient_diversity_ratio_per_epoch_log;
//     vector<float> error_threshold_per_epoch_log;
//     vector<float> rmse_per_epoch_log;
//     vector<int> cur_precision_per_epoch_log;
//     vector<int> violation_times_log;
//     unsigned char bit_width_set[1] = {8};
//     int switched_epoch = 0;
//     int precision_idx = 0;

//     for (int e = 0; e < mf_info->params.epoch; e++){
//         float decaying_threshold = alpha + (beta * exp(-1*lambda_for_threshold*(e)));
//         error_threshold_per_epoch_log.push_back(decaying_threshold);
//         cur_precision_per_epoch_log.push_back(bit_width_set[precision_idx]);

//         std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
//         //! clean device mem (not neccesary)
//         for (int k = 0; k < mf_info->params.k; k++) {
//             sum_updated_val[k] = 0;
//             sum_updated_val_item[k] = 0;
//         }
//         cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_sum_updated_val_item, sum_updated_val_item, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);

//         error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();

//         std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
//         sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
//                             mf_info->d_R,
//                             mf_info->n,
//                             sgd_info->d_p,
//                             sgd_info->d_q,
//                             d_rand_state,
//                             lr_decay_arr[e],
//                             mf_info->params.k,
//                             1,
//                             e,
//                             update_count,
//                             update_vector_size,
//                             mf_info->params.lambda,
//                             cur_precision,
//                             first_sample_rating_idx,
//                             d_sum_updated_val,
//                             d_sum_norms,
//                             d_sum_updated_val_item,
//                             d_sum_norms_item,
//                             (unsigned char)bit_width_set[precision_idx],
//                             (unsigned char)bit_width_set[precision_idx]
//                         );

//         cudaDeviceSynchronize(); 
//         double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
//         sgd_update_execution_time += sgd_update_time_per_epoch;

//         cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
//         cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
//         rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
//         rmse_per_epoch_log.push_back(rmse);
//         cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
//         violation_times_log.push_back(violation_times);

// #ifdef WRITE_FILE
//         //! RMSE write code
//         string group_error_metric_output_file_path = string("./New_statistics/Muppet_icml/rmse_per_epoch/icml_rmse_per_epoch_") + mf_info->out_file + ".txt";  
//         print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);
// #endif

//         if (bit_width_set[precision_idx] == (unsigned char)32){
//             gradient_diversity_per_epoch_log.push_back(0);
//             gradient_diversity_ratio_per_epoch_log.push_back(0);     
//             continue;
//         };

//         error_computation_start_time = std::chrono::system_clock::now();
//         //! Get grad from device to compute diversity by epoch
//         cudaMemcpy(sum_norms, d_sum_norms, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
//         cudaMemcpy(sum_updated_val, d_sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);
//         cudaMemcpy(sum_norms_item, d_sum_norms_item, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
//         cudaMemcpy(sum_updated_val_item, d_sum_updated_val_item, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);        
//         //! Save
//         float norm_acc = 0 ;
//         float norm_acc_item = 0;
//         for (int w = 0; w < mf_info->params.num_workers; w++){
//             norm_acc += sum_norms[w];
//             norm_acc_item += sum_norms_item[w];
//         }
//         sum_norms_epoch[e] = norm_acc;
//         sum_norms_epoch_item[e] = norm_acc_item;

//         for (int k = 0; k < mf_info->params.k; k++) {
//             sum_updated_val_acc[k] = 0;
//             sum_updated_val_acc_item[k] = 0;
//             sum_updated_val_epoch[e * mf_info->params.k + k] = sum_updated_val[k];
//             sum_updated_val_epoch_item[e * mf_info->params.k + k] = sum_updated_val_item[k];
//         }
//         error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();

//         //! Compute metric
//         if (e - switched_epoch >= resolution_size){
//             error_computation_start_time = std::chrono::system_clock::now();
//             float sum_norms_resolution = 0;
//             float sum_norms_resolution_item = 0;
//             float sum_updated_val_norm = 0;
//             float sum_updated_val_norm_item = 0;
//             for (int wi = e - resolution_size; wi <= e; wi++){
//                 sum_norms_resolution += sum_norms_epoch[wi];
//                 sum_norms_resolution_item += sum_norms_epoch_item[wi];
//                 for (int k = 0; k < mf_info->params.k; k++) sum_updated_val_acc[k] += sum_updated_val_epoch[wi * mf_info->params.k + k];
//                 for (int k = 0; k < mf_info->params.k; k++) sum_updated_val_acc_item[k] += sum_updated_val_epoch_item[wi * mf_info->params.k + k];
//             }

//             for (int k = 0; k < mf_info->params.k; k++) {
//                 sum_updated_val_norm += powf(sum_updated_val_acc[k],2);
//                 sum_updated_val_norm_item += powf(sum_updated_val_acc_item[k],2);
//             }

//             float grad_diversity_this_epoch = ((sum_norms_resolution/sum_updated_val_norm) + (sum_norms_resolution_item/sum_updated_val_norm_item))/2.0f;
//             max_grad_diversity = max_grad_diversity < grad_diversity_this_epoch ? grad_diversity_this_epoch : max_grad_diversity;
//             float grad_ratio = max_grad_diversity/grad_diversity_this_epoch;
            
//             if (grad_ratio > decaying_threshold && ++violation_times == gamma) {
//                 precision_idx++;
//                 switched_epoch = e;
//                 max_grad_diversity = -1.f;
//                 violation_times = 0;
//             }
//             error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
//             gradient_diversity_per_epoch_log.push_back(grad_diversity_this_epoch);
//             gradient_diversity_ratio_per_epoch_log.push_back(grad_ratio);
//         }else{
//             gradient_diversity_per_epoch_log.push_back(0);
//             gradient_diversity_ratio_per_epoch_log.push_back(0);           
//         }
//         // cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
//     }

//     for (int i = 0; i < mf_info->params.epoch; i++){
//         cout << "Epoch " << setw(3) << left << (i + 1)
//              << "  RMSE " << setw(10) << left << fixed << setprecision(6) << rmse_per_epoch_log[i]
//              << "  Grad_div " << setw(10) << left << fixed << setprecision(6) << gradient_diversity_per_epoch_log[i] 
//              << "  Grad_ratio " << setw(10) << left << fixed  << setprecision(6) << gradient_diversity_ratio_per_epoch_log[i] 
//              << "  Threshold " << setw(10) << left << fixed  << setprecision(6) << error_threshold_per_epoch_log[i] 
//              << "  Cur_bitwidth " << setw(2) << left << cur_precision_per_epoch_log[i]
//              << "  Violation times " << setw(1) << left << violation_times_log[i]
//              << endl;
//     }
//     cout << "Error computation time           : " << error_computation_time << endl; 
//     cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
//     cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
//     cout << "Total MF time(ms)                : " << (error_computation_time + sgd_update_execution_time)/1000 << endl;
    
// #ifdef WRITE_FILE
//     double preprocess_exec_time = additional_info_init_exec_time;    
//     map<string, double> statistics_map;

//     statistics_map["preprocess"] = preprocess_exec_time / 1000;
//     statistics_map["switching"] = error_computation_time / 1000;
//     statistics_map["update"] = sgd_update_execution_time / 1000;
//     statistics_map["total"] = (preprocess_exec_time + error_computation_time + sgd_update_execution_time) / 1000;
//     statistics_map["rmse"] = rmse;

//     string exec_rmse_output_file_path = string("./New_statistics/Muppet_icml/time_rmse/icml_time_rmse_") + mf_info->out_file + ".txt";  
    
//     print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
// #endif
// }

//! 기존 버전과 같이 수정하기
void muppet_training_mf_parameter_switching_version(Mf_info* mf_info, SGD* sgd_info){
    //* Random shuffle and transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    double additional_info_init_exec_time = 0;
    double error_computation_time = 0;

    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();
    float quantization_error = 0;
    float *sum_norms;
    float *d_sum_norms;
    float *sum_updated_val;
    float *d_sum_updated_val;

    float *sum_norms_epoch = new float[mf_info->params.epoch];
    float *sum_updated_val_epoch = new float[mf_info->params.k * mf_info->params.epoch];
    float *gradient_diversity_per_epoch = new float[mf_info->params.epoch];
    float *sum_updated_val_acc = new float[mf_info->params.k];

    unsigned int block_num = mf_info->params.num_workers/div;
    cudaMalloc(&d_sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMallocHost(&sum_norms, sizeof(float) * mf_info->params.num_workers);
    cudaMalloc(&d_sum_updated_val, sizeof(float) * mf_info->params.k);
    cudaMallocHost(&sum_updated_val, sizeof(float) * mf_info->params.k);
    additional_info_init_exec_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - additional_info_init_start_point).count();

    cudaMalloc(&sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k);
    cudaMalloc(&sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k);
    void* p = (void*)(sgd_info->d_half_p);
    void* q = (void*)(sgd_info->d_half_q);

    double precision_switching_and_error_comp_execution_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;
    float initial_error = 1; 
    unsigned char cur_precision = 0;
    int resolution_size = 3;
    float max_grad_diversity = -1.f;
    float alpha  = 1.0f;
    float beta = 1.5f;
    float gamma = 0.1f;
    
    vector<float> gradient_diversity_per_epoch_log;
    vector<float> gradient_diversity_ratio_per_epoch_log;
    vector<float> error_threshold_per_epoch_log;
    vector<float> rmse_per_epoch_log;
    vector<int> cur_precision_per_epoch_log;
    unsigned char bit_width_set[2] = {16, 32};
    int switched_epoch = 0;
    int precision_idx = 0;

    for (int e = 0; e < mf_info->params.epoch; e++){
        float decaying_threshold = alpha + (beta * exp(-1*gamma*(e)));
        error_threshold_per_epoch_log.push_back(decaying_threshold);
        cur_precision_per_epoch_log.push_back(bit_width_set[precision_idx]);
        
        std::chrono::time_point<std::chrono::system_clock> error_computation_start_time = std::chrono::system_clock::now();
        //! clean device mem (not neccesary)
        for (int k = 0; k < mf_info->params.k; k++) sum_updated_val[k] = 0;
        cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet_parameter_precision_switching<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                            mf_info->d_R,
                            mf_info->n,
                            (void*)p,
                            (void*)q,
                            d_rand_state,
                            lr_decay_arr[e],
                            mf_info->params.k,
                            1,
                            e,
                            update_count,
                            update_vector_size,
                            mf_info->params.lambda,
                            bit_width_set[precision_idx],
                            first_sample_rating_idx,
                            d_sum_updated_val,
                            d_sum_norms
        );

        cudaDeviceSynchronize(); 
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;
        
        if (bit_width_set[precision_idx] == 16){
            cudaMemcpy(sgd_info->half_p, sgd_info->d_half_p, sizeof(half) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
            cudaMemcpy(sgd_info->half_q, sgd_info->d_half_q, sizeof(half) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);

            transform_feature_vector_half2float(sgd_info->half_p, sgd_info->p, mf_info->max_user, mf_info->params.k);
            transform_feature_vector_half2float(sgd_info->half_q, sgd_info->q, mf_info->max_item, mf_info->params.k);
        }else{
            cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
            cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        }
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        rmse_per_epoch_log.push_back(rmse);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

        if (bit_width_set[precision_idx] == (unsigned char)32){
            gradient_diversity_per_epoch_log.push_back(0);
            gradient_diversity_ratio_per_epoch_log.push_back(0);     
            continue;
        };

        error_computation_start_time = std::chrono::system_clock::now();
        //! Get grad from device to compute diversity by epoch
        cudaMemcpy(sum_norms, d_sum_norms, sizeof(float) * mf_info->params.num_workers, cudaMemcpyDeviceToHost);
        cudaMemcpy(sum_updated_val, d_sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyDeviceToHost);
        
        //! Save
        float norm_acc = 0 ;
        
        for (int w = 0; w < mf_info->params.num_workers; w++) norm_acc += sum_norms[w];
        sum_norms_epoch[e] = norm_acc;
        for (int k = 0; k < mf_info->params.k; k++) {
            sum_updated_val_acc[k] = 0;
            sum_updated_val_epoch[e * mf_info->params.k + k] = sum_updated_val[k];
        }
        error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();

        //! Compute metric
        if (e - switched_epoch >= resolution_size){
            error_computation_start_time = std::chrono::system_clock::now();
            float sum_norms_resolution = 0;
            float sum_updated_val_norm = 0;
            for (int wi = e - resolution_size; wi <= e; wi++){
                sum_norms_resolution += sum_norms_epoch[wi];
                for (int k = 0; k < mf_info->params.k; k++) sum_updated_val_acc[k] += sum_updated_val_epoch[wi * mf_info->params.k + k];
            }

            for (int k = 0; k < mf_info->params.k; k++) {
                sum_updated_val_norm += powf(sum_updated_val_acc[k],2);
            }

            float grad_diversity_this_epoch = sum_norms_resolution/sum_updated_val_norm;
            max_grad_diversity = max_grad_diversity < grad_diversity_this_epoch ? grad_diversity_this_epoch : max_grad_diversity;
            float grad_ratio = max_grad_diversity/grad_diversity_this_epoch;
            
            if (grad_ratio > decaying_threshold) {
                precision_idx++;
                switched_epoch = e;
                max_grad_diversity = -1.f;
                transition_params_half2float(mf_info, sgd_info);
                p = (void*)(sgd_info->d_p);
                q = (void*)(sgd_info->d_q);
                cudaFree(sgd_info->d_half_p);
                cudaFree(sgd_info->d_half_q);
            }
            error_computation_time += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - error_computation_start_time).count();
            gradient_diversity_per_epoch_log.push_back(grad_diversity_this_epoch);
            gradient_diversity_ratio_per_epoch_log.push_back(grad_ratio);
        }else{
            gradient_diversity_per_epoch_log.push_back(0);
            gradient_diversity_ratio_per_epoch_log.push_back(0);           
        }
    //     // cudaMemcpy(d_sum_updated_val, sum_updated_val, sizeof(float) * mf_info->params.k, cudaMemcpyHostToDevice);
    }

    for (int i = 0; i < mf_info->params.epoch; i++){
        cout << "Epoch " << setw(3) << left << (i + 1)
             << "  RMSE " << setw(10) << left << fixed << setprecision(6) << rmse_per_epoch_log[i]
             << "  Grad_div " << setw(10) << left << fixed << setprecision(6) << gradient_diversity_per_epoch_log[i] 
             << "  Grad_ratio " << setw(10) << left << fixed  << setprecision(6) << gradient_diversity_ratio_per_epoch_log[i] 
             << "  Threshold " << setw(10) << left << fixed  << setprecision(6) << error_threshold_per_epoch_log[i] 
             << "  Cur_bitwidth " << setw(2) << left << cur_precision_per_epoch_log[i]
             << endl;
    }

    cout << "Error computation time           : " << error_computation_time << endl; 
    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (error_computation_time + sgd_update_execution_time)/1000 << endl;
}

// void adaptive_fixed_point_training_mf(Mf_info* mf_info, SGD* sgd_info){
//     //* Random shuffle and transfer rating triplets to GPU 
//     random_shuffle(mf_info->R, mf_info->R + mf_info->n);

//     cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
//     cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
//     gpuErr(cudaPeekAtLastError());
    
//     //* Convert testset to COO format
//     mf_info->test_COO = test_set_preprocess(mf_info);
//     cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
//     gpuErr(cudaPeekAtLastError());
//     cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
//     //* Initialize random states
//     curandState* d_rand_state;
//     cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
//     init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
//     cudaDeviceSynchronize();
//     gpuErr(cudaPeekAtLastError());

//     //* Learning rate initialization
//     float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
//     for (int i = 0; i < mf_info->params.epoch; i++){
//         lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
//     }
    
//     int update_vector_size = 128;
//     int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
//     int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
//     unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
//     cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
//     cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
//     cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
//     cout << "Start SGD update..." << endl;
    
//     unsigned int div = mf_info->params.thread_block_size/32;
    
//     unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
//     unsigned int group_error_size = error_kernel_work_groups;
//     unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
//     unsigned int seg_size = 32;
//     float* d_e_group;
//     cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

//     double additional_info_init_exec_time = 0;
//     std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();

//     unsigned int block_num = mf_info->params.num_workers/div;
//     double precision_switching_and_error_comp_execution_time = 0;
//     double sgd_update_execution_time = 0;
//     double rmse;    
//     int start_idx = 5;
//     unsigned int num_groups = 10000;

//     vector<float> rmse_per_epoch_log;
    
//     //! For log 
//     int* count_p_bitwidth = new int[mf_info->params.num_workers * 2];
//     int* count_q_bitwidth = new int[mf_info->params.num_workers * 2];
//     int* count_backward_bitwidth = new int[mf_info->params.num_workers * 2];
//     int* d_count_p_bitwidth;
//     int* d_count_q_bitwidth;
//     int* d_count_backward_bitwidth;

//     cudaMalloc(&d_count_p_bitwidth, sizeof(int) * mf_info->params.num_workers * 2);
//     cudaMalloc(&d_count_q_bitwidth, sizeof(int) * mf_info->params.num_workers * 2);
//     cudaMalloc(&d_count_backward_bitwidth, sizeof(int) * mf_info->params.num_workers * 2);

//     vector<int> count_p_bitwidth_8_log;
//     vector<int> count_q_bitwidth_8_log;
//     vector<int> count_backward_8_bitwidth_log;
//     vector<int> count_p_bitwidth_16_log;
//     vector<int> count_q_bitwidth_16_log;
//     vector<int> count_backward_bitwidth_16_log;

//     for (int e = 0; e < mf_info->params.epoch; e++){
//         for (int i = 0; i < mf_info->params.num_workers * 2; i++){
//             count_p_bitwidth[i] = 0;
//             count_q_bitwidth[i] = 0;
//             count_backward_bitwidth[i] = 0;
//         }

//         cudaMemcpy(d_count_p_bitwidth, count_p_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_count_q_bitwidth, count_q_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyHostToDevice);
//         cudaMemcpy(d_count_backward_bitwidth, count_backward_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyHostToDevice);

//         std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
//         sgd_k128_kernel_hogwild_warp32_lrate_mean_diff_adaptive_fixed_point<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
//                             mf_info->d_R,
//                             mf_info->n,
//                             sgd_info->d_p,
//                             sgd_info->d_q,
//                             d_rand_state,
//                             lr_decay_arr[e],
//                             mf_info->params.k,
//                             1,
//                             e,
//                             update_count,
//                             update_vector_size,
//                             mf_info->params.lambda,
//                             d_count_p_bitwidth,
//                             d_count_q_bitwidth,
//                             d_count_backward_bitwidth
//         );

//         cudaDeviceSynchronize(); 
//         double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
//         sgd_update_execution_time += sgd_update_time_per_epoch;

//         cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
//         cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
//         rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
//         rmse_per_epoch_log.push_back(rmse);
//         cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
//         cudaMemcpy(count_p_bitwidth, d_count_p_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyDeviceToHost);
//         cudaMemcpy(count_q_bitwidth, d_count_q_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyDeviceToHost);
//         cudaMemcpy(count_backward_bitwidth, d_count_backward_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyDeviceToHost);

//         int p_8 = 0;
//         int q_8 = 0;
//         int b_8 = 0;
//         int p_16 = 0;
//         int q_16 = 0;
//         int b_16 = 0;

//         for (int i = 0; i < mf_info->params.num_workers; i++){
//             p_8 += count_p_bitwidth[i];
//             q_8 += count_q_bitwidth[i];
//             b_8 += count_backward_bitwidth[i];
//             p_16 += count_p_bitwidth[i + 2048];
//             q_16 += count_q_bitwidth[i + 2048];
//             b_16 += count_backward_bitwidth[i + 2048];
//         }

//         count_p_bitwidth_8_log.push_back(p_8);
//         count_q_bitwidth_8_log.push_back(q_8);
//         count_backward_8_bitwidth_log.push_back(b_8);
//         count_p_bitwidth_16_log.push_back(p_16);
//         count_q_bitwidth_16_log.push_back(q_16);
//         count_backward_bitwidth_16_log.push_back(b_16);
//     }
//     int total = update_count * update_vector_size * 2048;

//     for (int i = 0; i < mf_info->params.epoch; i++){
//         cout << "Epoch     " << setw(3)  << i + 1 << "\t"
//              << "p_8bit  : " << setw(10) << count_p_bitwidth_8_log[i] / (float)total * 100.0f << "\t"
//              << "p_16bit : " << setw(10) << count_p_bitwidth_16_log[i] / (float)total * 100.0f << "\t"
//              << "q_8bit  : " << setw(10) << count_q_bitwidth_8_log[i] / (float)total * 100.0f << "\t"
//              << "q_16bit : " << setw(10) << count_q_bitwidth_16_log[i] / (float)total * 100.0f << "\t"
//              << "b_8bit  : " << setw(10) << count_backward_8_bitwidth_log[i] / (float)total * 100.0f << "\t"
//              << "b_16bit : " << setw(10) << count_backward_bitwidth_16_log[i] / (float)total * 100.0f << "\n";
//     }


//     cout << "Parameters update per epoch                     : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
//     cout << "Total parameters update                         : " << sgd_update_execution_time << endl; 
// }

void adaptive_fixed_point_training_mf(Mf_info* mf_info, SGD* sgd_info){
    //* Random shuffle and transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();

    unsigned int block_num = mf_info->params.num_workers/div;
    double precision_switching_and_error_comp_execution_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    vector<float> rmse_per_epoch_log;

    for (int e = 0; e < mf_info->params.epoch; e++){
        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        
        if (mf_info->params.k == 128){
            sgd_k128_kernel_hogwild_warp32_lrate_mean_diff_adaptive_fixed_point<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                1,
                                e,
                                update_count,
                                update_vector_size,
                                mf_info->params.lambda
            );
        }else if (mf_info->params.k == 64){
            sgd_k64_kernel_hogwild_warp32_lrate_mean_diff_adaptive_fixed_point<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                                mf_info->d_R,
                                mf_info->n,
                                sgd_info->d_p,
                                sgd_info->d_q,
                                d_rand_state,
                                lr_decay_arr[e],
                                mf_info->params.k,
                                1,
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
        rmse_per_epoch_log.push_back(rmse);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;

#ifdef WRITE_FILE
        //! RMSE write code
        string group_error_metric_output_file_path = string("./New_statistics/Adaptive_fixed_point_cvpr/rmse_per_epoch/cvpr_grouping_rmse_per_epoch_") + mf_info->out_file + ".txt";  
        print_rmse(mf_info, group_error_metric_output_file_path, e, rmse);        
#endif
    }

    cout << "Parameters update per epoch      : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update          : " << sgd_update_execution_time << endl; 
    cout << "Total MF time(ms)                : " << (sgd_update_execution_time)/1000 << endl;

#ifdef WRITE_FILE

    map<string, double> statistics_map;

    statistics_map["preprocess"] = 0;
    statistics_map["switching"] = 0;
    statistics_map["update"] = sgd_update_execution_time / 1000;
    statistics_map["total"] = sgd_update_execution_time / 1000;
    statistics_map["rmse"] = rmse;

    string exec_rmse_output_file_path = string("./New_statistics/Adaptive_fixed_point_cvpr/time_rmse/cvpr_time_rmse_") + mf_info->out_file + ".txt";  
    
    print_exec_time_and_rmse(exec_rmse_output_file_path, statistics_map);
#endif

}

void adaptive_fixed_point_training_mf_log(Mf_info* mf_info, SGD* sgd_info){
    //* Random shuffle and transfer rating triplets to GPU 
    random_shuffle(mf_info->R, mf_info->R + mf_info->n);

    cudaMalloc(&(mf_info->d_R), sizeof(Node)*mf_info->n);
    cudaMemcpy(mf_info->d_R, mf_info->R, sizeof(Node) * mf_info->n, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());
    
    //* Convert testset to COO format
    mf_info->test_COO = test_set_preprocess(mf_info);
    cudaMalloc(&mf_info->d_test_COO, sizeof(Node)*mf_info->test_n);
    gpuErr(cudaPeekAtLastError());
    cudaMemcpy(mf_info->d_test_COO, mf_info->test_COO, sizeof(Node) * mf_info->test_n, cudaMemcpyHostToDevice);
    
    //* Initialize random states
    curandState* d_rand_state;
    cudaMalloc(&d_rand_state, sizeof(curandState)*mf_info->params.num_workers);
    init_rand_state<<<((mf_info->params.num_workers+255)/256),256>>>(d_rand_state, mf_info->params.num_workers);
    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());

    //* Learning rate initialization
    float* lr_decay_arr = (float*)malloc(sizeof(float)*mf_info->params.epoch);
    for (int i = 0; i < mf_info->params.epoch; i++){
        lr_decay_arr[i] = mf_info->params.learning_rate/(1.0 + (mf_info->params.decay*pow(i,1.5f)));
    }
    
    int update_vector_size = 128;
    int update_count = ceil(static_cast<double>(mf_info->n) / (mf_info->params.num_workers * update_vector_size));
    int sample_ratings_num = (float)(update_count * update_vector_size) * mf_info->sample_ratio;
    unsigned int first_sample_rating_idx = (update_count * update_vector_size) - sample_ratings_num;
    
    cout << "Processed ratings per worker     : " << update_count * update_vector_size << endl;
    cout << "Sampled ratings per worker       : " << sample_ratings_num << endl;    
    cout << "Intial sampled ratings idx       : " << first_sample_rating_idx << endl;
    cout << "Start SGD update..." << endl;
    
    unsigned int div = mf_info->params.thread_block_size/32;
    
    unsigned int error_kernel_work_groups = ceil(mf_info->test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info->test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;
    float* d_e_group;
    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);

    double additional_info_init_exec_time = 0;
    std::chrono::time_point<std::chrono::system_clock> additional_info_init_start_point = std::chrono::system_clock::now();

    unsigned int block_num = mf_info->params.num_workers/div;
    double precision_switching_and_error_comp_execution_time = 0;
    double sgd_update_execution_time = 0;
    double rmse;    
    int start_idx = 5;
    unsigned int num_groups = 10000;

    vector<float> rmse_per_epoch_log;
    
    //! For log 
    int* count_p_bitwidth = new int[mf_info->params.num_workers * 2];
    int* count_q_bitwidth = new int[mf_info->params.num_workers * 2];
    int* count_backward_bitwidth = new int[mf_info->params.num_workers * 2];
    int* d_count_p_bitwidth;
    int* d_count_q_bitwidth;
    int* d_count_backward_bitwidth;

    cudaMalloc(&d_count_p_bitwidth, sizeof(int) * mf_info->params.num_workers * 2);
    cudaMalloc(&d_count_q_bitwidth, sizeof(int) * mf_info->params.num_workers * 2);
    cudaMalloc(&d_count_backward_bitwidth, sizeof(int) * mf_info->params.num_workers * 2);

    vector<int> count_p_bitwidth_8_log;
    vector<int> count_q_bitwidth_8_log;
    vector<int> count_backward_8_bitwidth_log;
    vector<int> count_p_bitwidth_16_log;
    vector<int> count_q_bitwidth_16_log;
    vector<int> count_backward_bitwidth_16_log;

    for (int e = 0; e < mf_info->params.epoch; e++){
        for (int i = 0; i < mf_info->params.num_workers * 2; i++){
            count_p_bitwidth[i] = 0;
            count_q_bitwidth[i] = 0;
            count_backward_bitwidth[i] = 0;
        }

        cudaMemcpy(d_count_p_bitwidth, count_p_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_count_q_bitwidth, count_q_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_count_backward_bitwidth, count_backward_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyHostToDevice);

        std::chrono::time_point<std::chrono::system_clock> sgd_update_start_time = std::chrono::system_clock::now();
        sgd_k128_kernel_hogwild_warp32_lrate_mean_diff_adaptive_fixed_point_log<<<mf_info->params.num_workers/div,mf_info->params.thread_block_size>>>(
                            mf_info->d_R,
                            mf_info->n,
                            sgd_info->d_p,
                            sgd_info->d_q,
                            d_rand_state,
                            lr_decay_arr[e],
                            mf_info->params.k,
                            1,
                            e,
                            update_count,
                            update_vector_size,
                            mf_info->params.lambda,
                            d_count_p_bitwidth,
                            d_count_q_bitwidth,
                            d_count_backward_bitwidth
        );

        cudaDeviceSynchronize(); 
        double sgd_update_time_per_epoch = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sgd_update_start_time).count();
        sgd_update_execution_time += sgd_update_time_per_epoch;

        cudaMemcpy(sgd_info->p, sgd_info->d_p, sizeof(float) * mf_info->max_user * mf_info->params.k, cudaMemcpyDeviceToHost);
        cudaMemcpy(sgd_info->q, sgd_info->d_q, sizeof(float) * mf_info->max_item * mf_info->params.k, cudaMemcpyDeviceToHost);
        rmse = gpu_test_rmse(mf_info, sgd_info, mf_info->d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
        rmse_per_epoch_log.push_back(rmse);
        cout << e + 1 << " " << lr_decay_arr[e] << " " << rmse << endl;
        
        cudaMemcpy(count_p_bitwidth, d_count_p_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(count_q_bitwidth, d_count_q_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(count_backward_bitwidth, d_count_backward_bitwidth, sizeof(int) * mf_info->params.num_workers * 2, cudaMemcpyDeviceToHost);

        int p_8 = 0;
        int q_8 = 0;
        int b_8 = 0;
        int p_16 = 0;
        int q_16 = 0;
        int b_16 = 0;

        for (int i = 0; i < mf_info->params.num_workers; i++){
            p_8 += count_p_bitwidth[i];
            q_8 += count_q_bitwidth[i];
            b_8 += count_backward_bitwidth[i];
            p_16 += count_p_bitwidth[i + 2048];
            q_16 += count_q_bitwidth[i + 2048];
            b_16 += count_backward_bitwidth[i + 2048];
        }

        count_p_bitwidth_8_log.push_back(p_8);
        count_q_bitwidth_8_log.push_back(q_8);
        count_backward_8_bitwidth_log.push_back(b_8);
        count_p_bitwidth_16_log.push_back(p_16);
        count_q_bitwidth_16_log.push_back(q_16);
        count_backward_bitwidth_16_log.push_back(b_16);
    }
    int total = update_count * update_vector_size * 2048;

    for (int i = 0; i < mf_info->params.epoch; i++){
        cout << "Epoch     " << setw(3)  << i + 1 << "\t"
             << "p_8bit  : " << setw(10) << count_p_bitwidth_8_log[i] / (float)total * 100.0f << "\t"
             << "p_16bit : " << setw(10) << count_p_bitwidth_16_log[i] / (float)total * 100.0f << "\t"
             << "q_8bit  : " << setw(10) << count_q_bitwidth_8_log[i] / (float)total * 100.0f << "\t"
             << "q_16bit : " << setw(10) << count_q_bitwidth_16_log[i] / (float)total * 100.0f << "\t"
             << "b_8bit  : " << setw(10) << count_backward_8_bitwidth_log[i] / (float)total * 100.0f << "\t"
             << "b_16bit : " << setw(10) << count_backward_bitwidth_16_log[i] / (float)total * 100.0f << "\n";
    }


    cout << "Parameters update per epoch                     : " << sgd_update_execution_time / mf_info->params.epoch << endl; 
    cout << "Total parameters update                         : " << sgd_update_execution_time << endl; 
}
