#ifndef PREPROCESS_UTILS_H
#define PREPROCESS_UTILS_H
#include "common_struct.h"
using namespace std;

__global__ void init_idx_arr(unsigned int* in, unsigned int n){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (; i < n; i += gridDim.x * blockDim.x)
        in[i] = i;
}

__global__ void user_item_histogram(const Node* R, unsigned int* user_bin, unsigned int* item_bin, unsigned int n){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (; i < n; i += gridDim.x * blockDim.x){
        atomicAdd(&(user_bin[R[i].u]),1);
        atomicAdd(&(item_bin[R[i].i]),1);
    }
}

__global__ void cpyfp32_arr2fp32_arr(float* output_arr, float* input_arr, unsigned int n){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (; i < n; i+= gridDim.x * blockDim.x){
        output_arr[i] = input_arr[i];
    }
}

__global__ void initialize_float_array_to_val(float* array, unsigned int n ,float val){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;

    for (; i < n; i += gridDim.x * blockDim.x)
        array[i] = val;
}

__global__ void matrix_reconst_gpu_triplet(Node* R, unsigned int* user2sorted_idx, unsigned int* item2sorted_idx, unsigned int n){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    for (; i < n; i+= gridDim.x * blockDim.x){
        unsigned int origin_user_idx = R[i].u;
        unsigned int origin_item_idx = R[i].i;
        R[i].u = user2sorted_idx[origin_user_idx];
        R[i].i = item2sorted_idx[origin_item_idx];
    }
}


Node* test_set_preprocess(Mf_info *mf_info){
    Node *test_COO = new Node[mf_info->test_n]; 
    unsigned int n = 0;
    for (int user = 0; user < mf_info->max_user; user++){
        for (map<unsigned int, float>::iterator it = mf_info->test_R[user].begin(); it != mf_info->test_R[user].end(); it++){
            double r = it->second;
            unsigned int i = it->first;
            test_COO[n].r = r;
            test_COO[n].u = user;
            test_COO[n].i = i;
            n++;
        }
    }
    return test_COO;
}

void user_item_rating_histogram(Mf_info* mf_info){
    double intialization_exec_time = 0;
    double histogram_exec_time = 0;
    double sorting_exec_time = 0;
    double total_exec_time = 0;

    unsigned int num_groups = 10000;

    cudaMalloc(&mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item);

    //* Initialize arrays on device
    // std::chrono::time_point<std::chrono::system_clock> intialization_start_point = std::chrono::system_clock::now();
    init_idx_arr<<<num_groups, 512>>>(mf_info->d_user2idx, mf_info->max_user);
    init_idx_arr<<<num_groups, 512>>>(mf_info->d_item2idx, mf_info->max_item);
    // intialization_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - intialization_start_point).count();
    // gpuErr(cudaPeekAtLastError());

    cudaMalloc(&mf_info->d_user2cnt, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&mf_info->d_item2cnt, sizeof(unsigned int) * mf_info->max_item);
    cudaMemset(mf_info->d_user2cnt, 0, sizeof(unsigned int) * mf_info->max_user);
    cudaMemset(mf_info->d_item2cnt, 0, sizeof(unsigned int) * mf_info->max_item);
    
    //* Histogram
    user_item_histogram<<<num_groups, 512>>>(mf_info->d_R, mf_info->d_user2cnt, mf_info->d_item2cnt, mf_info->n);

    //* Initialize arrays on host
    cudaMallocHost(&mf_info->user2cnt, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2cnt, sizeof(unsigned int) * mf_info->max_item);
    cudaMallocHost(&mf_info->user2idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2idx, sizeof(unsigned int) * mf_info->max_item);
    
    //* Initialization to zero 
    cudaDeviceSynchronize();

    // std::chrono::time_point<std::chrono::system_clock> histogram_start_point = std::chrono::system_clock::now();
    // cudaDeviceSynchronize();
    // histogram_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - histogram_start_point).count();
    // gpuErr(cudaPeekAtLastError());
    
    //* Sort
    thrust::device_ptr<unsigned int> thrust_d_user2cnt(mf_info->d_user2cnt);
    thrust::device_ptr<unsigned int> thrust_d_item2cnt(mf_info->d_item2cnt);
    thrust::device_ptr<unsigned int> thrust_d_user2idx(mf_info->d_user2idx);
    thrust::device_ptr<unsigned int> thrust_d_item2idx(mf_info->d_item2idx);
    
    // std::chrono::time_point<std::chrono::system_clock> sorting_start_point = std::chrono::system_clock::now();
    thrust::sort_by_key(thrust_d_user2cnt, thrust_d_user2cnt + mf_info->max_user, thrust_d_user2idx);
    thrust::sort_by_key(thrust_d_item2cnt, thrust_d_item2cnt + mf_info->max_item, thrust_d_item2idx);
    // sorting_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sorting_start_point).count();

    cudaMemcpy(mf_info->user2cnt, mf_info->d_user2cnt, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->item2cnt, mf_info->d_item2cnt, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->user2idx, mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->item2idx, mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);
    // gpuErr(cudaPeekAtLastError());

    // total_exec_time = intialization_exec_time + histogram_exec_time + sorting_exec_time;

    // cout << "\n<User & item histogram part exec time (micro sec)>" << endl;
    // cout << "Initialization              : " << intialization_exec_time << endl;
    // cout << "Histogram                   : " << histogram_exec_time << endl;
    // cout << "Sorting                     : " << sorting_exec_time << endl;
    // cout << "Total                       : " << total_exec_time << endl; 
}

void user_item_rating_shuffle(Mf_info* mf_info){
    double intialization_exec_time = 0;
    double histogram_exec_time = 0;
    double sorting_exec_time = 0;
    double total_exec_time = 0;

    unsigned int num_groups = 10000;

    cudaMalloc(&mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item);

    //* Initialize arrays on device
    // std::chrono::time_point<std::chrono::system_clock> intialization_start_point = std::chrono::system_clock::now();
    init_idx_arr<<<num_groups, 512>>>(mf_info->d_user2idx, mf_info->max_user);
    init_idx_arr<<<num_groups, 512>>>(mf_info->d_item2idx, mf_info->max_item);
    // intialization_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - intialization_start_point).count();
    // gpuErr(cudaPeekAtLastError());

    cudaMalloc(&mf_info->d_user2cnt, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&mf_info->d_item2cnt, sizeof(unsigned int) * mf_info->max_item);
    cudaMemset(mf_info->d_user2cnt, 0, sizeof(unsigned int) * mf_info->max_user);
    cudaMemset(mf_info->d_item2cnt, 0, sizeof(unsigned int) * mf_info->max_item);
    
    unsigned int *rand_user_num_array = new unsigned int [mf_info->max_user];
    unsigned int *rand_item_num_array = new unsigned int [mf_info->max_item];

    for (int i = 0; i < mf_info->max_user; i++) rand_user_num_array[i] = i;
    for (int i = 0; i < mf_info->max_item; i++) rand_item_num_array[i] = i;

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(rand_user_num_array, rand_user_num_array + mf_info->max_user, g);
    std::shuffle(rand_item_num_array, rand_item_num_array + mf_info->max_item, g);

    // for (int i = 0; i < 100; i++){
    //     cout << rand_user_num_array[i] << " ";
    // }
    // cout << "\n";
    // for (int i = 0; i < 100; i++){
    //     cout << rand_item_num_array[i] << " ";
    // }
    // cout << "\n";
    //* Histogram
    // user_item_histogram<<<num_groups, 512>>>(mf_info->d_R, mf_info->d_user2cnt, mf_info->d_item2cnt, mf_info->n);
    
    cudaMemcpy(mf_info->d_user2cnt, rand_user_num_array, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item2cnt, rand_item_num_array, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyHostToDevice);

    //* Initialize arrays on host
    cudaMallocHost(&mf_info->user2cnt, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2cnt, sizeof(unsigned int) * mf_info->max_item);
    cudaMallocHost(&mf_info->user2idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2idx, sizeof(unsigned int) * mf_info->max_item);
    
    //* Initialization to zero 
    cudaDeviceSynchronize();

    // std::chrono::time_point<std::chrono::system_clock> histogram_start_point = std::chrono::system_clock::now();
    // cudaDeviceSynchronize();
    // histogram_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - histogram_start_point).count();
    // gpuErr(cudaPeekAtLastError());
    
    //* Sort
    thrust::device_ptr<unsigned int> thrust_d_user2cnt(mf_info->d_user2cnt);
    thrust::device_ptr<unsigned int> thrust_d_item2cnt(mf_info->d_item2cnt);
    thrust::device_ptr<unsigned int> thrust_d_user2idx(mf_info->d_user2idx);
    thrust::device_ptr<unsigned int> thrust_d_item2idx(mf_info->d_item2idx);
    
    // std::chrono::time_point<std::chrono::system_clock> sorting_start_point = std::chrono::system_clock::now();
    thrust::sort_by_key(thrust_d_user2cnt, thrust_d_user2cnt + mf_info->max_user, thrust_d_user2idx);
    thrust::sort_by_key(thrust_d_item2cnt, thrust_d_item2cnt + mf_info->max_item, thrust_d_item2idx);
    // sorting_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - sorting_start_point).count();

    cudaMemcpy(mf_info->user2cnt, mf_info->d_user2cnt, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->item2cnt, mf_info->d_item2cnt, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->user2idx, mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->item2idx, mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 100; i++){
    //     cout << mf_info->user2idx[i] << " ";
    // }
    // cout << "\n";
    // for (int i = 0; i < 100; i++){
    //     cout << mf_info->item2idx[i] << " ";
    // }
    // cout << "\n";


    // gpuErr(cudaPeekAtLastError());

    // total_exec_time = intialization_exec_time + histogram_exec_time + sorting_exec_time;

    // cout << "\n<User & item histogram part exec time (micro sec)>" << endl;
    // cout << "Initialization              : " << intialization_exec_time << endl;
    // cout << "Histogram                   : " << histogram_exec_time << endl;
    // cout << "Sorting                     : " << sorting_exec_time << endl;
    // cout << "Total                       : " << total_exec_time << endl; 
}

void split_group_based_equal_size_not_strict(Mf_info* mf_info){
    // std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    double grouping_exec_time = 0;
    
    vector<unsigned int> user_group_end_idx; 
    vector<unsigned int> item_group_end_idx;
    
    unsigned int threshold_user_num = ceil(mf_info->max_user/(float)mf_info->user_group_num);
    unsigned int threshold_item_num = ceil(mf_info->max_item/(float)mf_info->item_group_num);

    unsigned int max_num_user_ratings = mf_info->user2cnt[mf_info->max_user-1];
    unsigned int min_num_user_ratings = mf_info->user2cnt[0];
    unsigned int max_num_item_ratings = mf_info->item2cnt[mf_info->max_item-1];
    unsigned int min_num_item_ratings = mf_info->item2cnt[0];
    
    unsigned int range_user_ratings = max_num_user_ratings - min_num_user_ratings;
    unsigned int range_item_ratings = max_num_item_ratings - min_num_item_ratings;

    unsigned int *rating_user_idx = new unsigned int[max_num_user_ratings + 1];
    unsigned int *rating_item_idx = new unsigned int[max_num_item_ratings + 1];

    for (int i = 1; i <= max_num_user_ratings; i++) rating_user_idx[i] = -1;
    for (int i = 1; i <= max_num_item_ratings; i++) rating_item_idx[i] = -1;

    for (int i = 0; i < mf_info->max_user; i++){
        unsigned int cur_rating = mf_info->user2cnt[i];
        rating_user_idx[cur_rating] = i;
    }

    for (int i = 0; i < mf_info->max_item; i++){
        unsigned int cur_rating = mf_info->item2cnt[i];
        rating_item_idx[cur_rating] = i;
    }

    unsigned int acc_ratings = 0;
    unsigned int prev_rating_idx = 0;
    for (int i = 1; i <= max_num_user_ratings; i++){
        if (rating_user_idx[i]!=-1){
            acc_ratings += (rating_user_idx[i]-prev_rating_idx+1);
            if (acc_ratings >= threshold_user_num){
                user_group_end_idx.push_back(rating_user_idx[i]);
                acc_ratings = 0;
            }
            else if (i == max_num_user_ratings){
                user_group_end_idx.push_back(rating_user_idx[i]);
            }
            prev_rating_idx = rating_user_idx[i]+1;
        }
    }
    // cout << "pop !" << endl;
    acc_ratings = 0;
    prev_rating_idx = 0;
    for (int i = 1; i <= max_num_item_ratings; i++){
        if (rating_item_idx[i]!=-1){
            acc_ratings += (rating_item_idx[i]-prev_rating_idx+1);
            if (acc_ratings >= threshold_item_num){
                item_group_end_idx.push_back(rating_item_idx[i]);
                acc_ratings = 0;
            }else if (i == max_num_item_ratings){
                item_group_end_idx.push_back(rating_item_idx[i]);
            }
            prev_rating_idx = rating_item_idx[i]+1;
        }
    }

    unsigned int start_idx;
    unsigned int end_idx = 0;

    for (int g = 0; g < user_group_end_idx.size(); g++){
        start_idx = end_idx;
        end_idx = user_group_end_idx[g] + 1;

        for (int u = start_idx; u < end_idx; u++){
            mf_info->user_group_idx[mf_info->user2idx[u]] = g;
        }
    }
    end_idx = 0;
    for (int g = 0; g < item_group_end_idx.size(); g++){
        start_idx = end_idx;
        end_idx = item_group_end_idx[g] + 1;

        for (int i = start_idx; i < end_idx; i++){
            mf_info->item_group_idx[mf_info->item2idx[i]] = g;
        }
    }

    mf_info->user_group_num = user_group_end_idx.size();
    mf_info->item_group_num = item_group_end_idx.size();
    // for (int i = 0; i < user_group_end_idx.size(); i++){
    //     cout << user_group_end_idx[i] << " ";
    // }
    // cout << "\n";
    // for (int i = 0; i < item_group_end_idx.size(); i++){
    //     cout << item_group_end_idx[i] << " ";
    // }
    
    // grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();
    // cout << "\n<User & item grouping exec time (micro sec)>" << endl;
    // cout << "Grouping                    : " << grouping_exec_time << endl; 
}

void split_group_based_equal_size_not_strict_ret_end_idx(Mf_info* mf_info){
    // std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    double grouping_exec_time = 0;
    
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);

    vector<unsigned int> user_group_end_idx; 
    vector<unsigned int> item_group_end_idx;
    
    unsigned int threshold_user_num = ceil(mf_info->max_user/(float)mf_info->user_group_num);
    unsigned int threshold_item_num = ceil(mf_info->max_item/(float)mf_info->item_group_num);

    unsigned int max_num_user_ratings = mf_info->user2cnt[mf_info->max_user-1];
    unsigned int min_num_user_ratings = mf_info->user2cnt[0];
    unsigned int max_num_item_ratings = mf_info->item2cnt[mf_info->max_item-1];
    unsigned int min_num_item_ratings = mf_info->item2cnt[0];
    
    unsigned int range_user_ratings = max_num_user_ratings - min_num_user_ratings;
    unsigned int range_item_ratings = max_num_item_ratings - min_num_item_ratings;

    unsigned int *rating_user_idx = new unsigned int[max_num_user_ratings + 1];
    unsigned int *rating_item_idx = new unsigned int[max_num_item_ratings + 1];

    for (int i = 1; i <= max_num_user_ratings; i++) rating_user_idx[i] = -1;
    for (int i = 1; i <= max_num_item_ratings; i++) rating_item_idx[i] = -1;

    for (int i = 0; i < mf_info->max_user; i++){
        unsigned int cur_rating = mf_info->user2cnt[i];
        rating_user_idx[cur_rating] = i;
    }

    for (int i = 0; i < mf_info->max_item; i++){
        unsigned int cur_rating = mf_info->item2cnt[i];
        rating_item_idx[cur_rating] = i;
    }

    unsigned int acc_ratings = 0;
    unsigned int prev_rating_idx = 0;
    for (int i = 1; i <= max_num_user_ratings; i++){
        if (rating_user_idx[i]!=-1){
            acc_ratings += (rating_user_idx[i]-prev_rating_idx+1);
            if (acc_ratings >= threshold_user_num){
                user_group_end_idx.push_back(rating_user_idx[i]);
                acc_ratings = 0;
            }
            else if (i == max_num_user_ratings){
                user_group_end_idx.push_back(rating_user_idx[i]);
            }
            prev_rating_idx = rating_user_idx[i]+1;
        }
    }
    // cout << "pop !" << endl;
    acc_ratings = 0;
    prev_rating_idx = 0;
    for (int i = 1; i <= max_num_item_ratings; i++){
        if (rating_item_idx[i]!=-1){
            acc_ratings += (rating_item_idx[i]-prev_rating_idx+1);
            if (acc_ratings >= threshold_item_num){
                item_group_end_idx.push_back(rating_item_idx[i]);
                acc_ratings = 0;
            }else if (i == max_num_item_ratings){
                item_group_end_idx.push_back(rating_item_idx[i]);
            }
            prev_rating_idx = rating_item_idx[i]+1;
        }
    }

    unsigned int start_idx;
    unsigned int end_idx = 0;

    for (int g = 0; g < user_group_end_idx.size(); g++){
        start_idx = end_idx;
        end_idx = user_group_end_idx[g] + 1;

        for (int u = start_idx; u < end_idx; u++){
            mf_info->user_group_idx[mf_info->user2idx[u]] = g;
        }
    }
    end_idx = 0;
    for (int g = 0; g < item_group_end_idx.size(); g++){
        start_idx = end_idx;
        end_idx = item_group_end_idx[g] + 1;

        for (int i = start_idx; i < end_idx; i++){
            mf_info->item_group_idx[mf_info->item2idx[i]] = g;
        }
    }

    mf_info->user_group_num = user_group_end_idx.size();
    mf_info->item_group_num = item_group_end_idx.size();

    cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    for (int i = 0; i < user_group_end_idx.size(); i++) {
        mf_info->user_group_end_idx[i] = user_group_end_idx[i];
        cout << user_group_end_idx[i] << " ";    
    }
    cout << "\n\n";
    for (int i = 0; i < item_group_end_idx.size(); i++) {
        mf_info->item_group_end_idx[i] = item_group_end_idx[i];
        cout << item_group_end_idx[i] << " ";    
    }

    cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);
}


void split_group_based_equal_size(Mf_info* mf_info){
    double grouping_exec_time = 0;

    // std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    unsigned int cur_group = 0;
    unsigned int user_group_size = ceil(mf_info->max_user/(double)mf_info->user_group_num);
    for (int i = 0; i < mf_info->max_user; i++){
        unsigned int user_end_idx = cur_group == (mf_info->user_group_num -1) ? mf_info->max_user - 1 : ((cur_group+1) * user_group_size)-1;
        if (i > user_end_idx){
            cur_group++;
            i = i - 1;
            continue;
        }
        mf_info->user_group_idx[mf_info->user2idx[i]] = cur_group;
    }

    cur_group = 0;
    unsigned int item_group_size = ceil(mf_info->max_item/(double)mf_info->item_group_num);
    for (int i = 0; i < mf_info->max_item; i++){
        unsigned int item_end_idx = cur_group == (mf_info->item_group_num -1) ? mf_info->max_item - 1 : ((cur_group+1) * item_group_size)-1;
        if (i > item_end_idx){
            cur_group++;
            i = i - 1;
            continue;
        }
        mf_info->item_group_idx[mf_info->item2idx[i]] = cur_group;
    }

    // grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();
    // cout << "\n<User & item grouping exec time (micro sec)>" << endl;
    // cout << "Grouping                    : " << grouping_exec_time << endl; 
}

void split_group_based_equal_size_ret_end_idx(Mf_info* mf_info){
    double grouping_exec_time = 0;

    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();
    unsigned int cur_group = 0;
    unsigned int user_group_size = ceil(mf_info->max_user/(double)mf_info->user_group_num);
    for (int i = 0; i < mf_info->max_user; i++){
        unsigned int user_end_idx = cur_group == (mf_info->user_group_num -1) ? mf_info->max_user - 1 : ((cur_group+1) * user_group_size)-1;
        if (i > user_end_idx){
            cur_group++;
            i = i - 1;
            continue;
        }
        mf_info->user_group_end_idx[cur_group] = i;
        mf_info->user_group_idx[mf_info->user2idx[i]] = cur_group;
    }

    cur_group = 0;
    unsigned int item_group_size = ceil(mf_info->max_item/(double)mf_info->item_group_num);
    for (int i = 0; i < mf_info->max_item; i++){
        unsigned int item_end_idx = cur_group == (mf_info->item_group_num -1) ? mf_info->max_item - 1 : ((cur_group+1) * item_group_size)-1;
        if (i > item_end_idx){
            cur_group++;
            i = i - 1;
            continue;
        }
        mf_info->item_group_end_idx[cur_group] = i;
        mf_info->item_group_idx[mf_info->item2idx[i]] = cur_group;
    }
    
    cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();
    cout << "\n<User & item grouping exec time (micro sec)>" << endl;
    cout << "Grouping                    : " << grouping_exec_time << endl; 
}

void split_user_item_ratio(Mf_info* mf_info){
    double grouping_exec_time = 0;

    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);
    cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    std::chrono::time_point<std::chrono::system_clock> grouping_start_point = std::chrono::system_clock::now();    
    mf_info->fp16_user_num = ceil(((double)mf_info->max_user) * mf_info->fp16_user_ratio);
    mf_info->fp16_item_num = ceil(((double)mf_info->max_item) * mf_info->fp16_item_ratio);

    cout << "User ratio    : " << mf_info->fp16_user_ratio << endl;
    cout << "Item ratio    : " << mf_info->fp16_item_ratio << endl;
    cout << "Fp16 user num : " << mf_info->fp16_user_num << endl;
    cout << "Fp16 item num : " << mf_info->fp16_item_num << endl;

    for (int i = 0; i < mf_info->max_user; i++) {
        unsigned int group = 0; // ! half
        if (mf_info->fp16_user_num <= i) group = 1; // !single
        mf_info->user_group_idx[mf_info->user2idx[i]] = group;
    }

    for (int i = 0; i < mf_info->max_item; i++) {
        unsigned int group = 0; // ! half
        if (mf_info->fp16_item_num <= i) group = 1; // ! single
        mf_info->item_group_idx[mf_info->item2idx[i]] = group;
    }
    
    mf_info->user_group_end_idx[0] = mf_info->fp16_user_num - 1;
    mf_info->item_group_end_idx[0] = mf_info->fp16_item_num - 1;
    mf_info->user_group_end_idx[1] = mf_info->max_user-1;
    mf_info->item_group_end_idx[1] = mf_info->max_item-1;

    cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num);
    cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num);

    cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->item_group_num, cudaMemcpyHostToDevice);

    grouping_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - grouping_start_point).count();
    cout << "\n<User & item grouping exec time (micro sec)>" << endl;
    cout << "Grouping                    : " << grouping_exec_time << endl; 
}

void split_group_based_rating_num_exp (Mf_info* mf_info){
    unsigned int* user_group_rating_range = new unsigned int [mf_info->user_group_num];
    unsigned int* item_group_rating_range = new unsigned int [mf_info->item_group_num];

    unsigned int max_num_user_ratings = mf_info->user2cnt[mf_info->max_user-1];
    unsigned int min_num_user_ratings = mf_info->user2cnt[0];
    unsigned int max_num_item_ratings = mf_info->item2cnt[mf_info->max_item-1];
    unsigned int min_num_item_ratings = mf_info->item2cnt[0];
    
    unsigned int range_user_ratings = max_num_user_ratings - min_num_user_ratings;
    unsigned int range_item_ratings = max_num_item_ratings - min_num_item_ratings;
    
    double diff = 0;
    float prev = min_num_user_ratings;
    
    //! Get each user group rating range    
    for (int i = 0; i < mf_info->user_group_num; i++){
        if (i == 0) diff = pow(range_user_ratings, 1.0/mf_info->user_group_num);
        else diff = pow(range_user_ratings, (double)(i+1)/mf_info->user_group_num) - pow(range_user_ratings, (double)(i)/mf_info->user_group_num);
        float tmp = prev + diff;
        // user_group_rating_range[i] = prev + (unsigned int)ceil(diff);
        user_group_rating_range[i] = (unsigned int)ceil(tmp);
        prev = tmp;
    }

    //! Get each item group rating range    
    prev = min_num_item_ratings;
    for (int i = 0; i < mf_info->item_group_num; i++){
        if (i == 0) diff = pow(range_item_ratings, 1.0/mf_info->item_group_num);
        else diff = pow(range_item_ratings, (double)(i+1)/mf_info->item_group_num) - pow(range_item_ratings, (double)(i)/mf_info->item_group_num);
        float tmp = prev + diff;
        // item_group_rating_range[i] = prev + (unsigned int)ceil(diff);
        item_group_rating_range[i] = (unsigned int)ceil(tmp);
        prev = tmp;
        // prev = item_group_rating_range[i];
    }

    //! Get each user group idx    
    unsigned int cur_group = 0;
    for (int i = 0; i < mf_info->max_user; i++){
        if (mf_info->user2cnt[i] > user_group_rating_range[cur_group]){
            cur_group++;
            for (int j = cur_group; j < mf_info->user_group_num; j++){
                if (mf_info->user2cnt[i] > user_group_rating_range[j]){
                    // user_group_end_idx[j] = user_group_end_idx[j-1];
                }else{
                    cur_group = j;
                    break;
                }
            }
        }
        // user_group_end_idx[cur_group] = i;
        mf_info->user_group_idx[mf_info->user2idx[i]] = cur_group;
    }

    //! Get each item group idx
    cur_group = 0;
    for (int i = 0; i < mf_info->max_item; i++){
        if (mf_info->item2cnt[i] > item_group_rating_range[cur_group]){
            cur_group++;
            for (int j = cur_group; j < mf_info->item_group_num; j++){
                if (mf_info->item2cnt[i] > item_group_rating_range[j]){
                    // item_group_end_idx[j] = item_group_end_idx[j-1];
                }else{
                    cur_group = j;
                    break;
                }
            }
        }
        // item_group_end_idx[cur_group] = i;
        mf_info->item_group_idx[mf_info->item2idx[i]] = cur_group;
    }
}

void split_group_based_rating_num (Mf_info* mf_info){
    unsigned int* user_group_rating_range = new unsigned int [mf_info->user_group_num];
    unsigned int* item_group_rating_range = new unsigned int [mf_info->item_group_num];

    unsigned int max_num_user_ratings = mf_info->user2cnt[mf_info->max_user-1];
    unsigned int min_num_user_ratings = mf_info->user2cnt[0];
    unsigned int max_num_item_ratings = mf_info->item2cnt[mf_info->max_item-1];
    unsigned int min_num_item_ratings = mf_info->item2cnt[0];

    unsigned int range_user_ratings = max_num_user_ratings - min_num_user_ratings;
    unsigned int range_item_ratings = max_num_item_ratings - min_num_item_ratings;

    cout << "User rating range : " << range_user_ratings << endl;
    cout << "Item rating range : " << range_item_ratings << endl;

    double diff = 0;
    unsigned int prev = min_num_user_ratings;
    
    //! Get each user group rating range    
    for (int i = 0; i < mf_info->user_group_num; i++){
        diff = ((double)range_user_ratings/(double)mf_info->user_group_num);
        user_group_rating_range[i] = prev + (unsigned int)ceil(diff);
        if (i == mf_info->user_group_num - 1) user_group_rating_range[i] = max_num_user_ratings;
        prev = user_group_rating_range[i];
    }

    //! Get each item group rating range    
    prev = min_num_item_ratings;
    for (int i = 0; i < mf_info->item_group_num; i++){
        diff = ((double)range_item_ratings)/((double)mf_info->item_group_num);
        item_group_rating_range[i] = prev + (unsigned int)ceil(diff);
        if (i == mf_info->item_group_num - 1) item_group_rating_range[i] = max_num_item_ratings;
        prev = item_group_rating_range[i];
    }
    
    //! Get each user group idx    
    unsigned int cur_group = 0;
    for (int i = 0; i < mf_info->max_user; i++){
        if (mf_info->user2cnt[i] > user_group_rating_range[cur_group]){
            cur_group++;
            for (int j = cur_group; j < mf_info->user_group_num; j++){
                if (mf_info->user2cnt[i] > user_group_rating_range[j]){
                    // user_group_end_idx[j] = user_group_end_idx[j-1];
                }else{
                    cur_group = j;
                    break;
                }
            }
        }
        // user_group_end_idx[cur_group] = i;
        mf_info->user_group_idx[mf_info->user2idx[i]] = cur_group;
    }

    //! Get each item group idx
    cur_group = 0;
    for (int i = 0; i < mf_info->max_item; i++){
        if (mf_info->item2cnt[i] > item_group_rating_range[cur_group]){
            cur_group++;
            for (int j = cur_group; j < mf_info->item_group_num; j++){
                if (mf_info->item2cnt[i] > item_group_rating_range[j]){
                    // item_group_end_idx[j] = item_group_end_idx[j-1];
                }else{
                    cur_group = j;
                    break;
                }
            }
        }
        // item_group_end_idx[cur_group] = i;
        mf_info->item_group_idx[mf_info->item2idx[i]] = cur_group;
    }
}

// void generate_map_idx_info(Mf_info* mf_info){
//     double generate_map_exec_time = 0;
//     mf_info->user_group_size = (unsigned int*)calloc(mf_info->user_group_num, sizeof(unsigned int));
//     mf_info->item_group_size = (unsigned int*)calloc(mf_info->item_group_num, sizeof(unsigned int));
//     cudaMallocHost(&mf_info->user_index_info, sizeof(Index_info_node)*mf_info->max_user);
//     cudaMallocHost(&mf_info->item_index_info, sizeof(Index_info_node)*mf_info->max_item);

//     std::chrono::time_point<std::chrono::system_clock> generate_map_start_point = std::chrono::system_clock::now();

//     for (int i = 0; i < mf_info->max_user; i++){
//         unsigned int user_group = mf_info->user_group_idx[mf_info->user2idx[i]];
//         mf_info->user_index_info[mf_info->user2idx[i]].g = user_group;
//         mf_info->user_index_info[mf_info->user2idx[i]].v = mf_info->user_group_size[user_group];   
//         mf_info->user_group_size[user_group] += 1;
//     }
    
//     for (int i = 0; i < mf_info->max_item; i++){
//         unsigned int item_group = mf_info->item_group_idx[mf_info->item2idx[i]];
//         mf_info->item_index_info[mf_info->item2idx[i]].g = item_group;
//         mf_info->item_index_info[mf_info->item2idx[i]].v = mf_info->item_group_size[item_group];
//         mf_info->item_group_size[item_group] += 1;
//     }

//     generate_map_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - generate_map_start_point).count();
    
//     cout << "\n<User & item generating map info exec time (micro sec)>" << endl;
//     cout << "Generate map index info     : " << generate_map_exec_time << endl; 

//     cudaMalloc(&mf_info->d_user_index_info, sizeof(Index_info_node)*mf_info->max_user);
//     cudaMalloc(&mf_info->d_item_index_info, sizeof(Index_info_node)*mf_info->max_item);

//     cudaMemcpy(mf_info->d_user_index_info, mf_info->user_index_info, sizeof(Index_info_node) * mf_info->max_user, cudaMemcpyHostToDevice);
//     cudaMemcpy(mf_info->d_item_index_info, mf_info->item_index_info, sizeof(Index_info_node) * mf_info->max_item, cudaMemcpyHostToDevice);

//     // free(mf_info->user_group_idx);
//     // free(mf_info->item_group_idx);
// }

void generate_map_idx_info(Mf_info* mf_info){
    // double generate_map_exec_time = 0;
    mf_info->user_group_size = (unsigned int*)calloc(mf_info->user_group_num, sizeof(unsigned int));
    mf_info->item_group_size = (unsigned int*)calloc(mf_info->item_group_num, sizeof(unsigned int));
    cudaMallocHost(&mf_info->user_index_info, sizeof(Index_info_node)*mf_info->max_user);
    cudaMallocHost(&mf_info->item_index_info, sizeof(Index_info_node)*mf_info->max_item);

    // std::chrono::time_point<std::chrono::system_clock> generate_map_start_point = std::chrono::system_clock::now();

    for (int i = 0; i < mf_info->max_user; i++){
        unsigned int user_group = mf_info->user_group_idx[mf_info->user2idx[i]];
        mf_info->user_index_info[mf_info->user2idx[i]].g = user_group;
        mf_info->user_index_info[mf_info->user2idx[i]].v = mf_info->user_group_size[user_group];   
        mf_info->user_group_size[user_group] += 1;
        // ! for test
        // mf_info->user_index_info[mf_info->user2idx[i]].v = mf_info->user2idx[i];   
    }
    
    for (int i = 0; i < mf_info->max_item; i++){
        unsigned int item_group = mf_info->item_group_idx[mf_info->item2idx[i]];
        mf_info->item_index_info[mf_info->item2idx[i]].g = item_group;
        mf_info->item_index_info[mf_info->item2idx[i]].v = mf_info->item_group_size[item_group];
        mf_info->item_group_size[item_group] += 1;
        // ! for test
        // mf_info->item_index_info[mf_info->item2idx[i]].v = mf_info->item2idx[i];
    }

    // generate_map_exec_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - generate_map_start_point).count();
    
    // cout << "\n<User & item generating map info exec time (micro sec)>" << endl;
    // cout << "Generate map index info     : " << generate_map_exec_time << endl; 

    cudaMalloc(&mf_info->d_user_index_info, sizeof(Index_info_node)*mf_info->max_user);
    cudaMalloc(&mf_info->d_item_index_info, sizeof(Index_info_node)*mf_info->max_item);

    cudaMemcpy(mf_info->d_user_index_info, mf_info->user_index_info, sizeof(Index_info_node) * mf_info->max_user, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item_index_info, mf_info->item_index_info, sizeof(Index_info_node) * mf_info->max_item, cudaMemcpyHostToDevice);

    // free(mf_info->user_group_idx);
    // free(mf_info->item_group_idx);
}

void matrix_reconstruction(Mf_info *mf_info){
    mf_info->sorted_idx2user = new unsigned int[mf_info->max_user];
    mf_info->sorted_idx2item = new unsigned int[mf_info->max_item];
    cudaMallocHost(&mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);

    mf_info->user_group_size = (unsigned int*)calloc(mf_info->user_group_num, sizeof(unsigned int));
    mf_info->item_group_size = (unsigned int*)calloc(mf_info->item_group_num, sizeof(unsigned int));

    for(int i = 0; i < mf_info->max_user; i++){
        unsigned int converted_user_idx = i;
        unsigned int original_user_idx = mf_info->user2idx[i];
        unsigned int user_group = mf_info->user_group_idx[mf_info->user2idx[i]];
        mf_info->user2sorted_idx[original_user_idx] = converted_user_idx;
        mf_info->sorted_idx2user[converted_user_idx] = original_user_idx;
        mf_info->user_group_size[user_group]++;
    }

    for(int i = 0; i < mf_info->max_item; i++){
        unsigned int converted_item_idx = i;
        unsigned int original_item_idx = mf_info->item2idx[i];
        unsigned int item_group = mf_info->item_group_idx[mf_info->item2idx[i]];
        mf_info->item2sorted_idx[original_item_idx] = converted_item_idx;
        mf_info->sorted_idx2item[converted_item_idx] = original_item_idx;
        mf_info->item_group_size[item_group]++;
    }

    cudaMalloc(&mf_info->d_user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&mf_info->d_item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);
    
    cudaMemcpy(mf_info->d_user2sorted_idx, mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item2sorted_idx, mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyHostToDevice);
    gpuErr(cudaPeekAtLastError());

    int num_groups = 10000;
    matrix_reconst_gpu_triplet<<<num_groups, 512>>>(mf_info->d_R, mf_info->d_user2sorted_idx, mf_info->d_item2sorted_idx, mf_info->n);
    matrix_reconst_gpu_triplet<<<num_groups, 512>>>(mf_info->d_test_COO, mf_info->d_user2sorted_idx, mf_info->d_item2sorted_idx, mf_info->test_n);

    cudaDeviceSynchronize();
    gpuErr(cudaPeekAtLastError());
}

// void check_group_size(Mf_info* mf_info, unsigned int* user_group_end_idx, unsigned int* item_group_end_idx){

//     mf_info->user_group_size = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->user_group_num);
//     mf_info->item_group_size = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->item_group_num);

//     for (int i = 0; i < mf_info->user_group_num; i++) mf_info->user_group_size[i] = (i == 0 ? user_group_end_idx[i] + 1 : user_group_end_idx[i] - user_group_end_idx[i-1]);
//     for (int i = 0; i < mf_info->item_group_num; i++) mf_info->item_group_size[i] = (i == 0 ? item_group_end_idx[i] + 1 : item_group_end_idx[i] - item_group_end_idx[i-1]);

//     for (int i = 0; i < mf_info->user_group_num; i++) cout << mf_info->user_group_size[i] << " ";
//     for (int i = 0; i < mf_info->item_group_num; i++) cout << mf_info->item_group_size[i] << " ";
// }
#endif

