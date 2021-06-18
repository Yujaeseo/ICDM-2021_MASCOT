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
    unsigned int num_groups = 10000;

    cudaMalloc(&mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item);

    // Initialize arrays on device
    init_idx_arr<<<num_groups, 512>>>(mf_info->d_user2idx, mf_info->max_user);
    init_idx_arr<<<num_groups, 512>>>(mf_info->d_item2idx, mf_info->max_item);

    cudaMalloc(&mf_info->d_user2cnt, sizeof(unsigned int) * mf_info->max_user);
    cudaMalloc(&mf_info->d_item2cnt, sizeof(unsigned int) * mf_info->max_item);
    cudaMemset(mf_info->d_user2cnt, 0, sizeof(unsigned int) * mf_info->max_user);
    cudaMemset(mf_info->d_item2cnt, 0, sizeof(unsigned int) * mf_info->max_item);
    
    // Histogram
    user_item_histogram<<<num_groups, 512>>>(mf_info->d_R, mf_info->d_user2cnt, mf_info->d_item2cnt, mf_info->n);

    // Initialize arrays on host
    cudaMallocHost(&mf_info->user2cnt, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2cnt, sizeof(unsigned int) * mf_info->max_item);
    cudaMallocHost(&mf_info->user2idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2idx, sizeof(unsigned int) * mf_info->max_item);
    
    // Initialization to zero 
    cudaDeviceSynchronize();
    
    // Sort
    thrust::device_ptr<unsigned int> thrust_d_user2cnt(mf_info->d_user2cnt);
    thrust::device_ptr<unsigned int> thrust_d_item2cnt(mf_info->d_item2cnt);
    thrust::device_ptr<unsigned int> thrust_d_user2idx(mf_info->d_user2idx);
    thrust::device_ptr<unsigned int> thrust_d_item2idx(mf_info->d_item2idx);
    
    thrust::sort_by_key(thrust_d_user2cnt, thrust_d_user2cnt + mf_info->max_user, thrust_d_user2idx);
    thrust::sort_by_key(thrust_d_item2cnt, thrust_d_item2cnt + mf_info->max_item, thrust_d_item2idx);

    cudaMemcpy(mf_info->user2cnt, mf_info->d_user2cnt, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->item2cnt, mf_info->d_item2cnt, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->user2idx, mf_info->d_user2idx, sizeof(unsigned int) * mf_info->max_user, cudaMemcpyDeviceToHost);
    cudaMemcpy(mf_info->item2idx, mf_info->d_item2idx, sizeof(unsigned int) * mf_info->max_item, cudaMemcpyDeviceToHost);

    cudaFree(mf_info->d_user2idx);
    cudaFree(mf_info->d_item2idx);
    cudaFree(mf_info->d_user2cnt);
    cudaFree(mf_info->d_item2cnt);
}

void split_group_based_equal_size_not_strict_ret_end_idx(Mf_info* mf_info){
    double grouping_exec_time = 0;
    
    mf_info->user_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_user);
    mf_info->item_group_idx = (unsigned int*)malloc(sizeof(unsigned int) * mf_info->max_item);

    vector<unsigned int> user_group_end_idx; 
    vector<unsigned int> item_group_end_idx;
    
    unsigned int threshold_user_num = ceil(mf_info->max_user/(float)mf_info->params.user_group_num);
    unsigned int threshold_item_num = ceil(mf_info->max_item/(float)mf_info->params.item_group_num);

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

    mf_info->params.user_group_num = user_group_end_idx.size();
    mf_info->params.item_group_num = item_group_end_idx.size();
    
    cout << "The number of user groups   : " << mf_info->params.user_group_num << endl;
    cout << "The number of item groups   : " << mf_info->params.item_group_num << endl;

    cudaMallocHost(&mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->params.user_group_num);
    cudaMallocHost(&mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->params.item_group_num);

    for (int i = 0; i < user_group_end_idx.size(); i++) {
        mf_info->user_group_end_idx[i] = user_group_end_idx[i];
        cout << user_group_end_idx[i] << " ";    
    }
    cout << "\n\n";
    for (int i = 0; i < item_group_end_idx.size(); i++) {
        mf_info->item_group_end_idx[i] = item_group_end_idx[i];
        cout << item_group_end_idx[i] << " ";    
    }

    cudaMalloc(&mf_info->d_user_group_end_idx, sizeof(unsigned int) * mf_info->params.user_group_num);
    cudaMalloc(&mf_info->d_item_group_end_idx, sizeof(unsigned int) * mf_info->params.item_group_num);

    cudaMemcpy(mf_info->d_user_group_end_idx, mf_info->user_group_end_idx, sizeof(unsigned int) * mf_info->params.user_group_num, cudaMemcpyHostToDevice);
    cudaMemcpy(mf_info->d_item_group_end_idx, mf_info->item_group_end_idx, sizeof(unsigned int) * mf_info->params.item_group_num, cudaMemcpyHostToDevice);
}

void matrix_reconstruction(Mf_info *mf_info){
    mf_info->sorted_idx2user = new unsigned int[mf_info->max_user];
    mf_info->sorted_idx2item = new unsigned int[mf_info->max_item];
    cudaMallocHost(&mf_info->user2sorted_idx, sizeof(unsigned int) * mf_info->max_user);
    cudaMallocHost(&mf_info->item2sorted_idx, sizeof(unsigned int) * mf_info->max_item);

    mf_info->user_group_size = (unsigned int*)calloc(mf_info->params.user_group_num, sizeof(unsigned int));
    mf_info->item_group_size = (unsigned int*)calloc(mf_info->params.item_group_num, sizeof(unsigned int));

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

#endif

