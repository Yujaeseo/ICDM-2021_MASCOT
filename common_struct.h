#ifndef COMMON_STRUCT_H
#include <map>
#include <vector>
#include <cuda_fp16.h>
using namespace std;
#define COMMON_STRUCT_H

struct Node{
    float r;
    unsigned int u;
    unsigned int i;
};

struct Index_info_node{
    unsigned int g;
    unsigned int v;
};

struct Parameter{
    Parameter(){}
    float lambda;
    float learning_rate;
    float decay;
    float mean_val;
    float std_val;
    float sample_ratio;
    unsigned int interval;
    float error_threshold;
    unsigned int user_group_num;
    unsigned int item_group_num;
    unsigned int k;
    unsigned int num_workers;
    unsigned int epoch;
    unsigned int thread_block_size;
};

struct Mf_info{
    Mf_info():max_user(0), max_item(0), n(0), test_n(0) {}
    Node* R;
    Node* d_R;
    Node* test_COO;
    Node* d_test_COO;
    Index_info_node* user_index_info;
    Index_info_node* item_index_info;
    Index_info_node* d_user_index_info;
    Index_info_node* d_item_index_info;

    unsigned int* d_user2cnt;
    unsigned int* d_item2cnt;
    unsigned int* d_user2idx;
    unsigned int* d_item2idx; 

    unsigned int* user2cnt;
    unsigned int* item2cnt;
    unsigned int* user2idx;
    unsigned int* item2idx;

    unsigned int* user_group_idx;
    unsigned int* item_group_idx; 
    unsigned int* user_group_size;
    unsigned int* item_group_size;
    
    unsigned char* user_group_prec_info;
    unsigned char* item_group_prec_info;
    unsigned char* d_user_group_prec_info;
    unsigned char* d_item_group_prec_info;

    unsigned int* user_group_end_idx;
    unsigned int* item_group_end_idx;
    unsigned int* d_user_group_end_idx;
    unsigned int* d_item_group_end_idx;

    unsigned int* user2sorted_idx; 
    unsigned int* item2sorted_idx;
    unsigned int* sorted_idx2user;
    unsigned int* sorted_idx2item;

    unsigned int* d_user2sorted_idx; 
    unsigned int* d_item2sorted_idx;
    
    float* user_group_error;
    float* item_group_error;
    float* d_user_group_error;
    float* d_item_group_error;

    bool is_yahoo;
    unsigned int version;
    map<unsigned int, unsigned int> user_map, item_map, user_map2orig, item_map2orig;
    vector<map<unsigned int, float>> test_R;
    unsigned int max_user, max_item, n, test_n;
    Parameter params;
};

struct SGD{
    SGD(){}
    float* p;
    float* q;

    float* d_p;
    float* d_q;

    short* half_p;
    short* half_q;

    __half* d_half_p;
    __half* d_half_q;

    void** user_group_ptr;
    void** item_group_ptr;
    void** user_group_d_ptr;
    void** item_group_d_ptr;
    void** d_user_group_ptr;
    void** d_item_group_ptr;
};

#endif