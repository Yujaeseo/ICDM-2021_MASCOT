#include <cuda_fp16.h>

#define NUM_USER_GROUPS 10
#define NUM_ITEM_GROUPS 10
// #define HALF_COMP 
#define SINGLE_COMP 

// #define GRAD_CRITERIA __float2half_rn(0.000000009f) 
#define GRAD_CRITERIA __float2half_rn(0.f)

__global__ void sgd_k128_kernel_hogwild_warp32_lrate(
                            const Node *R,
                            unsigned int nnz,
                            float *p,
                            float *q,
                            curandState *state,
                            float lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda
                            )
{    
    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        // All threads read x from laneid 0
        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            //read the p & q into register file.
            int base_p = u*k;
            int base_q = v*k;

            float tmp_p1 = p[base_p + lane_id];
            float tmp_q1 = q[base_q + lane_id];

            float tmp_p2 = p[base_p + lane_id + 32];
            float tmp_q2 = q[base_q + lane_id + 32];

            float tmp_p3 = p[base_p + lane_id + 64];
            float tmp_q3 = q[base_q + lane_id + 64];

            float tmp_p4 = p[base_p + lane_id + 96];
            float tmp_q4 = q[base_q + lane_id + 96];

        
            float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            float ruv = r - tmp_product;

            p[base_p + lane_id +  0] = tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1);
            q[base_q + lane_id +  0] = tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1);

            p[base_p + lane_id + 32] = tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2);
            q[base_q + lane_id + 32] = tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2);

            p[base_p + lane_id + 64] = tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3);
            q[base_q + lane_id + 64] = tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3);

            p[base_p + lane_id + 96] = tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4);
            q[base_q + lane_id + 96] = tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4);
        }    
    }
}

// __global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing(
//                             const Node *R,
//                             unsigned int nnz,
//                             __half** p,
//                             __half** q,
//                             curandState *state,
//                             __half lrate,
//                             int k,
//                             int num_iters,
//                             int current_iter,
//                             int update_count_this_block,
//                             int update_vector_size,
//                             __half lambda,
//                             Index_info_node* user_index_info,
//                             Index_info_node* item_index_info
//                             )
// {    
//     __shared__ __half* p_s[NUM_GROUPS];
//     __shared__ __half* q_s[NUM_GROUPS];

//     if (threadIdx.x < NUM_GROUPS)
//         p_s[threadIdx.x] = p[threadIdx.x];
//     if (threadIdx.x < NUM_GROUPS)
//         q_s[threadIdx.x] = q[threadIdx.x];
//     __syncthreads();
    
//     for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
//     {
//         int lane_id = threadIdx.x%32;
//         int local_wid = threadIdx.x/32;
//         int local_w_num = blockDim.x/32;
//         int wid = local_w_num*blockIdx.x + local_wid;  
        
//         unsigned int start_id = 0;
//         if(lane_id == 0)
//         {
//             unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
//             start_id = origin%nnz;
//         }

//         start_id = __shfl_sync(0xffffffff,start_id, 0);
        
//         for(int i = 0;i < update_vector_size;i++)
//         {
//             int offset = (start_id + i)%nnz;
            
//             __half r = __float2half_rn(__ldg(&R[offset].r));
//             int orig_u = __ldg(&R[offset].u);
//             int orig_v = __ldg(&R[offset].i);
            
//             int user_group = user_index_info[orig_u].g;
//             int u = user_index_info[orig_u].v;
//             int item_group = item_index_info[orig_v].g;
//             int v = item_index_info[orig_v].v;
//             // printf("%d ", threadIdx.x);
//             int base_p = u*k;
//             int base_q = v*k;

//             __half tmp_p1 = (p_s[user_group])[base_p + lane_id];
//             __half tmp_q1 = (q_s[item_group])[base_q + lane_id];

//             __half tmp_p2 = (p_s[user_group])[base_p + lane_id + 32];
//             __half tmp_q2 = (q_s[item_group])[base_q + lane_id + 32];

//             __half tmp_p3 = (p_s[user_group])[base_p + lane_id + 64];
//             __half tmp_q3 = (q_s[item_group])[base_q + lane_id + 64];

//             __half tmp_p4 = (p_s[user_group])[base_p + lane_id + 96];
//             __half tmp_q4 = (q_s[item_group])[base_q + lane_id + 96];

//             __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
//             tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//             __half ruv = r - tmp_product;

//             (p_s[user_group])[base_p + lane_id] = tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1);
//             (q_s[item_group])[base_q + lane_id] = tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1);

//             (p_s[user_group])[base_p + lane_id + 32] = tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2);
//             (q_s[item_group])[base_q + lane_id + 32] = tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2);

//             (p_s[user_group])[base_p + lane_id + 64] = tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3);
//             (q_s[item_group])[base_q + lane_id + 64] = tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3);

//             (p_s[user_group])[base_p + lane_id + 96] = tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4);
//             (q_s[item_group])[base_q + lane_id + 96] = tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4);
//         }    
//     }
// }

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info
                            )
{    
    __shared__ void* p_s[NUM_USER_GROUPS];
    __shared__ void* q_s[NUM_ITEM_GROUPS];
    __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
    __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];

    if (threadIdx.x < NUM_USER_GROUPS){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
    }
    if (threadIdx.x < NUM_USER_GROUPS){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
    }
    __syncthreads();
    
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];
            //! both precisions are half
            if (!user_prec && !item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4);
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_q1 = __float2half_rn(tmp_q1_f);
                __half tmp_q2 = __float2half_rn(tmp_q2_f);
                __half tmp_q3 = __float2half_rn(tmp_q3_f);
                __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                __half tmp_p1 = __float2half_rn(tmp_p1_f);
                __half tmp_p2 = __float2half_rn(tmp_p2_f);
                __half tmp_p3 = __float2half_rn(tmp_p3_f);
                __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4);      
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }
        }    
    }
}

//! ORIGINAL
__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            unsigned int* acc_user_group_error,
                            unsigned int* acc_item_group_error,
                            unsigned int* user_group_update_cnt,
                            unsigned int* item_group_update_cnt,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    // __shared__ void* p_s[NUM_USER_GROUPS];
    // __shared__ void* q_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
    // __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_update_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_update_cnt_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_zero_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_zero_cnt_s[NUM_ITEM_GROUPS];

    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_update_cnt_s = (unsigned int*)&q_s[item_group_num];
    unsigned int* item_group_update_cnt_s = (unsigned int*)&user_group_update_cnt_s[user_group_num];
    unsigned int* user_group_zero_cnt_s = (unsigned int*)&item_group_update_cnt_s[item_group_num];
    unsigned int* item_group_zero_cnt_s = (unsigned int*)&user_group_zero_cnt_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_zero_cnt_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_zero_cnt_s[threadIdx.x] = 0;
        user_group_update_cnt_s[threadIdx.x] = 0;
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_zero_cnt_s[threadIdx.x] = 0;
        item_group_update_cnt_s[threadIdx.x] = 0;
    }
    __syncthreads();

    unsigned int processed_cnt = 0;
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);
    __half zero = __float2half_rn(0.f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                // const __half tmp_p1_grad = lrate*((ruv*tmp_q1) - (lambda*tmp_p1));
                // const __half tmp_q1_grad = lrate*((ruv*tmp_p1) - (lambda*tmp_q1));
                // const __half tmp_p2_grad = lrate*((ruv*tmp_q2) - (lambda*tmp_p2));
                // const __half tmp_q2_grad = lrate*((ruv*tmp_p2) - (lambda*tmp_q2));
                // const __half tmp_p3_grad = lrate*((ruv*tmp_q3) - (lambda*tmp_p3));
                // const __half tmp_q3_grad = lrate*((ruv*tmp_p3) - (lambda*tmp_q3));
                // const __half tmp_p4_grad = lrate*((ruv*tmp_q4) - (lambda*tmp_p4));
                // const __half tmp_q4_grad = lrate*((ruv*tmp_p4) - (lambda*tmp_q4));
                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + (tmp_p1_grad);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + (tmp_q1_grad);
                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + (tmp_p2_grad);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + (tmp_q2_grad);
                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + (tmp_p3_grad);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + (tmp_q3_grad);
                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + (tmp_p4_grad);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + (tmp_q4_grad);

                const __half tmp_p1_grad = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_grad = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_grad = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_grad = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_grad = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_grad = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_grad = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_grad = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_grad, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_grad, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_grad ,tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_grad ,tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_grad ,tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_grad ,tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_grad ,tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_grad ,tmp_q4);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_zero_cnt = 0;
                    unsigned int item_zero_cnt = 0;
                    
                    if (lrate*tmp_p1_grad == zero) {user_zero_cnt += 1;}
                    if (lrate*tmp_q1_grad == zero) {item_zero_cnt += 1;}
                    if (lrate*tmp_p2_grad == zero) {user_zero_cnt += 1;}
                    if (lrate*tmp_q2_grad == zero) {item_zero_cnt += 1;}
                    if (lrate*tmp_p3_grad == zero) {user_zero_cnt += 1;}
                    if (lrate*tmp_q3_grad == zero) {item_zero_cnt += 1;}
                    if (lrate*tmp_p4_grad == zero) {user_zero_cnt += 1;}
                    if (lrate*tmp_q4_grad == zero) {item_zero_cnt += 1;}

                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);

                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);

                    if (lane_id == 0){
                        user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
                        item_group_zero_cnt_s[item_group] += item_zero_cnt;
                        user_group_update_cnt_s[user_group] += 1;
                        item_group_update_cnt_s[item_group] += 1;
                    }
                }
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];

                const __half tmp_q1 = __float2half_rn(tmp_q1_f);
                const __half tmp_q2 = __float2half_rn(tmp_q2_f);
                const __half tmp_q3 = __float2half_rn(tmp_q3_f);
                const __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                // const __half tmp_p1_grad = lrate*((ruv*tmp_q1) - (lambda*tmp_p1));
                // const __half tmp_p2_grad = lrate*((ruv*tmp_q2) - (lambda*tmp_p2));
                // const __half tmp_p3_grad = lrate*((ruv*tmp_q3) - (lambda*tmp_p3));
                // const __half tmp_p4_grad = lrate*((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_p1_grad = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_p2_grad = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_p3_grad = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_p4_grad = ((ruv*tmp_q4) - (lambda*tmp_p4));
                
                // tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + (tmp_p1_grad);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + (tmp_p2_grad);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + (tmp_p3_grad);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);
                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + (tmp_p4_grad);

                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_grad, tmp_p1);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_grad, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_grad, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_grad, tmp_p4);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_zero_cnt = 0;
                    if (lrate*tmp_p1_grad == zero) {user_zero_cnt += 1;}
                    if (lrate*tmp_p2_grad == zero) {user_zero_cnt += 1;}
                    if (lrate*tmp_p3_grad == zero) {user_zero_cnt += 1;}
                    if (lrate*tmp_p4_grad == zero) {user_zero_cnt += 1;}

                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
                        user_group_update_cnt_s[user_group] += 1;
                    }
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const __half tmp_p1 = __float2half_rn(tmp_p1_f);
                const __half tmp_p2 = __float2half_rn(tmp_p2_f);
                const __half tmp_p3 = __float2half_rn(tmp_p3_f);
                const __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                // const __half tmp_q1_grad = lrate*((ruv*tmp_p1) - (lambda*tmp_q1));
                // const __half tmp_q2_grad = lrate*((ruv*tmp_p2) - (lambda*tmp_q2));
                // const __half tmp_q3_grad = lrate*((ruv*tmp_p3) - (lambda*tmp_q3));
                // const __half tmp_q4_grad = lrate*((ruv*tmp_p4) - (lambda*tmp_q4));
                
                // tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + tmp_q1_grad;
                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + tmp_q2_grad;
                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + tmp_q3_grad;
                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + tmp_q4_grad;    
                
                const __half tmp_q1_grad = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_q2_grad = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_q3_grad = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_q4_grad = ((ruv*tmp_p4) - (lambda*tmp_q4));
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_grad, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_grad, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_grad, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_grad, tmp_q4);    

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_zero_cnt = 0;

                    if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;
                    
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        item_group_zero_cnt_s[item_group] += item_zero_cnt;
                        item_group_update_cnt_s[item_group] += 1;
                    }
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }
            processed_cnt += 1;
        }
    }

    if (threadIdx.x < user_group_num) {
        acc_user_group_error[gridDim.x * threadIdx.x + blockIdx.x] = user_group_zero_cnt_s[threadIdx.x];
        user_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = user_group_update_cnt_s[threadIdx.x];
    };
    if (threadIdx.x < item_group_num) {
        acc_item_group_error[gridDim.x * threadIdx.x + blockIdx.x] = item_group_zero_cnt_s[threadIdx.x];
        item_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = item_group_update_cnt_s[threadIdx.x];
    }
}

//! ORIGINAL
__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_fp32_version(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned int* acc_user_group_error,
                            unsigned int* acc_item_group_error,
                            unsigned int* user_group_update_cnt,
                            unsigned int* item_group_update_cnt,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_update_cnt_s = (unsigned int*)&q_s[item_group_num];
    unsigned int* item_group_update_cnt_s = (unsigned int*)&user_group_update_cnt_s[user_group_num];
    unsigned int* user_group_zero_cnt_s = (unsigned int*)&item_group_update_cnt_s[item_group_num];
    unsigned int* item_group_zero_cnt_s = (unsigned int*)&user_group_zero_cnt_s[user_group_num];

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_zero_cnt_s[threadIdx.x] = 0;
        user_group_update_cnt_s[threadIdx.x] = 0;
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_zero_cnt_s[threadIdx.x] = 0;
        item_group_update_cnt_s[threadIdx.x] = 0;
    }
    __syncthreads();
    
    unsigned int processed_cnt = 0;

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            float r = __ldg(&R[offset].r);
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            //! both precisions are single
            float* tmp_p_ptr = (float*)p_s[user_group];
            float* tmp_q_ptr = (float*)q_s[item_group];

            const float tmp_p1 = tmp_p_ptr[base_p + lane_id];
            const float tmp_q1 = tmp_q_ptr[base_q + lane_id];
            const float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            const float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            const float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            const float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            const float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            const float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

            float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            const float ruv = r - tmp_product;
            
            const float tmp_p1_grad = lrate*((ruv*tmp_q1) - (lambda*tmp_p1));
            const float tmp_q1_grad = lrate*((ruv*tmp_p1) - (lambda*tmp_q1));
            const float tmp_p2_grad = lrate*((ruv*tmp_q2) - (lambda*tmp_p2));
            const float tmp_q2_grad = lrate*((ruv*tmp_p2) - (lambda*tmp_q2));
            const float tmp_p3_grad = lrate*((ruv*tmp_q3) - (lambda*tmp_p3));
            const float tmp_q3_grad = lrate*((ruv*tmp_p3) - (lambda*tmp_q3));
            const float tmp_p4_grad = lrate*((ruv*tmp_q4) - (lambda*tmp_p4));
            const float tmp_q4_grad = lrate*((ruv*tmp_p4) - (lambda*tmp_q4));

            tmp_p_ptr[base_p + lane_id] = tmp_p1 + (tmp_p1_grad);
            tmp_q_ptr[base_q + lane_id] = tmp_q1 + (tmp_q1_grad);
            tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + (tmp_p2_grad);
            tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + (tmp_q2_grad);
            tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + (tmp_p3_grad);
            tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + (tmp_q3_grad);
            tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + (tmp_p4_grad);
            tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + (tmp_q4_grad);

            if (processed_cnt >= first_sample_rating_idx){
                unsigned int user_zero_cnt = 0;
                unsigned int item_zero_cnt = 0;
                
                // if (__float2half_rn(tmp_p1_grad) == __float2half_rn(0.0f)) {user_zero_cnt += 1;}
                // if (__float2half_rn(tmp_q1_grad) == __float2half_rn(0.0f)) {item_zero_cnt += 1;}
                // if (__float2half_rn(tmp_p2_grad) == __float2half_rn(0.0f)) {user_zero_cnt += 1;}
                // if (__float2half_rn(tmp_q2_grad) == __float2half_rn(0.0f)) {item_zero_cnt += 1;}
                // if (__float2half_rn(tmp_p3_grad) == __float2half_rn(0.0f)) {user_zero_cnt += 1;}
                // if (__float2half_rn(tmp_q3_grad) == __float2half_rn(0.0f)) {item_zero_cnt += 1;}
                // if (__float2half_rn(tmp_p4_grad) == __float2half_rn(0.0f)) {user_zero_cnt += 1;}
                // if (__float2half_rn(tmp_q4_grad) == __float2half_rn(0.0f)) {item_zero_cnt += 1;}

                if (tmp_p1_grad == 0.0f) {user_zero_cnt += 1;}
                if (tmp_q1_grad == 0.0f) {item_zero_cnt += 1;}
                if (tmp_p2_grad == 0.0f) {user_zero_cnt += 1;}
                if (tmp_q2_grad == 0.0f) {item_zero_cnt += 1;}
                if (tmp_p3_grad == 0.0f) {user_zero_cnt += 1;}
                if (tmp_q3_grad == 0.0f) {item_zero_cnt += 1;}
                if (tmp_p4_grad == 0.0f) {user_zero_cnt += 1;}
                if (tmp_q4_grad == 0.0f) {item_zero_cnt += 1;}

                user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
                user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
                user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
                user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
                user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);

                item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
                item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
                item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
                item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
                item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);

                if (lane_id == 0){
                    user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
                    item_group_zero_cnt_s[item_group] += item_zero_cnt;
                    user_group_update_cnt_s[user_group] += 1;
                    item_group_update_cnt_s[item_group] += 1;
                }
            }
            processed_cnt += 1;
        }
    }

    if (threadIdx.x < user_group_num) {
        acc_user_group_error[gridDim.x * threadIdx.x + blockIdx.x] = user_group_zero_cnt_s[threadIdx.x];
        user_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = user_group_update_cnt_s[threadIdx.x];
    };
    if (threadIdx.x < item_group_num) {
        acc_item_group_error[gridDim.x * threadIdx.x + blockIdx.x] = item_group_zero_cnt_s[threadIdx.x];
        item_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = item_group_update_cnt_s[threadIdx.x];
    }
}

// __global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error(
//                             const Node *R,
//                             unsigned int nnz,
//                             void** p,
//                             void** q,
//                             curandState *state,
//                             __half lrate,
//                             int k,
//                             int num_iters,
//                             int current_iter,
//                             int update_count_this_block,
//                             int update_vector_size,
//                             __half lambda,
//                             Index_info_node* user_index_info,
//                             Index_info_node* item_index_info,
//                             unsigned char* user_group_prec_info,
//                             unsigned char* item_group_prec_info,
//                             unsigned int* acc_user_group_error,
//                             unsigned int* acc_item_group_error,
//                             unsigned int* user_group_update_cnt,
//                             unsigned int* item_group_update_cnt,
//                             unsigned int first_sample_rating_idx
//                             )
// {    
//     __shared__ void* p_s[NUM_USER_GROUPS];
//     __shared__ void* q_s[NUM_ITEM_GROUPS];
//     __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
//     __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];
//     __shared__ unsigned int user_group_update_cnt_s[NUM_USER_GROUPS];
//     __shared__ unsigned int item_group_update_cnt_s[NUM_ITEM_GROUPS];
//     __shared__ unsigned int user_group_zero_cnt_s[NUM_USER_GROUPS];
//     __shared__ unsigned int item_group_zero_cnt_s[NUM_ITEM_GROUPS];

//     if (threadIdx.x < NUM_USER_GROUPS){
//         p_s[threadIdx.x] = p[threadIdx.x];
//         user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
//         user_group_zero_cnt_s[threadIdx.x] = 0;
//         user_group_update_cnt_s[threadIdx.x] = 0;
//     }
//     if (threadIdx.x < NUM_ITEM_GROUPS){
//         q_s[threadIdx.x] = q[threadIdx.x];
//         item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
//         item_group_zero_cnt_s[threadIdx.x] = 0;
//         item_group_update_cnt_s[threadIdx.x] = 0;
//     }
//     __syncthreads();
//     if (threadIdx.x == 0)
//         printf("%d",sizeof(void*));
//     unsigned int processed_cnt = 0;
//     float lrate_f = __half2float(lrate);
//     float lambda_f = __half2float(lambda);
//     __half zero = __float2half_rn(0.f);

//     for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
//     {
//         int lane_id = threadIdx.x%32;
//         int local_wid = threadIdx.x/32;
//         int local_w_num = blockDim.x/32;
//         int wid = local_w_num*blockIdx.x + local_wid;  
        
//         unsigned int start_id = 0;
//         if(lane_id == 0)
//         {
//             unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
//             start_id = origin%nnz;
//         }

//         start_id = __shfl_sync(0xffffffff,start_id, 0);
        
//         for(int i = 0;i < update_vector_size;i++)
//         {
//             int offset = (start_id + i)%nnz;
            
//             __half r = __float2half_rn(__ldg(&R[offset].r));
//             int orig_u = __ldg(&R[offset].u);
//             int orig_v = __ldg(&R[offset].i);
            
//             int user_group = user_index_info[orig_u].g;
//             int u = user_index_info[orig_u].v;
//             int item_group = item_index_info[orig_v].g;
//             int v = item_index_info[orig_v].v;
//             int base_p = u*k;
//             int base_q = v*k;

//             unsigned char user_prec = user_group_prec_s[user_group];
//             unsigned char item_prec = item_group_prec_s[item_group];

//             //! both precisions are half
//             if (!user_prec && !item_prec){
//                 unsigned int user_zero_cnt = 0;
//                 unsigned int item_zero_cnt = 0;
//                 __half* tmp_p_ptr = (__half*)p_s[user_group];
//                 __half* tmp_q_ptr = (__half*)q_s[item_group];

//                 __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 __half ruv = r - tmp_product;
                
//                 __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
//                 __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
//                 __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
//                 __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
//                 __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
//                 __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
//                 __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
//                 __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;

//                 tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;

//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
//                     if (lane_id == 0){
//                         user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
//                         item_group_zero_cnt_s[item_group] += item_zero_cnt;
//                         user_group_update_cnt_s[user_group] += 1;
//                         item_group_update_cnt_s[item_group] += 1;
//                     }
//                 }        
//             }
//             //! user half item single
//             else if (!user_prec && item_prec){
//                 unsigned int user_zero_cnt = 0;
//                 __half* tmp_p_ptr = (__half*)p_s[user_group];
//                 float* tmp_q_ptr = (float*)q_s[item_group];

//                 float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
//                 __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
//                 __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
//                 __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];
//                 __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];

//                 __half tmp_q1 = __float2half_rn(tmp_q1_f);
//                 __half tmp_q2 = __float2half_rn(tmp_q2_f);
//                 __half tmp_q3 = __float2half_rn(tmp_q3_f);
//                 __half tmp_q4 = __float2half_rn(tmp_q4_f);

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 __half ruv = r - tmp_product;

//                 __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
//                 __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
//                 __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
//                 __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
                
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
//                 tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);
//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;

//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
//                     if (lane_id == 0){
//                         user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
//                         user_group_update_cnt_s[user_group] += 1;
//                     }
//                 }        
//             }
//             //! user single item half
//             else if (user_prec && !item_prec){
//                 unsigned int item_zero_cnt = 0;
//                 // if (lane_id == 0)
//                 // printf("%d ",processed_cnt);
//                 float* tmp_p_ptr = (float*)p_s[user_group];
//                 __half* tmp_q_ptr = (__half*)q_s[item_group];

//                 float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
//                 __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
//                 __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
//                 __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
//                 __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
//                 __half tmp_p1 = __float2half_rn(tmp_p1_f);
//                 __half tmp_p2 = __float2half_rn(tmp_p2_f);
//                 __half tmp_p3 = __float2half_rn(tmp_p3_f);
//                 __half tmp_p4 = __float2half_rn(tmp_p4_f);

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 __half ruv = r - tmp_product;
                
//                 __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
//                 __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
//                 __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
//                 __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;
                
//                 tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);    
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;
                    
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
//                     if (lane_id == 0){
//                         item_group_zero_cnt_s[item_group] += item_zero_cnt;
//                         item_group_update_cnt_s[item_group] += 1;
//                     }
//                 }     
//             }
//             //! user single item single
//             else{
//                 // if (lane_id == 0)
//                 // printf("%d ",processed_cnt);
//                 float* tmp_p_ptr = (float*)p_s[user_group];
//                 float* tmp_q_ptr = (float*)q_s[item_group];

//                 float tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 float tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 float ruv = __half2float(r) - tmp_product;

//                 tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
//             }

//             processed_cnt += 1;
//         }
//     }

//     if (threadIdx.x < NUM_USER_GROUPS) {
//         acc_user_group_error[gridDim.x * threadIdx.x + blockIdx.x] = user_group_zero_cnt_s[threadIdx.x];
//         user_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = user_group_update_cnt_s[threadIdx.x];
//     };
//     if (threadIdx.x < NUM_ITEM_GROUPS) {
//         acc_item_group_error[gridDim.x * threadIdx.x + blockIdx.x] = item_group_zero_cnt_s[threadIdx.x];
//         item_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = item_group_update_cnt_s[threadIdx.x];
//     }
// }

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_range_based_indexing_eval_only_idexing(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&q_s[item_group_num];
    unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];
    
    for (int i = threadIdx.x; i < user_group_num; i+=blockDim.x){
        p_s[i] = p[i];
        user_group_prec_s[i] = user_group_prec_info[i];
        user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }

    for (int i = threadIdx.x; i < item_group_num; i+=blockDim.x){
        q_s[i] = q[i];
        item_group_prec_s[i] = item_group_prec_info[i];
        item_group_end_idx_s[i+1] = item_group_end_idx[i];
    }
    // if (threadIdx.x < user_group_num){
    //     p_s[threadIdx.x] = p[threadIdx.x];
    //     user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
    //     user_group_end_idx_s[threadIdx.x+1] = user_group_end_idx[threadIdx.x];
    // }

    // if (threadIdx.x < item_group_num) {
    //     q_s[threadIdx.x] = q[threadIdx.x];
    //     item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
    //     item_group_end_idx_s[threadIdx.x+1] = item_group_end_idx[threadIdx.x];
    // }
    
    if (threadIdx.x == 0){
        user_group_end_idx_s[0] = -1;
        item_group_end_idx_s[0] = -1;      
    }
    __syncthreads();
    __half lrate = __float2half(lrate_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

    //     // All threads read x from laneid 0
        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = 0;
            int item_group = 0;

            for (int t = 0; t < user_group_num; t+=32){
                int val = 0;
                if (t + lane_id < user_group_num){
                    int from = user_group_end_idx_s[t+lane_id];
                    int to = user_group_end_idx_s[t+lane_id+1];
                    val = (u > from && u <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    user_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }

            if (user_group != 0)
                u = u-user_group_end_idx_s[user_group]-1;

            for (int t = 0; t < item_group_num; t+=32){
                int val = 0;
                if (t + lane_id < item_group_num){
                    int from = item_group_end_idx_s[t+lane_id];
                    int to = item_group_end_idx_s[t+lane_id+1];
                    val = (v > from && v <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    item_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }
            
            if (item_group != 0)
                v = v-item_group_end_idx_s[item_group]-1;

            int base_p = u*k;
            int base_q = v*k;
            
            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            __half r_h = __float2half(r);
            __half* tmp_p_ptr = (__half*)p_s[user_group];
            __half* tmp_q_ptr = (__half*)q_s[item_group];

            const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
            const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
            const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

            __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            const __half ruv = r_h - tmp_product;
            
            const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
            const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
            const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
            const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
            const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
            const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
            const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
            const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

            tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
            tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
            tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
            tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
            tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
            tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
            tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
            tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);               
        }    
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_range_based_indexing(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            unsigned int* acc_user_group_error,
                            unsigned int* acc_item_group_error,
                            unsigned int* user_group_update_cnt,
                            unsigned int* item_group_update_cnt,
                            unsigned int first_sample_rating_idx,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_update_cnt_s = (unsigned int*)&q_s[item_group_num];
    unsigned int* item_group_update_cnt_s = (unsigned int*)&user_group_update_cnt_s[user_group_num];
    unsigned int* user_group_zero_cnt_s = (unsigned int*)&item_group_update_cnt_s[item_group_num];
    unsigned int* item_group_zero_cnt_s = (unsigned int*)&user_group_zero_cnt_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_zero_cnt_s[item_group_num];
    unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];


    // __shared__ void* p_s[NUM_USER_GROUPS];
    // __shared__ void* q_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
    // __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_update_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_update_cnt_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_zero_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_zero_cnt_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_end_idx_s[NUM_USER_GROUPS + 1];
    // __shared__ unsigned int item_group_end_idx_s[NUM_ITEM_GROUPS + 1];


    // __shared__ int user_ans[4];
    // __shared__ int item_ans[4];
    // unsigned int user_loop_num = ceilf(user_group_num/(float)32);
    // unsigned int item_loop_num = ceilf(item_group_num/(float)32);

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_zero_cnt_s[threadIdx.x] = 0;
        user_group_update_cnt_s[threadIdx.x] = 0;
        user_group_end_idx_s[threadIdx.x+1] = user_group_end_idx[threadIdx.x];
    }

    if (threadIdx.x < item_group_num) {
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_zero_cnt_s[threadIdx.x] = 0;
        item_group_update_cnt_s[threadIdx.x] = 0;
        item_group_end_idx_s[threadIdx.x+1] = item_group_end_idx[threadIdx.x];
    }
    
    if (threadIdx.x == 0){
        user_group_end_idx_s[0] = -1;
        item_group_end_idx_s[0] = -1;      
    }

    __syncthreads();
    unsigned int processed_cnt = 0;
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);
    __half zero = __float2half_rn(0.f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

    //     // All threads read x from laneid 0
        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = 0;
            int item_group = 0;

            for (int t = 0; t < user_group_num; t+=32){
                int val = 0;
                if (t + lane_id < user_group_num){
                    int from = user_group_end_idx_s[t+lane_id];
                    int to = user_group_end_idx_s[t+lane_id+1];
                    val = (u > from && u <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    user_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }

            if (user_group != 0)
                u = u-user_group_end_idx_s[user_group]-1;

            for (int t = 0; t < item_group_num; t+=32){
                int val = 0;
                if (t + lane_id < item_group_num){
                    int from = item_group_end_idx_s[t+lane_id];
                    int to = item_group_end_idx_s[t+lane_id+1];
                    val = (v > from && v <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    item_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }
            
            if (item_group != 0)
                v = v-item_group_end_idx_s[item_group]-1;

            int base_p = u*k;
            int base_q = v*k;
            
            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                unsigned int user_zero_cnt = 0;
                unsigned int item_zero_cnt = 0;
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;
                
                __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
                __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
                __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
                __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
                __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
                __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
                __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
                __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);
                
                if (processed_cnt >= first_sample_rating_idx){
                    if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;

                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
                        item_group_zero_cnt_s[item_group] += item_zero_cnt;
                        user_group_update_cnt_s[user_group] += 1;
                        item_group_update_cnt_s[item_group] += 1;
                    }
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                unsigned int user_zero_cnt = 0;
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];
                __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];

                __half tmp_q1 = __float2half_rn(tmp_q1_f);
                __half tmp_q2 = __float2half_rn(tmp_q2_f);
                __half tmp_q3 = __float2half_rn(tmp_q3_f);
                __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;

                __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
                __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
                __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
                __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
                
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
                
                if (processed_cnt >= first_sample_rating_idx){
                    if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;

                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
                        user_group_update_cnt_s[user_group] += 1;
                    }
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                unsigned int item_zero_cnt = 0;
                // if (lane_id == 0)
                // printf("%d ",processed_cnt);
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                __half tmp_p1 = __float2half_rn(tmp_p1_f);
                __half tmp_p2 = __float2half_rn(tmp_p2_f);
                __half tmp_p3 = __float2half_rn(tmp_p3_f);
                __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;
                
                __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
                __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
                __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
                __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);    
                
                if (processed_cnt >= first_sample_rating_idx){
                    if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;
                    
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        item_group_zero_cnt_s[item_group] += item_zero_cnt;
                        item_group_update_cnt_s[item_group] += 1;
                    }
                }     
            }
            //! user single item single
            else{
                // if (lane_id == 0)
                // printf("%d ",processed_cnt);
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }

            processed_cnt += 1;

        }    
    }

    if (threadIdx.x < user_group_num) {
        acc_user_group_error[gridDim.x * threadIdx.x + blockIdx.x] = user_group_zero_cnt_s[threadIdx.x];
        user_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = user_group_update_cnt_s[threadIdx.x];
    };
    if (threadIdx.x < item_group_num) {
        acc_item_group_error[gridDim.x * threadIdx.x + blockIdx.x] = item_group_zero_cnt_s[threadIdx.x];
        item_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = item_group_update_cnt_s[threadIdx.x];
    }
}

// __global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_range_based_indexing(
//                             const Node *R,
//                             unsigned int nnz,
//                             void** p,
//                             void** q,
//                             curandState *state,
//                             __half lrate,
//                             int k,
//                             int num_iters,
//                             int current_iter,
//                             int update_count_this_block,
//                             int update_vector_size,
//                             __half lambda,
//                             unsigned char* user_group_prec_info,
//                             unsigned char* item_group_prec_info,
//                             unsigned int* acc_user_group_error,
//                             unsigned int* acc_item_group_error,
//                             unsigned int* user_group_update_cnt,
//                             unsigned int* item_group_update_cnt,
//                             unsigned int first_sample_rating_idx,
//                             unsigned int* user_group_end_idx,
//                             unsigned int* item_group_end_idx
//                             )
// {    

//     __shared__ void* p_s[NUM_USER_GROUPS];
//     __shared__ void* q_s[NUM_ITEM_GROUPS];
//     __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
//     __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];
//     __shared__ unsigned int user_group_update_cnt_s[NUM_USER_GROUPS];
//     __shared__ unsigned int item_group_update_cnt_s[NUM_ITEM_GROUPS];
//     __shared__ unsigned int user_group_zero_cnt_s[NUM_USER_GROUPS];
//     __shared__ unsigned int item_group_zero_cnt_s[NUM_ITEM_GROUPS];
//     __shared__ unsigned int user_group_end_idx_s[NUM_USER_GROUPS + 1];
//     __shared__ unsigned int item_group_end_idx_s[NUM_ITEM_GROUPS + 1];


//     // __shared__ int user_ans[4];
//     // __shared__ int item_ans[4];
//     // unsigned int user_loop_num = ceilf(user_group_num/(float)32);
//     // unsigned int item_loop_num = ceilf(item_group_num/(float)32);

//     if (threadIdx.x < NUM_USER_GROUPS){
//         p_s[threadIdx.x] = p[threadIdx.x];
//         user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
//         user_group_zero_cnt_s[threadIdx.x] = 0;
//         user_group_update_cnt_s[threadIdx.x] = 0;
//         user_group_end_idx_s[threadIdx.x+1] = user_group_end_idx[threadIdx.x];
//     }

//     if (threadIdx.x < NUM_ITEM_GROUPS) {
//         q_s[threadIdx.x] = q[threadIdx.x];
//         item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
//         item_group_zero_cnt_s[threadIdx.x] = 0;
//         item_group_update_cnt_s[threadIdx.x] = 0;
//         item_group_end_idx_s[threadIdx.x+1] = item_group_end_idx[threadIdx.x];
//     }
    
//     if (threadIdx.x == 0){
//         user_group_end_idx_s[0] = -1;
//         item_group_end_idx_s[0] = -1;      
//     }

//     __syncthreads();
//     unsigned int processed_cnt = 0;
//     float lrate_f = __half2float(lrate);
//     float lambda_f = __half2float(lambda);
//     __half zero = __float2half_rn(0.f);

//     for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
//     {
//         int lane_id = threadIdx.x%32;
//         int local_wid = threadIdx.x/32;
//         int local_w_num = blockDim.x/32;
//         int wid = local_w_num*blockIdx.x + local_wid;  
        
//         unsigned int start_id = 0;
//         if(lane_id == 0)
//         {
//             unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
//             start_id = origin%nnz;
//         }

//     //     // All threads read x from laneid 0
//         start_id = __shfl_sync(0xffffffff,start_id, 0);
        
//         for(int i = 0;i < update_vector_size;i++)
//         {
//             int offset = (start_id + i)%nnz;
            
//             __half r = __float2half_rn(__ldg(&R[offset].r));
//             int u = __ldg(&R[offset].u);
//             int v = __ldg(&R[offset].i);
            
//             int user_group = 0;
//             int item_group = 0;

//             for (int t = 0; t < NUM_USER_GROUPS; t+=32){
//                 int val = 0;
//                 if (t + lane_id < NUM_USER_GROUPS){
//                     int from = user_group_end_idx_s[t+lane_id];
//                     int to = user_group_end_idx_s[t+lane_id+1];
//                     val = (u > from && u <= to) * (t + lane_id);
//                 }
//                 unsigned bitpack = __ballot_sync(0xffffffff, val);
//                 if (bitpack != 0){
//                     user_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
//                     break;
//                 }
//             }

//             if (user_group != 0)
//                 u = u-user_group_end_idx_s[user_group]-1;

//             for (int t = 0; t < NUM_ITEM_GROUPS; t+=32){
//                 int val = 0;
//                 if (t + lane_id < NUM_ITEM_GROUPS){
//                     int from = item_group_end_idx_s[t+lane_id];
//                     int to = item_group_end_idx_s[t+lane_id+1];
//                     val = (v > from && v <= to) * (t + lane_id);
//                 }
//                 unsigned bitpack = __ballot_sync(0xffffffff, val);
//                 if (bitpack != 0){
//                     item_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
//                     break;
//                 }
//             }
            
//             if (item_group != 0)
//                 v = v-item_group_end_idx_s[item_group]-1;

//             int base_p = u*k;
//             int base_q = v*k;
            
//             unsigned char user_prec = user_group_prec_s[user_group];
//             unsigned char item_prec = item_group_prec_s[item_group];

//             //! both precisions are half
//             if (!user_prec && !item_prec){
//                 unsigned int user_zero_cnt = 0;
//                 unsigned int item_zero_cnt = 0;
//                 __half* tmp_p_ptr = (__half*)p_s[user_group];
//                 __half* tmp_q_ptr = (__half*)q_s[item_group];

//                 __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 __half ruv = r - tmp_product;
                
//                 __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
//                 __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
//                 __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
//                 __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
//                 __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
//                 __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
//                 __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
//                 __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;

//                 tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;

//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
//                     if (lane_id == 0){
//                         user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
//                         item_group_zero_cnt_s[item_group] += item_zero_cnt;
//                         user_group_update_cnt_s[user_group] += 1;
//                         item_group_update_cnt_s[item_group] += 1;
//                     }
//                 }        
//             }
//             //! user half item single
//             else if (!user_prec && item_prec){
//                 unsigned int user_zero_cnt = 0;
//                 __half* tmp_p_ptr = (__half*)p_s[user_group];
//                 float* tmp_q_ptr = (float*)q_s[item_group];

//                 float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
//                 __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
//                 __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
//                 __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];
//                 __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];

//                 __half tmp_q1 = __float2half_rn(tmp_q1_f);
//                 __half tmp_q2 = __float2half_rn(tmp_q2_f);
//                 __half tmp_q3 = __float2half_rn(tmp_q3_f);
//                 __half tmp_q4 = __float2half_rn(tmp_q4_f);

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 __half ruv = r - tmp_product;

//                 __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
//                 __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
//                 __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
//                 __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
                
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
//                 tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);
//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
//                     if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;

//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
//                     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
//                     if (lane_id == 0){
//                         user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
//                         user_group_update_cnt_s[user_group] += 1;
//                     }
//                 }        
//             }
//             //! user single item half
//             else if (user_prec && !item_prec){
//                 unsigned int item_zero_cnt = 0;
//                 // if (lane_id == 0)
//                 // printf("%d ",processed_cnt);
//                 float* tmp_p_ptr = (float*)p_s[user_group];
//                 __half* tmp_q_ptr = (__half*)q_s[item_group];

//                 float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
//                 __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
//                 __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
//                 __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
//                 __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
//                 __half tmp_p1 = __float2half_rn(tmp_p1_f);
//                 __half tmp_p2 = __float2half_rn(tmp_p2_f);
//                 __half tmp_p3 = __float2half_rn(tmp_p3_f);
//                 __half tmp_p4 = __float2half_rn(tmp_p4_f);

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 __half ruv = r - tmp_product;
                
//                 __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
//                 __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
//                 __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
//                 __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;
                
//                 tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);    
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
//                     if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;
                    
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
//                     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
//                     if (lane_id == 0){
//                         item_group_zero_cnt_s[item_group] += item_zero_cnt;
//                         item_group_update_cnt_s[item_group] += 1;
//                     }
//                 }     
//             }
//             //! user single item single
//             else{
//                 // if (lane_id == 0)
//                 // printf("%d ",processed_cnt);
//                 float* tmp_p_ptr = (float*)p_s[user_group];
//                 float* tmp_q_ptr = (float*)q_s[item_group];

//                 float tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 float tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 float ruv = __half2float(r) - tmp_product;

//                 tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

//                 tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

//                 tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

//                 tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
//             }

//             processed_cnt += 1;

//         }    
//     }

//     if (threadIdx.x < NUM_USER_GROUPS) {
//         acc_user_group_error[gridDim.x * threadIdx.x + blockIdx.x] = user_group_zero_cnt_s[threadIdx.x];
//         user_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = user_group_update_cnt_s[threadIdx.x];
//     };
//     if (threadIdx.x < NUM_ITEM_GROUPS) {
//         acc_item_group_error[gridDim.x * threadIdx.x + blockIdx.x] = item_group_zero_cnt_s[threadIdx.x];
//         item_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = item_group_update_cnt_s[threadIdx.x];
//     }
// }


// #ifdef GRADIENT_DIVERSITY
__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    
    // __shared__ void* p_s[NUM_USER_GROUPS];
    // __shared__ void* q_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
    // __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_update_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_update_cnt_s[NUM_ITEM_GROUPS];
    // __shared__ float user_group_sum_updated_val_s[NUM_USER_GROUPS*128];
    // __shared__ float item_group_sum_updated_val_s[NUM_ITEM_GROUPS*128];
    // __shared__ float user_group_sum_norms_s[NUM_USER_GROUPS];
    // __shared__ float item_group_sum_norms_s[NUM_ITEM_GROUPS];

    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_update_cnt_s = (unsigned int*)&q_s[item_group_num];
    unsigned int* item_group_update_cnt_s = (unsigned int*)&user_group_update_cnt_s[user_group_num];
    float* user_group_sum_updated_val_s = (float*)&item_group_update_cnt_s[item_group_num]; 
    float* item_group_sum_updated_val_s = (float*)&user_group_sum_updated_val_s[user_group_num * k];
    float* user_group_sum_norms_s = (float*)&item_group_sum_updated_val_s[item_group_num * k];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_sum_norms_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int tid = threadIdx.x;

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_sum_norms_s[threadIdx.x] = 0.f;
        user_group_update_cnt_s[threadIdx.x] = 0;
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_sum_norms_s[threadIdx.x] = 0.f;
        item_group_update_cnt_s[threadIdx.x] = 0;
    }

    for (;tid<user_group_num*128;tid+=blockDim.x) user_group_sum_updated_val_s[tid] = 0.f;
    tid = threadIdx.x;
    for (;tid<item_group_num*128;tid+=blockDim.x) item_group_sum_updated_val_s[tid] = 0.f;

    // 0으로 초기화
    __syncthreads();

    unsigned int processed_cnt = 0;
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);
    __half zero = __float2half_rn(0.f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                // const __half tmp_p1_updated_val = lrate*(ruv*tmp_q1 - lambda*tmp_p1);
                // const __half tmp_q1_updated_val = lrate*(ruv*tmp_p1 - lambda*tmp_q1);
                // const __half tmp_p2_updated_val = lrate*(ruv*tmp_q2 - lambda*tmp_p2);
                // const __half tmp_q2_updated_val = lrate*(ruv*tmp_p2 - lambda*tmp_q2);
                // const __half tmp_p3_updated_val = lrate*(ruv*tmp_q3 - lambda*tmp_p3);
                // const __half tmp_q3_updated_val = lrate*(ruv*tmp_p3 - lambda*tmp_q3);
                // const __half tmp_p4_updated_val = lrate*(ruv*tmp_q4 - lambda*tmp_p4);
                // const __half tmp_q4_updated_val = lrate*(ruv*tmp_p4 - lambda*tmp_q4);

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    const unsigned int user_group_base = user_group * k;
                    const unsigned int item_group_base = item_group * k;
                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    // float norm_p = __powf(tmp_p1_updated_val_f,2) + __powf(tmp_p2_updated_val_f,2) + __powf(tmp_p3_updated_val_f,2) + __powf(tmp_p4_updated_val_f,2); // longer 
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        // user_group_sum_norms_s[user_group] += norm_p;
                        // item_group_sum_norms_s[item_group] += norm_q;

                        atomicAdd(user_group_sum_norms_s + user_group, norm_p);
                        atomicAdd(item_group_sum_norms_s + item_group, norm_q);
                    }

                    // user_group_sum_updated_val_s[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const __half tmp_q1 = __float2half_rn(tmp_q1_f);
                const __half tmp_q2 = __float2half_rn(tmp_q2_f);
                const __half tmp_q3 = __float2half_rn(tmp_q3_f);
                const __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);

                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base = user_group * k;
                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        // user_group_sum_norms_s[user_group] += (((norm_p)));

                        atomicAdd(user_group_sum_norms_s + user_group, norm_p);
                    }

                    // user_group_sum_updated_val_s[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);

                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const __half tmp_p1 = __float2half_rn(tmp_p1_f);
                const __half tmp_p2 = __float2half_rn(tmp_p2_f);
                const __half tmp_p3 = __float2half_rn(tmp_p3_f);
                const __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);    
                
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_group_base = item_group * k;
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        // item_group_sum_norms_s[item_group] += (((norm_q)));

                        atomicAdd(item_group_sum_norms_s + item_group, norm_q);
                    }

                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }
            processed_cnt += 1;
        }
    }
    __syncthreads();

    int local_wid = threadIdx.x/32;
    int local_w_num = blockDim.x/32;
    int lane_id = threadIdx.x%32;

    for (; local_wid < user_group_num; local_wid += local_w_num){
        int base_user_group = local_wid * k;

        // grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val_s[base_user_group + lane_id];
        // grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val_s[base_user_group + lane_id + 32];
        // grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val_s[base_user_group + lane_id + 64];
        // grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val_s[base_user_group + lane_id + 96];

        atomicAdd(grad_sum_norm_p + base_user_group + lane_id, user_group_sum_updated_val_s[base_user_group + lane_id]);
        atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 32, user_group_sum_updated_val_s[base_user_group + lane_id + 32]);
        atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 64, user_group_sum_updated_val_s[base_user_group + lane_id + 64]);
        atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 96, user_group_sum_updated_val_s[base_user_group + lane_id + 96]);

    }

    local_wid = threadIdx.x/32;
    
    for (; local_wid < item_group_num; local_wid += local_w_num){
        int base_item_group = local_wid * k;
       
        // grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val_s[base_item_group + lane_id];
        // grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val_s[base_item_group + lane_id + 32];
        // grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val_s[base_item_group + lane_id + 64];
        // grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val_s[base_item_group + lane_id + 96];
        
        atomicAdd(grad_sum_norm_q + base_item_group + lane_id, item_group_sum_updated_val_s[base_item_group + lane_id]);
        atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 32, item_group_sum_updated_val_s[base_item_group + lane_id + 32]);
        atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 64, item_group_sum_updated_val_s[base_item_group + lane_id + 64]);
        atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 96, item_group_sum_updated_val_s[base_item_group + lane_id + 96]);
    }

    if (threadIdx.x < user_group_num) {
        norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    }

    if (threadIdx.x < item_group_num) {
        norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_eval_only_indexing(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&q_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int tid = threadIdx.x;

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
    }

    __syncthreads();

    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
   
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const __half tmp_q1 = __float2half_rn(tmp_q1_f);
                const __half tmp_q2 = __float2half_rn(tmp_q2_f);
                const __half tmp_q3 = __float2half_rn(tmp_q3_f);
                const __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const __half tmp_p1 = __float2half_rn(tmp_p1_f);
                const __half tmp_p2 = __float2half_rn(tmp_p2_f);
                const __half tmp_p3 = __float2half_rn(tmp_p3_f);
                const __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);    
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }
        }
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_grouped_cache(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num,
                            unsigned int uncached_user_num,
                            unsigned int uncached_item_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_updated_val_s = (float*)&q_s[item_group_num]; 
    float* item_group_sum_updated_val_s = (float*)&user_group_sum_updated_val_s[(user_group_num - uncached_user_num) * k];
    float* user_group_sum_norms_s = (float*)&item_group_sum_updated_val_s[(item_group_num - uncached_item_num) * k];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_sum_norms_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int tid = threadIdx.x;

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_sum_norms_s[threadIdx.x] = 0.f;
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_sum_norms_s[threadIdx.x] = 0.f;
    }

    // for (;tid<(user_group_num - uncached_user_num)*128;tid+=blockDim.x) user_group_sum_updated_val_s[tid] = 0.f;
    // tid = threadIdx.x;
    // for (;tid<(item_group_num - uncached_item_num)*128;tid+=blockDim.x) item_group_sum_updated_val_s[tid] = 0.f;

    // 0으로 초기화
    __syncthreads();

    unsigned int processed_cnt = 0;
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);
    __half zero = __float2half_rn(0.f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                // const __half tmp_p1_updated_val = lrate*(ruv*tmp_q1 - lambda*tmp_p1);
                // const __half tmp_q1_updated_val = lrate*(ruv*tmp_p1 - lambda*tmp_q1);
                // const __half tmp_p2_updated_val = lrate*(ruv*tmp_q2 - lambda*tmp_p2);
                // const __half tmp_q2_updated_val = lrate*(ruv*tmp_p2 - lambda*tmp_q2);
                // const __half tmp_p3_updated_val = lrate*(ruv*tmp_q3 - lambda*tmp_p3);
                // const __half tmp_q3_updated_val = lrate*(ruv*tmp_p3 - lambda*tmp_q3);
                // const __half tmp_p4_updated_val = lrate*(ruv*tmp_q4 - lambda*tmp_p4);
                // const __half tmp_q4_updated_val = lrate*(ruv*tmp_p4 - lambda*tmp_q4);

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    // float norm_p = __powf(tmp_p1_updated_val_f,2) + __powf(tmp_p2_updated_val_f,2) + __powf(tmp_p3_updated_val_f,2) + __powf(tmp_p4_updated_val_f,2); // longer 
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                        item_group_sum_norms_s[item_group] += norm_q;
                    }

                    if (user_group >= uncached_user_num) {
                        user_group_base = (user_group - uncached_user_num) * k;
                        // user_group_sum_updated_val_s[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);

                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);                        
                    } 
                    else {
                        user_group_base = user_group * k;
                        // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                        
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    }

                    if (item_group >= uncached_item_num) {
                        item_group_base = (item_group - uncached_item_num) * k;
                        // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    else {
                        item_group_base = item_group * k;
                        // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                        
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const __half tmp_q1 = __float2half_rn(tmp_q1_f);
                const __half tmp_q2 = __float2half_rn(tmp_q2_f);
                const __half tmp_q3 = __float2half_rn(tmp_q3_f);
                const __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);

                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                
                if (processed_cnt >= first_sample_rating_idx){
                    // unsigned int user_group_base = user_group * k;
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += (((norm_p)));
                    }

                    if (user_group >= uncached_user_num) {
                        user_group_base = (user_group - uncached_user_num) * k;
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);                        
                    } 
                    else {
                        user_group_base = user_group * k;
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);
                    }
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const __half tmp_p1 = __float2half_rn(tmp_p1_f);
                const __half tmp_p2 = __float2half_rn(tmp_p2_f);
                const __half tmp_p3 = __float2half_rn(tmp_p3_f);
                const __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);    
                
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    // unsigned int item_group_base = item_group * k;
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        item_group_sum_norms_s[item_group] += (((norm_q)));
                    }

                    if (item_group >= uncached_item_num) {
                        item_group_base = (item_group - uncached_item_num) * k;
                        // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    else {
                        item_group_base = item_group * k;
                        // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                        
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }

            processed_cnt += 1;
        }
    }

    __syncthreads();

    int local_wid = threadIdx.x/32;
    int local_w_num = blockDim.x/32;
    int lane_id = threadIdx.x%32;

    // for (; local_wid < user_group_num; local_wid += local_w_num){
    //     int base_user_group = local_wid * k;
    //     if (local_wid >= uncached_user_num){
    //         int converted_user_group = (local_wid - uncached_user_num) * k;
    //         grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val_s[converted_user_group + lane_id];
    //         grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val_s[converted_user_group + lane_id + 32];
    //         grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val_s[converted_user_group + lane_id + 64];
    //         grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val_s[converted_user_group + lane_id + 96];
    //     }
    //     // else{
    //     //     // grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val[base_user_group + lane_id];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val[base_user_group + lane_id + 32];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val[base_user_group + lane_id + 64];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val[base_user_group + lane_id + 96];

    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id, user_group_sum_updated_val[base_user_group + lane_id]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 32, user_group_sum_updated_val[base_user_group + lane_id + 32]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 64, user_group_sum_updated_val[base_user_group + lane_id + 64]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 96, user_group_sum_updated_val[base_user_group + lane_id + 96]);
    //     // }
    // }

    // local_wid = threadIdx.x/32;
    
    // for (; local_wid < item_group_num; local_wid += local_w_num){
    //     int base_item_group = local_wid * k;
    //     if (local_wid >= uncached_item_num){
    //         int converted_item_group = (local_wid - uncached_item_num) * k;
    //         grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val_s[converted_item_group + lane_id];
    //         grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val_s[converted_item_group + lane_id + 32];
    //         grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val_s[converted_item_group + lane_id + 64];
    //         grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val_s[converted_item_group + lane_id + 96];
    //     }
    //     // else {
    //     //     // grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val[base_item_group + lane_id];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val[base_item_group + lane_id + 32];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val[base_item_group + lane_id + 64];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val[base_item_group + lane_id + 96];
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id, item_group_sum_updated_val[base_item_group + lane_id]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 32, item_group_sum_updated_val[base_item_group + lane_id + 32]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 64, item_group_sum_updated_val[base_item_group + lane_id + 64]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 96, item_group_sum_updated_val[base_item_group + lane_id + 96]);
    //     // }
    // }

    if (threadIdx.x < user_group_num) {
        norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    }

    if (threadIdx.x < item_group_num) {
        norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    }
}


__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            // Index_info_node* user_index_info,
                            // Index_info_node* item_index_info,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num,
                            unsigned int uncached_user_num,
                            unsigned int uncached_item_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_updated_val_s = (float*)&q_s[item_group_num]; 
    float* item_group_sum_updated_val_s = (float*)&user_group_sum_updated_val_s[(user_group_num - uncached_user_num) * k];
    float* user_group_sum_norms_s = (float*)&item_group_sum_updated_val_s[(item_group_num - uncached_item_num) * k];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int tid = threadIdx.x;

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_sum_norms_s[threadIdx.x] = 0.f;
        user_group_end_idx_s[threadIdx.x+1] = user_group_end_idx[threadIdx.x];    
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_sum_norms_s[threadIdx.x] = 0.f;
        item_group_end_idx_s[threadIdx.x+1] = item_group_end_idx[threadIdx.x];
    }
    
    if (threadIdx.x == 0){
        user_group_end_idx_s[0] = -1;
        item_group_end_idx_s[0] = -1;      
    }
    
    // for (;tid<(user_group_num - uncached_user_num)*128;tid+=blockDim.x) user_group_sum_updated_val_s[tid] = 0.f;
    // tid = threadIdx.x;
    // for (;tid<(item_group_num - uncached_item_num)*128;tid+=blockDim.x) item_group_sum_updated_val_s[tid] = 0.f;

    // 0으로 초기화
    __syncthreads();

    unsigned int processed_cnt = 0;
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);
    __half zero = __float2half_rn(0.f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = 0;
            int item_group = 0;

            for (int t = 0; t < user_group_num; t+=32){
                int val = 0;
                if (t + lane_id < user_group_num){
                    int from = user_group_end_idx_s[t+lane_id];
                    int to = user_group_end_idx_s[t+lane_id+1];
                    val = (u > from && u <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    user_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }

            if (user_group != 0)
                u = u-user_group_end_idx_s[user_group]-1;

            for (int t = 0; t < item_group_num; t+=32){
                int val = 0;
                if (t + lane_id < item_group_num){
                    int from = item_group_end_idx_s[t+lane_id];
                    int to = item_group_end_idx_s[t+lane_id+1];
                    val = (v > from && v <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    item_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }
            
            if (item_group != 0)
                v = v-item_group_end_idx_s[item_group]-1;

            int base_p = u*k;
            int base_q = v*k;

            // __half r = __float2half_rn(__ldg(&R[offset].r));
            // int orig_u = __ldg(&R[offset].u);
            // int orig_v = __ldg(&R[offset].i);
            
            // int user_group = user_index_info[orig_u].g;
            // int u = user_index_info[orig_u].v;
            // int item_group = item_index_info[orig_v].g;
            // int v = item_index_info[orig_v].v;
            // int base_p = u*k;
            // int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                // const __half tmp_p1_updated_val = lrate*(ruv*tmp_q1 - lambda*tmp_p1);
                // const __half tmp_q1_updated_val = lrate*(ruv*tmp_p1 - lambda*tmp_q1);
                // const __half tmp_p2_updated_val = lrate*(ruv*tmp_q2 - lambda*tmp_p2);
                // const __half tmp_q2_updated_val = lrate*(ruv*tmp_p2 - lambda*tmp_q2);
                // const __half tmp_p3_updated_val = lrate*(ruv*tmp_q3 - lambda*tmp_p3);
                // const __half tmp_q3_updated_val = lrate*(ruv*tmp_p3 - lambda*tmp_q3);
                // const __half tmp_p4_updated_val = lrate*(ruv*tmp_q4 - lambda*tmp_p4);
                // const __half tmp_q4_updated_val = lrate*(ruv*tmp_p4 - lambda*tmp_q4);

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    // float norm_p = __powf(tmp_p1_updated_val_f,2) + __powf(tmp_p2_updated_val_f,2) + __powf(tmp_p3_updated_val_f,2) + __powf(tmp_p4_updated_val_f,2); // longer 
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                        item_group_sum_norms_s[item_group] += norm_q;
                    }

                    if (user_group >= uncached_user_num) {
                        user_group_base = (user_group - uncached_user_num) * k;
                        // user_group_sum_updated_val_s[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);

                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);                        
                    } 
                    else {
                        user_group_base = user_group * k;
                        // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                        
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    }

                    if (item_group >= uncached_item_num) {
                        item_group_base = (item_group - uncached_item_num) * k;
                        // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    else {
                        item_group_base = item_group * k;
                        // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                        
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const __half tmp_q1 = __float2half_rn(tmp_q1_f);
                const __half tmp_q2 = __float2half_rn(tmp_q2_f);
                const __half tmp_q3 = __float2half_rn(tmp_q3_f);
                const __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_updated_val);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_updated_val);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_updated_val);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_updated_val);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);  

                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                
                if (processed_cnt >= first_sample_rating_idx){
                    // unsigned int user_group_base = user_group * k;
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += (((norm_p)));
                    }

                    if (user_group >= uncached_user_num) {
                        user_group_base = (user_group - uncached_user_num) * k;
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);                        
                    } 
                    else {
                        user_group_base = user_group * k;
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);
                    }
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const __half tmp_p1 = __float2half_rn(tmp_p1_f);
                const __half tmp_p2 = __float2half_rn(tmp_p2_f);
                const __half tmp_p3 = __float2half_rn(tmp_p3_f);
                const __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);    

                // tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*tmp_q1_updated_val;
                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*tmp_q2_updated_val;
                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*tmp_q3_updated_val;
                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*tmp_q4_updated_val;    

                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    // unsigned int item_group_base = item_group * k;
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        item_group_sum_norms_s[item_group] += (((norm_q)));
                    }

                    if (item_group >= uncached_item_num) {
                        item_group_base = (item_group - uncached_item_num) * k;
                        // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    else {
                        item_group_base = item_group * k;
                        // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                        
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }

            processed_cnt += 1;
        }
    }

    __syncthreads();

    int local_wid = threadIdx.x/32;
    int local_w_num = blockDim.x/32;
    int lane_id = threadIdx.x%32;

    // for (; local_wid < user_group_num; local_wid += local_w_num){
    //     int base_user_group = local_wid * k;
    //     if (local_wid >= uncached_user_num){
    //         int converted_user_group = (local_wid - uncached_user_num) * k;
    //         grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val_s[converted_user_group + lane_id];
    //         grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val_s[converted_user_group + lane_id + 32];
    //         grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val_s[converted_user_group + lane_id + 64];
    //         grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val_s[converted_user_group + lane_id + 96];
    //     }
    //     // else{
    //     //     // grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val[base_user_group + lane_id];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val[base_user_group + lane_id + 32];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val[base_user_group + lane_id + 64];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val[base_user_group + lane_id + 96];

    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id, user_group_sum_updated_val[base_user_group + lane_id]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 32, user_group_sum_updated_val[base_user_group + lane_id + 32]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 64, user_group_sum_updated_val[base_user_group + lane_id + 64]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 96, user_group_sum_updated_val[base_user_group + lane_id + 96]);
    //     // }
    // }

    // local_wid = threadIdx.x/32;
    
    // for (; local_wid < item_group_num; local_wid += local_w_num){
    //     int base_item_group = local_wid * k;
    //     if (local_wid >= uncached_item_num){
    //         int converted_item_group = (local_wid - uncached_item_num) * k;
    //         grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val_s[converted_item_group + lane_id];
    //         grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val_s[converted_item_group + lane_id + 32];
    //         grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val_s[converted_item_group + lane_id + 64];
    //         grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val_s[converted_item_group + lane_id + 96];
    //     }
    //     // else {
    //     //     // grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val[base_item_group + lane_id];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val[base_item_group + lane_id + 32];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val[base_item_group + lane_id + 64];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val[base_item_group + lane_id + 96];
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id, item_group_sum_updated_val[base_item_group + lane_id]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 32, item_group_sum_updated_val[base_item_group + lane_id + 32]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 64, item_group_sum_updated_val[base_item_group + lane_id + 64]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 96, item_group_sum_updated_val[base_item_group + lane_id + 96]);
    //     // }
    // }

    if (threadIdx.x < user_group_num) {
        norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    }

    if (threadIdx.x < item_group_num) {
        norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    }
}
__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    extern __shared__ float array[];

    //! Original
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_norms_s = (float*)&q_s[item_group_num];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    


    // void** p_s = (void**)array;
    // void** q_s = (void**)&p_s[user_group_num + 32 - (user_group_num%32)];
    // float* user_group_sum_norms_s = (float*)&q_s[item_group_num + 32 - (item_group_num%32)];
    // float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num + 32 - (user_group_num%32)];
    // unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num + 32 - (item_group_num%32)]; 
    // unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[(user_group_num + 1) + 32 -((user_group_num + 1)%32)];
    // unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[(item_group_num + 1) + 32 -((item_group_num + 1)%32)];
    // unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num]; 


    //! group 수 128까지 이내만 가능함
    // if (threadIdx.x < user_group_num){
    //     p_s[threadIdx.x] = p[threadIdx.x];
    //     user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
    //     user_group_sum_norms_s[threadIdx.x] = 0.f;
    //     user_group_end_idx_s[threadIdx.x+1] = user_group_end_idx[threadIdx.x];    
    // }
    // if (threadIdx.x < item_group_num){
    //     q_s[threadIdx.x] = q[threadIdx.x];
    //     item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
    //     item_group_sum_norms_s[threadIdx.x] = 0.f;
    //     item_group_end_idx_s[threadIdx.x+1] = item_group_end_idx[threadIdx.x];
    // }
    
    // if (threadIdx.x == 0){
    //     user_group_end_idx_s[0] = -1;
    //     item_group_end_idx_s[0] = -1;      
    // }

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        p_s[i] = p[i];
        user_group_prec_s[i] = user_group_prec_info[i];
        user_group_sum_norms_s[i] = 0.f;
        user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        q_s[i] = q[i];
        item_group_prec_s[i] = item_group_prec_info[i];
        item_group_sum_norms_s[i] = 0.f;
        item_group_end_idx_s[i+1] = item_group_end_idx[i];
    }

    if (threadIdx.x == 0){
        user_group_end_idx_s[0] = -1;
        item_group_end_idx_s[0] = -1;      
    }

    __syncthreads();

    unsigned int processed_cnt = 0;
    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = 0;
            int item_group = 0;

            // for (int t = 0; t < user_group_num; t+=32){
            //     int val = 0;
            //     if (t + lane_id < user_group_num){
            //         int from = user_group_end_idx_s[t+lane_id];
            //         int to = user_group_end_idx_s[t+lane_id+1];
            //         val = (u > from && u <= to) * (t + lane_id + 1);
            //     }
            //     unsigned bitpack = __ballot_sync(0xffffffff, val);
            //     if (bitpack != 0){
            //         user_group = __shfl_sync(0xffffffff,val-1,__ffs(bitpack)-1);
            //         break;
            //     }
            // }
            //! 더 좋음
            for (int t = user_group_num;; t-=32){
                int val = 0;
                if (t - lane_id - 1 > -1){
                    int from = user_group_end_idx_s[t-lane_id-1];
                    int to = user_group_end_idx_s[t-lane_id];
                    val = (u > from && u <= to) * (t - lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                    break;
                }
            }
            // for (int t = (user_group_num + 1) + 32 -((user_group_num + 1)%32);; t-=32){
            //     int val = 0;
            //     if (t - lane_id - 1 > -1 && t-lane_id <= user_group_num){
            //         int from = user_group_end_idx_s[t-lane_id-1];
            //         int to = user_group_end_idx_s[t-lane_id];
            //         val = (u > from && u <= to) * (t - lane_id);
            //     }
            //     unsigned bitpack = __ballot_sync(0xffffffff, val);
            //     if (bitpack != 0){
            //         user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
            //         break;
            //     }
            // }
            if (user_group != 0) u = u-user_group_end_idx_s[user_group]-1;

            // for (int t = 0; t < item_group_num; t+=32){
            //     int val = 0;
            //     if (t + lane_id < item_group_num){
            //         int from = item_group_end_idx_s[t+lane_id];
            //         int to = item_group_end_idx_s[t+lane_id+1];
            //         val = (v > from && v <= to) * (t + lane_id + 1);
            //     }
            //     unsigned bitpack = __ballot_sync(0xffffffff, val);
            //     if (bitpack != 0){
            //         item_group = __shfl_sync(0xffffffff,val-1,__ffs(bitpack)-1);
            //         break;
            //     }
            // }
            //! 더 좋음
            for (int t = item_group_num;; t-=32){
                int val = 0;
                if (t - lane_id -1 > -1){
                    int from = item_group_end_idx_s[t-lane_id-1];
                    int to = item_group_end_idx_s[t-lane_id];
                    val = (v > from && v <= to) * (t - lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                    break;
                }
            }            
            // for (int t = (item_group_num + 1) + 32 -((item_group_num + 1)%32);; t-=32){
            //     int val = 0;
            //     if (t - lane_id -1 > -1 && t-lane_id <= item_group_num){
            //         int from = item_group_end_idx_s[t-lane_id-1];
            //         int to = item_group_end_idx_s[t-lane_id];
            //         val = (v > from && v <= to) * (t - lane_id);
            //     }
            //     unsigned bitpack = __ballot_sync(0xffffffff, val);
            //     if (bitpack != 0){
            //         item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
            //         break;
            //     }
            // }            
            if (item_group != 0) v = v-item_group_end_idx_s[item_group]-1;

            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                        item_group_sum_norms_s[item_group] += norm_q;
                    }

                    user_group_base = user_group * k;
                    // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;
                    // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                    }

                    user_group_base = user_group * k;
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val);
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        item_group_sum_norms_s[item_group] += (((norm_q)));
                    }

                    item_group_base = item_group * k;
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val);

                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
            processed_cnt += 1;
        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        norm_sum_p[gridDim.x * i + blockIdx.x] = user_group_sum_norms_s[i];
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        norm_sum_q[gridDim.x * i + blockIdx.x] = item_group_sum_norms_s[i];
    }

    //! Group 128이하에서 가능한 코드
    // if (threadIdx.x < user_group_num) {
    //     norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    // }

    // if (threadIdx.x < item_group_num) {
    //     norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    // }
}

// __global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_reg_cache(
//                             const Node *R,
//                             unsigned int nnz,
//                             void** p,
//                             void** q,
//                             curandState *state,
//                             float lrate_f,
//                             int k,
//                             int num_iters,
//                             int current_iter,
//                             int update_count_this_block,
//                             int update_vector_size,
//                             float lambda_f,
//                             unsigned int* user_group_end_idx,
//                             unsigned int* item_group_end_idx,
//                             unsigned char* user_group_prec_info,
//                             unsigned char* item_group_prec_info,
//                             float* grad_sum_norm_p,
//                             float* grad_sum_norm_q,
//                             float* norm_sum_p,
//                             float* norm_sum_q,
//                             float* user_group_sum_updated_val,
//                             float* item_group_sum_updated_val,
//                             unsigned int first_sample_rating_idx,
//                             unsigned int user_group_num,
//                             unsigned int item_group_num
//                             )
// {    
//     extern __shared__ float array[];
//     // int reg_user[5]={-1,-1,-1,-1,-1};
//     // int reg_item[5]={-1,-1,-1,-1,-1};
//     int reg_user = -1;
//     int reg_item = -1;
//     int reg_user2 = -1;
//     int reg_item2 = -1;
//     //! Original
//     void** p_s = (void**)array;
//     void** q_s = (void**)&p_s[user_group_num];
//     float* user_group_sum_norms_s = (float*)&q_s[item_group_num];
//     float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
//     unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num]; 
//     unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
//     unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
//     unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    
 

//     // void** p_s = (void**)array;
//     // void** q_s = (void**)&p_s[user_group_num + 32 - (user_group_num%32)];
//     // float* user_group_sum_norms_s = (float*)&q_s[item_group_num + 32 - (item_group_num%32)];
//     // float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num + 32 - (user_group_num%32)];
//     // unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num + 32 - (item_group_num%32)]; 
//     // unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[(user_group_num + 1) + 32 -((user_group_num + 1)%32)];
//     // unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[(item_group_num + 1) + 32 -((item_group_num + 1)%32)];
//     // unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num]; 

//     //! group 수 128까지 이내만 가능함
//     // if (threadIdx.x < user_group_num){
//     //     p_s[threadIdx.x] = p[threadIdx.x];
//     //     user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
//     //     user_group_sum_norms_s[threadIdx.x] = 0.f;
//     //     user_group_end_idx_s[threadIdx.x+1] = user_group_end_idx[threadIdx.x];    
//     // }
//     // if (threadIdx.x < item_group_num){
//     //     q_s[threadIdx.x] = q[threadIdx.x];
//     //     item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
//     //     item_group_sum_norms_s[threadIdx.x] = 0.f;
//     //     item_group_end_idx_s[threadIdx.x+1] = item_group_end_idx[threadIdx.x];
//     // }
//     int lane_id = threadIdx.x%32;

//     if (threadIdx.x == 0){
//         user_group_end_idx_s[0] = -1;
//         item_group_end_idx_s[0] = -1;      
//     }
//     int user_reg_cnt = 0;

//     int item_reg_cnt = 0;
//     // #pragma unroll
//     // for (int i = 32 - 1 - lane_id; i > -1; i-=31){
//     //     reg_item[item_reg_cnt] = item_group_end_idx[i]; 
//     //     item_reg_cnt+=1;
//     // }
    
//     // user_group_end_idx_s[0] = -1;
//     // item_group_end_idx_s[0] = -1;      


//     for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
//         p_s[i] = p[i];
//         user_group_prec_s[i] = user_group_prec_info[i];
//         user_group_sum_norms_s[i] = 0.f;
//         // user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
//         user_group_end_idx_s[i] = user_group_end_idx[i]; 
//     }

//     for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
//         q_s[i] = q[i];
//         item_group_prec_s[i] = item_group_prec_info[i];
//         item_group_sum_norms_s[i] = 0.f;
//         // item_group_end_idx_s[i+1] = item_group_end_idx[i];
//         item_group_end_idx_s[i] = item_group_end_idx[i];

//     }

//     if (threadIdx.x == 0){
//         // user_group_end_idx_s[0] = -1;
//         // item_group_end_idx_s[0] = -1;      
//     }

//     __syncthreads();
//     // #pragma unroll
//     // for (int i = 52 - 1 - lane_id; i > -1; i-=31){
//     //     if (i > 52 - 1  - 32)
//     //         reg_user = user_group_end_idx_s[i]; 
//     //     else
//     //         reg_user2 =user_group_end_idx_s[i];
//     //     user_reg_cnt +=1;
//     // }

//     // #pragma unroll
//     // for (int i = 55 - 1 - lane_id; i > -1; i-=31){
//     //     if (i > 55 - 1 - 32)
//     //         reg_item = item_group_end_idx_s[i]; 
//     //     else
//     //         reg_item2 = item_group_end_idx_s[i];
//     //     item_reg_cnt +=1;
//     // }

//     if (31-1-lane_id > -1)reg_user = user_group_end_idx_s[31-1-lane_id];
//     if (31-1-lane_id-31 > -1) reg_user2 = user_group_end_idx_s[31-1-lane_id-31];

//     if (51-1-lane_id > -1) reg_item = item_group_end_idx_s[51-1-lane_id];
//     if (51-1-lane_id-31 > -1)  reg_item2 = item_group_end_idx_s[51-1-lane_id-31];

//     unsigned int processed_cnt = 0;
//     __half lrate = __float2half(lrate_f);
//     __half lambda = __float2half(lambda_f);

//     for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
//     {
//         // int lane_id = threadIdx.x%32;
//         int local_wid = threadIdx.x/32;
//         int local_w_num = blockDim.x/32;
//         int wid = local_w_num*blockIdx.x + local_wid;  
        
//         unsigned int start_id = 0;
//         if(lane_id == 0)
//         {
//             unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
//             start_id = origin%nnz;
//         }

//         start_id = __shfl_sync(0xffffffff,start_id, 0);
        
//         for(int i = 0;i < update_vector_size;i++)
//         {
//             int offset = (start_id + i)%nnz;

//             float r = __ldg(&R[offset].r);
//             int u = __ldg(&R[offset].u);
//             int v = __ldg(&R[offset].i);
            
//             int user_group = -1;
//             int item_group = -1;

//             // for (int t = 0; t < user_group_num; t+=32){
//             //     int val = 0;
//             //     if (t + lane_id < user_group_num){
//             //         int from = user_group_end_idx_s[t+lane_id];
//             //         int to = user_group_end_idx_s[t+lane_id+1];
//             //         val = (u > from && u <= to) * (t + lane_id + 1);
//             //     }
//             //     unsigned bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff,val-1,__ffs(bitpack)-1);
//             //         break;
//             //     }
//             // }

//             //! 더 좋음
//             // #pragma unroll
//             // for (int t = user_group_num;; t-=32){
//             //     int val = 0;
//             //     if (t - lane_id - 1 > -1){
//             //         int from = user_group_end_idx_s[t-lane_id-1];
//             //         int to = user_group_end_idx_s[t-lane_id];
//             //         val = (u > from && u <= to) * (t - lane_id);
//             //     }
//             //     unsigned bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
//             //         break;
//             //     }
//             // }

//             int to = reg_user;
//             int from = __shfl_down_sync(0xffffffff, to, 1);
//             int val = (u > from && u <= to);
//             unsigned bitpack = __ballot_sync(0xffffffff, val);
//             if (bitpack != 0){
//                 user_group = __shfl_sync(0xffffffff, user_group_num - 1 - lane_id , __ffs(bitpack)-1);
//             }  

//             if (user_group == -1){
//                 to = reg_user2;
//                 from = __shfl_down_sync(0xffffffff, to, 1);
//                 val = (u > from && u <= to);
//                 bitpack = __ballot_sync(0xffffffff, val);
//                 if (bitpack != 0){
//                     user_group = __shfl_sync(0xffffffff, user_group_num -1 - 31 - lane_id , __ffs(bitpack)-1);
//                 }  
//             }

//             to = reg_item;
//             from = __shfl_down_sync(0xffffffff, to, 1);
//             val = (v > from && v <= to);
//             bitpack = __ballot_sync(0xffffffff, val);
//             if (bitpack != 0){
//                 item_group = __shfl_sync(0xffffffff, item_group_num - 1 - lane_id , __ffs(bitpack)-1);
//             }

//             if (item_group == -1){
//                 to = reg_item2;
//                 from = __shfl_down_sync(0xffffffff, to, 1);
//                 val = (v > from && v <= to);
//                 bitpack = __ballot_sync(0xffffffff, val);
//                 if (bitpack != 0){
//                     item_group = __shfl_sync(0xffffffff, item_group_num- 1 - 31 - lane_id , __ffs(bitpack)-1);
//                 }   
//             }

//             // if (bitpack == 0){
//             //     to = reg_user[1];
//             //     from = __shfl_down_sync(0xffffffff, to, 1);
//             //     val = (u > from && u <= to);
//             //     bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff, user_group_num - 1 -31 - lane_id , __ffs(bitpack)-1);
//             //         // printf("%d " , user_group);

//             //         // user_group = user_group_num - (user_group * 32) - lane_id;
//             //         // break;
//             //     }  
//             // }
            
//             // if (bitpack == 0){
//             //     to = reg_user[2];
//             //     from = __shfl_down_sync(0xffffffff, to, 1);
//             //     val = (u > from && u <= to);
//             //     bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff, user_group_num - 1 - 62 - lane_id , __ffs(bitpack)-1);
//             //         // printf("%d " , user_group);

//             //         // user_group = user_group_num - (user_group * 32) - lane_id;
//             //         // break;
//             //     }  
//             // }

//             // if (USER_GROUP_NUM > 93){
//             //     to = reg_user[3];
//             //     from = __shfl_down_sync(0xffffffff, to, 1);
//             //     val = (u > from && u <= to);
//             //     bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff, user_group_num - 1 -93 - lane_id , __ffs(bitpack)-1);
//             //         // printf("%d " , user_group);

//             //         // user_group = user_group_num - (user_group * 32) - lane_id;
//             //         // break;
//             //     }  
//             // }
//             // if (USER_GROUP_NUM > 122){
//             //     to = reg_user[4];
//             //     from = __shfl_down_sync(0xffffffff, to, 1);
//             //     val = (u > from && u <= to);
//             //     bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff, user_group_num - 1 - lane_id , __ffs(bitpack)-1);
//             //         // printf("%d " , user_group);

//             //         // user_group = user_group_num - (user_group * 32) - lane_id;
//             //         // break;
//             //     }  
//             // }
//             // for (int i = 0;; i++){
//             //     int to = reg_user[i];
//             //     int from = __shfl_down_sync(0xffffffff, to, 1);
//             //     int val = (u > from && u <= to) * (i+1);
//             //     unsigned bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff,user_group_num - 1 - ((val-1) * 31) - lane_id , __ffs(bitpack)-1);
//             //         // printf("%d " , user_group);

//             //         // user_group = user_group_num - (user_group * 32) - lane_id;
//             //         break;
//             //     }  
//             // }

//             // for (int t = (user_group_num + 1) + 32 -((user_group_num + 1)%32);; t-=32){
//             //     int val = 0;
//             //     if (t - lane_id - 1 > -1 && t-lane_id <= user_group_num){
//             //         int from = user_group_end_idx_s[t-lane_id-1];
//             //         int to = user_group_end_idx_s[t-lane_id];
//             //         val = (u > from && u <= to) * (t - lane_id);
//             //     }
//             //     unsigned bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
//             //         break;
//             //     }
//             // }
//             if (user_group != 0) u = u-user_group_end_idx_s[user_group-1]-1;

//             // for (int t = 0; t < item_group_num; t+=32){
//             //     int val = 0;
//             //     if (t + lane_id < item_group_num){
//             //         int from = item_group_end_idx_s[t+lane_id];
//             //         int to = item_group_end_idx_s[t+lane_id+1];
//             //         val = (v > from && v <= to) * (t + lane_id + 1);
//             //     }
//             //     unsigned bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         item_group = __shfl_sync(0xffffffff,val-1,__ffs(bitpack)-1);
//             //         break;
//             //     }
//             // }
//             //! 더 좋음
//             // #pragma unroll
//             // for (int t = item_group_num;; t-=32){
//             //     int val = 0;
//             //     if (t - lane_id -1 > -1){
//             //         int from = item_group_end_idx_s[t-lane_id-1];
//             //         int to = item_group_end_idx_s[t-lane_id];
//             //         val = (v > from && v <= to) * (t - lane_id);
//             //     }
//             //     unsigned bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
//             //         break;
//             //     }
//             // }       
                 
//             // for (int t = (item_group_num + 1) + 32 -((item_group_num + 1)%32);; t-=32){
//             //     int val = 0;
//             //     if (t - lane_id -1 > -1 && t-lane_id <= item_group_num){
//             //         int from = item_group_end_idx_s[t-lane_id-1];
//             //         int to = item_group_end_idx_s[t-lane_id];
//             //         val = (v > from && v <= to) * (t - lane_id);
//             //     }
//             //     unsigned bitpack = __ballot_sync(0xffffffff, val);
//             //     if (bitpack != 0){
//             //         item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
//             //         break;
//             //     }
//             // }            
//             if (item_group != 0) v = v-item_group_end_idx_s[item_group-1]-1;

//             int base_p = u*k;
//             int base_q = v*k;

//             unsigned char user_prec = user_group_prec_s[user_group];
//             unsigned char item_prec = item_group_prec_s[item_group];

//             //! both precisions are half
//             if (!user_prec && !item_prec){
//                 __half r_h = __float2half(r);
//                 __half* tmp_p_ptr = (__half*)p_s[user_group];
//                 __half* tmp_q_ptr = (__half*)q_s[item_group];

//                 const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 const __half ruv = r_h - tmp_product;

//                 const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
//                 const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
//                 const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
//                 const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
//                 const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
//                 const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
//                 const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
//                 const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

//                 tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
//                 tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
//                 tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
//                 tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
//                 const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
//                 const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
//                 const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
//                 const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
//                 const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
//                 const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
//                 const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
//                 const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

//                 if (processed_cnt >= first_sample_rating_idx){
//                     unsigned int user_group_base;
//                     unsigned int item_group_base;

//                     float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
//                     float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

//                     if (lane_id == 0){
//                         user_group_sum_norms_s[user_group] += norm_p;
//                         item_group_sum_norms_s[item_group] += norm_q;
//                     }

//                     user_group_base = user_group * k;
//                     // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
//                     // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
//                     // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
//                     // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

//                     item_group_base = item_group * k;
//                     // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
//                     // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
//                     // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
//                     // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
//                 }        
//             }
//             //! user half item single
//             else if (!user_prec && item_prec){
//                 __half* tmp_p_ptr = (__half*)p_s[user_group];
//                 float* tmp_q_ptr = (float*)q_s[item_group];

//                 const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
//                 const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
//                 const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
//                 const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

//                 const float tmp_p1_f = __half2float(tmp_p1);
//                 const float tmp_p2_f = __half2float(tmp_p2);
//                 const float tmp_p3_f = __half2float(tmp_p3);
//                 const float tmp_p4_f = __half2float(tmp_p4);

//                 float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 const float ruv = r - tmp_product;

//                 const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
//                 const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
//                 const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
//                 const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

//                 tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
//                 tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
//                 tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
//                 tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
//                 tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
//                 tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
//                 tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
//                 tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     unsigned int user_group_base;
//                     float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
                    
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

//                     if (lane_id == 0){
//                         user_group_sum_norms_s[user_group] += norm_p;
//                     }

//                     user_group_base = user_group * k;
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val);
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val);
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val);
//                     atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val);
//                 }        
//             }
//             //! user single item half
//             else if (user_prec && !item_prec){
//                 float* tmp_p_ptr = (float*)p_s[user_group];
//                 __half* tmp_q_ptr = (__half*)q_s[item_group];

//                 const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
//                 const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
//                 const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
//                 const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
//                 const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
//                 const float tmp_q1_f = __half2float(tmp_q1);
//                 const float tmp_q2_f = __half2float(tmp_q2);
//                 const float tmp_q3_f = __half2float(tmp_q3);
//                 const float tmp_q4_f = __half2float(tmp_q4);

//                 float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 const float ruv = r - tmp_product;

//                 const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
//                 const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
//                 const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
//                 const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
//                 tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
//                 tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
//                 tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
//                 tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
//                 tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
//                 tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
//                 tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
//                 tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));

//                 if (processed_cnt >= first_sample_rating_idx){
//                     unsigned int item_group_base;
//                     float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                    
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

//                     if (lane_id == 0){
//                         item_group_sum_norms_s[item_group] += (((norm_q)));
//                     }

//                     item_group_base = item_group * k;
                    
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val);
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val);
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val);
//                     atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val);

//                     // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
//                     // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
//                     // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
//                     // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
//                 }     
//             }
//             //! user single item single
//             else{
//                 float* tmp_p_ptr = (float*)p_s[user_group];
//                 float* tmp_q_ptr = (float*)q_s[item_group];

//                 float tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 float tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 float ruv = r - tmp_product;

//                 // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
//                 // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

//                 // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
//                 // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

//                 // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
//                 // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

//                 // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
//                 // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
                
//                 tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

//                 tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

//                 tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

//                 tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
//             }
//             processed_cnt += 1;
//         }
//     }

//     __syncthreads();

//     for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
//         norm_sum_p[gridDim.x * i + blockIdx.x] = user_group_sum_norms_s[i];
//     }

//     for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
//         norm_sum_q[gridDim.x * i + blockIdx.x] = item_group_sum_norms_s[i];
//     }

//     //! Group 128이하에서 가능한 코드
//     // if (threadIdx.x < user_group_num) {
//     //     norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
//     // }

//     // if (threadIdx.x < item_group_num) {
//     //     norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
//     // }
// }

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_reg_cache(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            int user_group_num,
                            int item_group_num
                            )
{    
    extern __shared__ float array[];
    int reg_user=-1;
    int reg_item=-1;
    // int reg_user2=-1;
    // int reg_item2=-1;

    //! Original
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_norms_s = (float*)&q_s[item_group_num];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    
 
    int lane_id = threadIdx.x%32;

    if (threadIdx.x == 0){
        user_group_end_idx_s[0] = -1;
        item_group_end_idx_s[0] = -1;      
    }      

    if (user_group_num-1-lane_id > -1) reg_user = user_group_end_idx[user_group_num-1-lane_id];
    if (item_group_num-1-lane_id > -1) reg_item = item_group_end_idx[item_group_num-1-lane_id];

    for (int i = threadIdx.x; i <= user_group_num-32; i+= blockDim.x){
        user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }  

    for (int i = threadIdx.x; i <= item_group_num-32; i+= blockDim.x){
        item_group_end_idx_s[i+1] = item_group_end_idx[i]; 
    }   


    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        p_s[i] = p[i];
        user_group_prec_s[i] = user_group_prec_info[i];
        user_group_sum_norms_s[i] = 0.f;
        // user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        q_s[i] = q[i];
        item_group_prec_s[i] = item_group_prec_info[i];
        item_group_sum_norms_s[i] = 0.f;
        // item_group_end_idx_s[i+1] = item_group_end_idx[i];
    }

    if (threadIdx.x == 0){
        user_group_end_idx_s[0] = -1;
        item_group_end_idx_s[0] = -1;      
    }

    __syncthreads();

    // printf("lane id : %d , val : %d ", lane_id, user_group_num-1-lane_id);
    // if (user_group_num-1-lane_id > -1) reg_user = user_group_end_idx[user_group_num-1-lane_id];
    // if (user_group_num-1-lane_id-31 > -1) reg_user2 = user_group_end_idx[user_group_num-1-lane_id-31];
    // if (user_group_num-1-lane_id-62 > -1) reg_user3 = user_group_end_idx[user_group_num-1-lane_id-62];
    // if (user_group_num-1-lane_id-93 > -1) reg_user4 = user_group_end_idx[user_group_num-1-lane_id-93];
    // if (user_group_num-1-lane_id-124 > -1) reg_user5 = user_group_end_idx[user_group_num-1-lane_id-124];
    // if (user_group_num-1-lane_id-155 > -1) reg_user6 = user_group_end_idx[user_group_num-1-lane_id-155];
    // if (user_group_num-1-lane_id-186 > -1) reg_user7 = user_group_end_idx[user_group_num-1-lane_id-186];
    
    // if (item_group_num-1-lane_id > -1) reg_item = item_group_end_idx[item_group_num-1-lane_id];
    // if (item_group_num-1-lane_id-31 > -1) reg_item2 = item_group_end_idx[item_group_num-1-lane_id-31];
    // if (item_group_num-1-lane_id-62 > -1) reg_item3 = item_group_end_idx[item_group_num-1-lane_id-62];
    // if (item_group_num-1-lane_id-93 > -1) reg_item4 = item_group_end_idx[item_group_num-1-lane_id-93];
    // if (item_group_num-1-lane_id-124 > -1) reg_item5 = item_group_end_idx[item_group_num-1-lane_id-124];
    // if (item_group_num-1-lane_id-155 > -1) reg_item6 = item_group_end_idx[item_group_num-1-lane_id-155];
    // if (item_group_num-1-lane_id-186 > -1) reg_item7 = item_group_end_idx[item_group_num-1-lane_id-186];

    unsigned int processed_cnt = 0;
    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = -1;
            int item_group = -1;

            int to = reg_user;
            int from = __shfl_down_sync(0xffffffff, to, 1);
            int val = (u > from && u <= to);
            unsigned bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                user_group = __shfl_sync(0xffffffff, user_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (user_group != 0) u = u-from-1;

            }else{
                for (int t = user_group_num-31;; t-=32){
                    val = 0;
                    if (t - lane_id - 1 > -1){
                        from = user_group_end_idx_s[t-lane_id-1];
                        to = user_group_end_idx_s[t-lane_id];
                        val = (u > from && u <= to) * (t - lane_id);
                    }
                    bitpack = __ballot_sync(0xffffffff, val);
                    if (bitpack != 0){
                        user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                        if (user_group != 0) u = u-user_group_end_idx_s[user_group]-1;
                        break;
                    }
                }
            }

            to = reg_item;
            from = __shfl_down_sync(0xffffffff, to, 1);
            val = (v > from && v <= to);
            bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                item_group = __shfl_sync(0xffffffff, item_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (item_group != 0) v = v-from-1;
            }else{
                //! 더 좋음
                for (int t = item_group_num-31;; t-=32){
                    val = 0;
                    if (t - lane_id -1 > -1){
                        from = item_group_end_idx_s[t-lane_id-1];
                        to = item_group_end_idx_s[t-lane_id];
                        val = (v > from && v <= to) * (t - lane_id);
                    }
                    bitpack = __ballot_sync(0xffffffff, val);
                    if (bitpack != 0){
                        item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                        if (item_group != 0) v = v-item_group_end_idx_s[item_group]-1;
                        break;
                    }
                }   
            }       

            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                        item_group_sum_norms_s[item_group] += norm_q;
                    }

                    user_group_base = user_group * k;
                    // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;
                    // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                    }

                    user_group_base = user_group * k;
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val);
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        item_group_sum_norms_s[item_group] += (((norm_q)));
                    }

                    item_group_base = item_group * k;
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val);

                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
            processed_cnt += 1;
// #endif

        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        norm_sum_p[gridDim.x * i + blockIdx.x] = user_group_sum_norms_s[i];
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        norm_sum_q[gridDim.x * i + blockIdx.x] = item_group_sum_norms_s[i];
    }

    //! Group 128이하에서 가능한 코드
    // if (threadIdx.x < user_group_num) {
    //     norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    // }

    // if (threadIdx.x < item_group_num) {
    //     norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    // }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_64reg_cache(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            int user_group_num,
                            int item_group_num
                            )
{    
    extern __shared__ float array[];
    int reg_user=-1;
    int reg_item=-1;
    int reg_user1=-1;
    int reg_item1=-1;

    //! Original
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_norms_s = (float*)&q_s[item_group_num];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = user_group_end_idx_s;
    if (user_group_num > 62) {
        item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num - 61];
        if (threadIdx.x == 0) user_group_end_idx_s[0] = -1;
    }
    // unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)item_group_end_idx_s;
    if (item_group_num > 62) {
        user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num - 61];
        if (threadIdx.x == 0) item_group_end_idx_s[0] = -1;
    }
    // unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    
 
    int lane_id = threadIdx.x%32;

    if (user_group_num-1-lane_id > -1) reg_user = user_group_end_idx[user_group_num-1-lane_id];
    if (item_group_num-1-lane_id > -1) reg_item = item_group_end_idx[item_group_num-1-lane_id];
    if (user_group_num-32-lane_id > -1) reg_user1 = user_group_end_idx[user_group_num-32-lane_id];
    if (item_group_num-32-lane_id > -1) reg_item1 = item_group_end_idx[item_group_num-32-lane_id];
    
    for (int i = threadIdx.x; i <= user_group_num-63; i+= blockDim.x){
        user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }  

    for (int i = threadIdx.x; i <= item_group_num-63; i+= blockDim.x){
        item_group_end_idx_s[i+1] = item_group_end_idx[i]; 
    }   

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        p_s[i] = p[i];
        user_group_prec_s[i] = user_group_prec_info[i];
        user_group_sum_norms_s[i] = 0.f;
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        q_s[i] = q[i];
        item_group_prec_s[i] = item_group_prec_info[i];
        item_group_sum_norms_s[i] = 0.f;
    }

    __syncthreads();

    unsigned int processed_cnt = 0;
    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = -1;
            int item_group = -1;

            int to = reg_user;
            int from = __shfl_down_sync(0xffffffff, to, 1);
            int val = (u > from && u <= to);
            unsigned bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                user_group = __shfl_sync(0xffffffff, user_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (user_group != 0) u = u-from-1;

            }else{
                to = reg_user1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (u > from && u <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    user_group = __shfl_sync(0xffffffff, user_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (user_group != 0) u = u-from-1;
                }else{
                    for (int t = user_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id - 1 > -1){
                            from = user_group_end_idx_s[t-lane_id-1];
                            to = user_group_end_idx_s[t-lane_id];
                            val = (u > from && u <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (user_group != 0) u = u-user_group_end_idx_s[user_group]-1;
                            break;
                        }
                    }
                }
            }

            to = reg_item;
            from = __shfl_down_sync(0xffffffff, to, 1);
            val = (v > from && v <= to);
            bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                item_group = __shfl_sync(0xffffffff, item_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (item_group != 0) v = v-from-1;
            }else{
                to = reg_item1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (v > from && v <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    item_group = __shfl_sync(0xffffffff, item_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (item_group != 0) v = v-from-1;
                }else{
                    //! 더 좋음
                    for (int t = item_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id -1 > -1){
                            from = item_group_end_idx_s[t-lane_id-1];
                            to = item_group_end_idx_s[t-lane_id];
                            val = (v > from && v <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (item_group != 0) v = v-item_group_end_idx_s[item_group]-1;
                            break;
                        }
                    }   
                }
            }       

            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                        item_group_sum_norms_s[item_group] += norm_q;
                    }

                    user_group_base = user_group * k;
                    // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;
                    // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                    }

                    user_group_base = user_group * k;
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val);
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        item_group_sum_norms_s[item_group] += (((norm_q)));
                    }

                    item_group_base = item_group * k;
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val);

                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
            processed_cnt += 1;
// #endif

        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        norm_sum_p[gridDim.x * i + blockIdx.x] = user_group_sum_norms_s[i];
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        norm_sum_q[gridDim.x * i + blockIdx.x] = item_group_sum_norms_s[i];
    }

    //! Group 128이하에서 가능한 코드
    // if (threadIdx.x < user_group_num) {
    //     norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    // }

    // if (threadIdx.x < item_group_num) {
    //     norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    // }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_info_on_device_mem(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            int user_group_num,
                            int item_group_num
                            )
{    
    unsigned int processed_cnt = 0;
    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        norm_sum_p[gridDim.x * i + blockIdx.x] = 0.0f;
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        norm_sum_q[gridDim.x * i + blockIdx.x] = 0.0f;
    }

    __syncthreads();

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            //! Search 
            int user_group = 0;
            int item_group = 0;

            for (int t = 0; t < user_group_num; t+=32){
                int val = 0;
                if (t + lane_id < user_group_num){
                    int from = user_group_end_idx[t+lane_id];
                    int to = user_group_end_idx[t+lane_id+1];
                    val = (u > from && u <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    user_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    if (user_group != 0) u = u-user_group_end_idx[user_group]-1;
                    break;
                }
            }

            for (int t = 0; t < item_group_num; t+=32){
                int val = 0;
                if (t + lane_id < item_group_num){
                    int from = item_group_end_idx[t+lane_id];
                    int to = item_group_end_idx[t+lane_id+1];
                    val = (v > from && v <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    item_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    if (item_group != 0) v = v-item_group_end_idx[item_group]-1;
                    break;
                }
            }
            
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_info[user_group];
            unsigned char item_prec = item_group_prec_info[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p[user_group];
                __half* tmp_q_ptr = (__half*)q[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        norm_sum_p[gridDim.x * user_group + blockIdx.x] += norm_p;
                        norm_sum_q[gridDim.x * item_group + blockIdx.x] += norm_q;
                    }

                    user_group_base = user_group * k;
                    // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;
                    // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p[user_group];
                float* tmp_q_ptr = (float*)q[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        norm_sum_p[gridDim.x * user_group + blockIdx.x] += norm_p;
                    }

                    user_group_base = user_group * k;
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val);
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p[user_group];
                __half* tmp_q_ptr = (__half*)q[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        norm_sum_q[gridDim.x * item_group + blockIdx.x] += norm_q;
                    }

                    item_group_base = item_group * k;
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val);

                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p[user_group];
                float* tmp_q_ptr = (float*)q[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
            processed_cnt += 1;
// #endif
        }
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_info_on_device_mem_search_order_opt(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            int user_group_num,
                            int item_group_num
                            )
{    
    unsigned int processed_cnt = 0;
    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        norm_sum_p[gridDim.x * i + blockIdx.x] = 0.0f;
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        norm_sum_q[gridDim.x * i + blockIdx.x] = 0.0f;
    }

    __syncthreads();

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            //! Search 
            int user_group = 0;
            int item_group = 0;

            for (int t = user_group_num;; t-=32){
                int val = 0;
                if (t - lane_id -1 > -1){
                    int from = user_group_end_idx[t-lane_id-1];
                    int to = user_group_end_idx[t-lane_id];
                    val = (u > from && u <= to) * (t - lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    user_group = __shfl_sync(0xffffffff,val-1,__ffs(bitpack)-1);
                    if (user_group != 0) u = u-user_group_end_idx[user_group]-1;
                    break;
                }
            }

            for (int t = item_group_num;; t-=32){
                int val = 0;
                if (t - lane_id -1 > -1){
                    int from = item_group_end_idx[t-lane_id-1];
                    int to = item_group_end_idx[t-lane_id];
                    val = (v > from && v <= to) * (t - lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    item_group = __shfl_sync(0xffffffff,val-1,__ffs(bitpack)-1);
                    if (item_group != 0) v = v-item_group_end_idx[item_group]-1;
                    break;
                }
            }
            
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_info[user_group];
            unsigned char item_prec = item_group_prec_info[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p[user_group];
                __half* tmp_q_ptr = (__half*)q[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        norm_sum_p[gridDim.x * user_group + blockIdx.x] += norm_p;
                        norm_sum_q[gridDim.x * item_group + blockIdx.x] += norm_q;
                    }

                    user_group_base = user_group * k;
                    // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;
                    // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p[user_group];
                float* tmp_q_ptr = (float*)q[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        norm_sum_p[gridDim.x * user_group + blockIdx.x] += norm_p;
                    }

                    user_group_base = user_group * k;
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val);
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p[user_group];
                __half* tmp_q_ptr = (__half*)q[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        norm_sum_q[gridDim.x * item_group + blockIdx.x] += norm_q;
                    }

                    item_group_base = item_group * k;
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val);

                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p[user_group];
                float* tmp_q_ptr = (float*)q[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
            processed_cnt += 1;
// #endif
        }
    }
}


__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_64reg_cache_eval_indexing(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            int user_group_num,
                            int item_group_num
                            )
{    
    extern __shared__ float array[];
    int reg_user=-1;
    int reg_item=-1;
    int reg_user1=-1;
    int reg_item1=-1;

    //! Original
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&q_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = user_group_end_idx_s;
    if (user_group_num > 62) {
        item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num - 61];
        if (threadIdx.x == 0) user_group_end_idx_s[0] = -1;
    }
    // unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)item_group_end_idx_s;
    if (item_group_num > 62) {
        user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num - 61];
        if (threadIdx.x == 0) item_group_end_idx_s[0] = -1;
    }
    // unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    
 
    int lane_id = threadIdx.x%32;

    if (user_group_num-1-lane_id > -1) reg_user = user_group_end_idx[user_group_num-1-lane_id];
    if (item_group_num-1-lane_id > -1) reg_item = item_group_end_idx[item_group_num-1-lane_id];
    if (user_group_num-32-lane_id > -1) reg_user1 = user_group_end_idx[user_group_num-32-lane_id];
    if (item_group_num-32-lane_id > -1) reg_item1 = item_group_end_idx[item_group_num-32-lane_id];
    
    for (int i = threadIdx.x; i <= user_group_num-63; i+= blockDim.x){
        user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }  

    for (int i = threadIdx.x; i <= item_group_num-63; i+= blockDim.x){
        item_group_end_idx_s[i+1] = item_group_end_idx[i]; 
    }   

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        p_s[i] = p[i];
        user_group_prec_s[i] = user_group_prec_info[i];
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        q_s[i] = q[i];
        item_group_prec_s[i] = item_group_prec_info[i];
    }

    __syncthreads();

    __half lrate = __float2half(lrate_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            // float r = __ldg(&R[offset].r);
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = -1;
            int item_group = -1;

            int to = reg_user;
            int from = __shfl_down_sync(0xffffffff, to, 1);
            int val = (u > from && u <= to);
            unsigned bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                user_group = __shfl_sync(0xffffffff, user_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (user_group != 0) u = u-from-1;

            }else{
                to = reg_user1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (u > from && u <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    user_group = __shfl_sync(0xffffffff, user_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (user_group != 0) u = u-from-1;
                }else{
                    for (int t = user_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id - 1 > -1){
                            from = user_group_end_idx_s[t-lane_id-1];
                            to = user_group_end_idx_s[t-lane_id];
                            val = (u > from && u <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (user_group != 0) u = u-user_group_end_idx_s[user_group]-1;
                            break;
                        }
                    }
                }
            }

            to = reg_item;
            from = __shfl_down_sync(0xffffffff, to, 1);
            val = (v > from && v <= to);
            bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                item_group = __shfl_sync(0xffffffff, item_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (item_group != 0) v = v-from-1;
            }else{
                to = reg_item1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (v > from && v <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    item_group = __shfl_sync(0xffffffff, item_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (item_group != 0) v = v-from-1;
                }else{
                    //! 더 좋음
                    for (int t = item_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id -1 > -1){
                            from = item_group_end_idx_s[t-lane_id-1];
                            to = item_group_end_idx_s[t-lane_id];
                            val = (v > from && v <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (item_group != 0) v = v-item_group_end_idx_s[item_group]-1;
                            break;
                        }
                    }   
                }
            }       

            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            __half r_h = __float2half(r);
            __half* tmp_p_ptr = (__half*)p_s[user_group];
            __half* tmp_q_ptr = (__half*)q_s[item_group];

            const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
            const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
            const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

            __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            const __half ruv = r_h - tmp_product;

            const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
            const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
            const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
            const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
            const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
            const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
            const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
            const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

            tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
            tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
            tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
            tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
            tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
            tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
            tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
            tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
        }
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_compute_fp32_64reg_cache_time_check_per_area(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            int user_group_num,
                            int item_group_num
                            )
{    
    extern __shared__ float array[];
    int reg_user=-1;
    int reg_item=-1;
    int reg_user1=-1;
    int reg_item1=-1;

    //! Original
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_norms_s = (float*)&q_s[item_group_num];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = user_group_end_idx_s;
    if (user_group_num > 62) {
        item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num - 61];
        if (threadIdx.x == 0) user_group_end_idx_s[0] = -1;
    }
    // unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)item_group_end_idx_s;
    if (item_group_num > 62) {
        user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num - 61];
        if (threadIdx.x == 0) item_group_end_idx_s[0] = -1;
    }
    // unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    
 
    int lane_id = threadIdx.x%32;

    if (user_group_num-1-lane_id > -1) reg_user = user_group_end_idx[user_group_num-1-lane_id];
    if (item_group_num-1-lane_id > -1) reg_item = item_group_end_idx[item_group_num-1-lane_id];
    if (user_group_num-32-lane_id > -1) reg_user1 = user_group_end_idx[user_group_num-32-lane_id];
    if (item_group_num-32-lane_id > -1) reg_item1 = item_group_end_idx[item_group_num-32-lane_id];
    
    for (int i = threadIdx.x; i <= user_group_num-63; i+= blockDim.x){
        user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }  

    for (int i = threadIdx.x; i <= item_group_num-63; i+= blockDim.x){
        item_group_end_idx_s[i+1] = item_group_end_idx[i]; 
    }   

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        p_s[i] = p[i];
        user_group_prec_s[i] = user_group_prec_info[i];
        user_group_sum_norms_s[i] = 0.f;
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        q_s[i] = q[i];
        item_group_prec_s[i] = item_group_prec_info[i];
        item_group_sum_norms_s[i] = 0.f;
    }

    __syncthreads();

    unsigned int processed_cnt = 0;
    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = -1;
            int item_group = -1;
//! Indexing area (start point)
            int to = reg_user;
            int from = __shfl_down_sync(0xffffffff, to, 1);
            int val = (u > from && u <= to);
            unsigned bitpack = __ballot_sync(0xffffffff, val);

            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                user_group = __shfl_sync(0xffffffff, user_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (user_group != 0) u = u-from-1;

            }else{
                to = reg_user1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (u > from && u <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    user_group = __shfl_sync(0xffffffff, user_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (user_group != 0) u = u-from-1;
                }else{
                    for (int t = user_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id - 1 > -1){
                            from = user_group_end_idx_s[t-lane_id-1];
                            to = user_group_end_idx_s[t-lane_id];
                            val = (u > from && u <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (user_group != 0) u = u-user_group_end_idx_s[user_group]-1;
                            break;
                        }
                    }
                }
            }

            to = reg_item;
            from = __shfl_down_sync(0xffffffff, to, 1);
            val = (v > from && v <= to);
            bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                item_group = __shfl_sync(0xffffffff, item_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (item_group != 0) v = v-from-1;
            }else{
                to = reg_item1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (v > from && v <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    item_group = __shfl_sync(0xffffffff, item_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (item_group != 0) v = v-from-1;
                }else{
                    //! 더 좋음
                    for (int t = item_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id -1 > -1){
                            from = item_group_end_idx_s[t-lane_id-1];
                            to = item_group_end_idx_s[t-lane_id];
                            val = (v > from && v <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (item_group != 0) v = v-item_group_end_idx_s[item_group]-1;
                            break;
                        }
                    }   
                }
            }       

            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];
//! Indexing area (end point)

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];
//! Read area (start point)
                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
//! Read area (end point)

//! Computation area (start point)
                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));
//! Computation area (end point)

//! Write area (start point)
                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
//! Write area (end point)

//! Diversity area (start point)
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                        item_group_sum_norms_s[item_group] += norm_q;
                    }

                    user_group_base = user_group * k;
                    // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;
                    // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                } 
//! Diversity area (end point)       
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                    }

                    user_group_base = user_group * k;
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val);
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));

                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        item_group_sum_norms_s[item_group] += (((norm_q)));
                    }

                    item_group_base = item_group * k;
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val);

                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
            processed_cnt += 1;
// #endif

        }
    }

    __syncthreads();

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        norm_sum_p[gridDim.x * i + blockIdx.x] = user_group_sum_norms_s[i];
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        norm_sum_q[gridDim.x * i + blockIdx.x] = item_group_sum_norms_s[i];
    }

    //! Group 128이하에서 가능한 코드
    // if (threadIdx.x < user_group_num) {
    //     norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    // }

    // if (threadIdx.x < item_group_num) {
    //     norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    // }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grad_diversity_not_grouped_only_switching(
                            const Node *R,
                            unsigned int nnz,
                            void* p,
                            void* q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned char cur_precision,
                            unsigned int first_sample_rating_idx,
                            float* sum_updated_val,
                            float* sum_norms
                            )
{

    unsigned int processed_cnt = 0;
    float sum_norms_reg = 0;
    int lane_id = threadIdx.x%32;
    int local_wid = threadIdx.x/32;
    int local_w_num = blockDim.x/32;
    int wid = local_w_num*blockIdx.x + local_wid;  

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {    
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        // All threads read x from laneid 0
        start_id = __shfl_sync(0xffffffff,start_id, 0);

        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            //read the p & q into register file.
            int base_p = u*k;
            int base_q = v*k;
            
            //! User & item = half
            if (!cur_precision){
                __half lrate = __float2half(lrate_f);
                __half lambda = __float2half(lambda_f);
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p;
                __half* tmp_q_ptr = (__half*)q;

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);
                
                if (processed_cnt >= first_sample_rating_idx){
                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    float norm_pq = norm_p + norm_q; 

                    norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 16);
                    norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 8);
                    norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 4);
                    norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 2);
                    norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 1);

                    if (lane_id == 0){
                        sum_norms_reg += norm_pq;
                    }

                    // atomic add global mem
                    atomicAdd(sum_updated_val + lane_id, tmp_p1_updated_val_f + tmp_q1_updated_val_f);
                    atomicAdd(sum_updated_val + lane_id + 32, tmp_p2_updated_val_f + tmp_q2_updated_val_f);
                    atomicAdd(sum_updated_val + lane_id + 64, tmp_p3_updated_val_f + tmp_q3_updated_val_f);
                    atomicAdd(sum_updated_val + lane_id + 96, tmp_p4_updated_val_f + tmp_q4_updated_val_f);
                }
            }
            // //! User & item = single
            else {
                float* tmp_p_ptr = (float*)p;
                float* tmp_q_ptr = (float*)q;

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }

            processed_cnt += 1;
        }    
    }

    if (lane_id == 0) sum_norms[wid] = sum_norms_reg;
}

//! Average of user, item gradient diversity version
// __global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grad_diversity_not_grouped_only_switching(
//                             const Node *R,
//                             unsigned int nnz,
//                             void* p,
//                             void* q,
//                             curandState *state,
//                             float lrate_f,
//                             int k,
//                             int num_iters,
//                             int current_iter,
//                             int update_count_this_block,
//                             int update_vector_size,
//                             float lambda_f,
//                             unsigned char cur_precision,
//                             unsigned int first_sample_rating_idx,
//                             float* sum_updated_val,
//                             float* sum_norms,
//                             float* sum_updated_val_item,
//                             float* sum_norms_item
//                             )
// {

//     unsigned int processed_cnt = 0;
//     float sum_norms_reg = 0;
//     float sum_norms_reg_item = 0;
//     int lane_id = threadIdx.x%32;
//     int local_wid = threadIdx.x/32;
//     int local_w_num = blockDim.x/32;
//     int wid = local_w_num*blockIdx.x + local_wid;  

//     for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
//     {    
//         unsigned int start_id = 0;
//         if(lane_id == 0)
//         {
//             unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
//             start_id = origin%nnz;
//         }

//         // All threads read x from laneid 0
//         start_id = __shfl_sync(0xffffffff,start_id, 0);

//         for(int i = 0;i < update_vector_size;i++)
//         {
//             int offset = (start_id + i)%nnz;
            
//             float r = __ldg(&R[offset].r);
//             int u = __ldg(&R[offset].u);
//             int v = __ldg(&R[offset].i);

//             //read the p & q into register file.
//             int base_p = u*k;
//             int base_q = v*k;
            
//             //! User & item = half
//             if (!cur_precision){
//                 __half lrate = __float2half(lrate_f);
//                 __half lambda = __float2half(lambda_f);
//                 __half r_h = __float2half(r);
//                 __half* tmp_p_ptr = (__half*)p;
//                 __half* tmp_q_ptr = (__half*)q;

//                 const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 const __half ruv = r_h - tmp_product;

//                 const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
//                 const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
//                 const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
//                 const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
//                 const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
//                 const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
//                 const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
//                 const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

//                 tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
//                 tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
//                 tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
//                 tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);
                
//                 const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
//                 const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
//                 const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
//                 const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
//                 const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
//                 const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
//                 const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
//                 const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);
                
//                 if (processed_cnt >= first_sample_rating_idx){
//                     float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
//                     float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));

//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
//                     norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
//                     norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);
                    
//                     if (lane_id == 0){
//                         sum_norms_reg += norm_p;
//                         sum_norms_reg_item += norm_q;                        
//                     }

//                     // atomic add global mem
//                     atomicAdd(sum_updated_val + lane_id, tmp_p1_updated_val_f);
//                     atomicAdd(sum_updated_val + lane_id + 32, tmp_p2_updated_val_f);
//                     atomicAdd(sum_updated_val + lane_id + 64, tmp_p3_updated_val_f);
//                     atomicAdd(sum_updated_val + lane_id + 96, tmp_p4_updated_val_f);

//                     atomicAdd(sum_updated_val_item + lane_id, tmp_q1_updated_val_f);
//                     atomicAdd(sum_updated_val_item + lane_id + 32, tmp_q2_updated_val_f);
//                     atomicAdd(sum_updated_val_item + lane_id + 64, tmp_q3_updated_val_f);
//                     atomicAdd(sum_updated_val_item + lane_id + 96, tmp_q4_updated_val_f);
//                 }
//             }
//             // //! User & item = single
//             else {
//                 float* tmp_p_ptr = (float*)p;
//                 float* tmp_q_ptr = (float*)q;

//                 float tmp_p1 = tmp_p_ptr[base_p + lane_id];
//                 float tmp_q1 = tmp_q_ptr[base_q + lane_id];
//                 float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//                 float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//                 float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//                 float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//                 float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//                 float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//                 float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 float ruv = r - tmp_product;
                
//                 tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
//                 tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

//                 tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
//                 tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

//                 tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
//                 tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

//                 tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
//                 tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
//             }

//             processed_cnt += 1;
//         }    
//     }

//     if (lane_id == 0) {
//         sum_norms[wid] = sum_norms_reg;
//         sum_norms_item[wid] = sum_norms_reg_item;
//     }
// }

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grad_diversity_not_switching_only_grouping(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            int user_group_num,
                            int item_group_num
                            )
{    
    // extern __shared__ float array[];

    // //! Original
    // void** p_s = (void**)array;
    // void** q_s = (void**)&p_s[user_group_num];
    // unsigned int* user_group_end_idx_s = (unsigned int*)&q_s[item_group_num]; 
    // unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    // unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num + 1];
    // unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    // for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
    //     p_s[i] = p[i];
    //     user_group_prec_s[i] = user_group_prec_info[i];
    //     user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    // }

    // for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
    //     q_s[i] = q[i];
    //     item_group_prec_s[i] = item_group_prec_info[i];
    //     item_group_end_idx_s[i+1] = item_group_end_idx[i];
    // }

    // if (threadIdx.x == 0){
    //     user_group_end_idx_s[0] = -1;
    //     item_group_end_idx_s[0] = -1;      
    // }

    extern __shared__ float array[];

    int reg_user=-1;
    int reg_item=-1;
    int reg_user1=-1;
    int reg_item1=-1;

    //! Original
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&q_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = user_group_end_idx_s;
    if (user_group_num > 62) {
        item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num - 61];
        if (threadIdx.x == 0) user_group_end_idx_s[0] = -1;
    }
    unsigned char* user_group_prec_s = (unsigned char*)item_group_end_idx_s;
    if (item_group_num > 62) {
        user_group_prec_s = (unsigned char*)&item_group_end_idx_s[item_group_num - 61];
        if (threadIdx.x == 0) item_group_end_idx_s[0] = -1;
    }

    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int lane_id = threadIdx.x%32;

    if (user_group_num-1-lane_id > -1) reg_user = user_group_end_idx[user_group_num-1-lane_id];
    if (item_group_num-1-lane_id > -1) reg_item = item_group_end_idx[item_group_num-1-lane_id];
    if (user_group_num-32-lane_id > -1) reg_user1 = user_group_end_idx[user_group_num-32-lane_id];
    if (item_group_num-32-lane_id > -1) reg_item1 = item_group_end_idx[item_group_num-32-lane_id];
    
    for (int i = threadIdx.x; i <= user_group_num-63; i+= blockDim.x){
        user_group_end_idx_s[i+1] = user_group_end_idx[i]; 
    }  

    for (int i = threadIdx.x; i <= item_group_num-63; i+= blockDim.x){
        item_group_end_idx_s[i+1] = item_group_end_idx[i]; 
    }   

    for (int i = threadIdx.x; i < user_group_num; i+= blockDim.x){
        p_s[i] = p[i];
        user_group_prec_s[i] = user_group_prec_info[i];
    }

    for (int i = threadIdx.x; i < item_group_num; i+= blockDim.x){
        q_s[i] = q[i];
        item_group_prec_s[i] = item_group_prec_info[i];
    }


    __syncthreads();
    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            // int user_group = 0;
            // int item_group = 0;

            // //! 더 좋음
            // for (int t = user_group_num;; t-=32){
            //     int val = 0;
            //     if (t - lane_id - 1 > -1){
            //         int from = user_group_end_idx_s[t-lane_id-1];
            //         int to = user_group_end_idx_s[t-lane_id];
            //         val = (u > from && u <= to) * (t - lane_id);
            //     }
            //     unsigned bitpack = __ballot_sync(0xffffffff, val);
            //     if (bitpack != 0){
            //         user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
            //         break;
            //     }
            // }

            // if (user_group != 0) u = u-user_group_end_idx_s[user_group]-1;

            // //! 더 좋음
            // for (int t = item_group_num;; t-=32){
            //     int val = 0;
            //     if (t - lane_id -1 > -1){
            //         int from = item_group_end_idx_s[t-lane_id-1];
            //         int to = item_group_end_idx_s[t-lane_id];
            //         val = (v > from && v <= to) * (t - lane_id);
            //     }
            //     unsigned bitpack = __ballot_sync(0xffffffff, val);
            //     if (bitpack != 0){
            //         item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
            //         break;
            //     }
            // }            

            // if (item_group != 0) v = v-item_group_end_idx_s[item_group]-1;


            int user_group = -1;
            int item_group = -1;

            int to = reg_user;
            int from = __shfl_down_sync(0xffffffff, to, 1);
            int val = (u > from && u <= to);
            unsigned bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                user_group = __shfl_sync(0xffffffff, user_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (user_group != 0) u = u-from-1;

            }else{
                to = reg_user1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (u > from && u <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    user_group = __shfl_sync(0xffffffff, user_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (user_group != 0) u = u-from-1;
                }else{
                    for (int t = user_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id - 1 > -1){
                            from = user_group_end_idx_s[t-lane_id-1];
                            to = user_group_end_idx_s[t-lane_id];
                            val = (u > from && u <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            user_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (user_group != 0) u = u-user_group_end_idx_s[user_group]-1;
                            break;
                        }
                    }
                }
            }

            to = reg_item;
            from = __shfl_down_sync(0xffffffff, to, 1);
            val = (v > from && v <= to);
            bitpack = __ballot_sync(0xffffffff, val);
            if (bitpack != 0){
                int act_lane = __ffs(bitpack)-1;
                item_group = __shfl_sync(0xffffffff, item_group_num - 1 - lane_id , act_lane);
                from = __shfl_sync(0xffffffff, from, act_lane);
                if (item_group != 0) v = v-from-1;
            }else{
                to = reg_item1;
                from = __shfl_down_sync(0xffffffff, to, 1);
                val = (v > from && v <= to);
                bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    int act_lane = __ffs(bitpack)-1;
                    item_group = __shfl_sync(0xffffffff, item_group_num - 32 - lane_id , act_lane);
                    from = __shfl_sync(0xffffffff, from, act_lane);
                    if (item_group != 0) v = v-from-1;
                }else{
                    //! 더 좋음
                    for (int t = item_group_num-62;; t-=32){
                        val = 0;
                        if (t - lane_id -1 > -1){
                            from = item_group_end_idx_s[t-lane_id-1];
                            to = item_group_end_idx_s[t-lane_id];
                            val = (v > from && v <= to) * (t - lane_id);
                        }
                        bitpack = __ballot_sync(0xffffffff, val);
                        if (bitpack != 0){
                            item_group = __shfl_sync(0xffffffff,val-1 ,__ffs(bitpack)-1);
                            if (item_group != 0) v = v-item_group_end_idx_s[item_group]-1;
                            break;
                        }
                    }   
                }
            }       

            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);  
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                tmp_p_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                tmp_q_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                tmp_p_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                tmp_q_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                tmp_q_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                tmp_q_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);     
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                tmp_p_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                tmp_p_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                tmp_p_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                tmp_p_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;
                
                tmp_p_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                tmp_q_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
        }
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_random_fp16(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            int user_group_num,
                            int item_group_num,
                            unsigned int fp16_user_num,
                            unsigned int fp16_item_num
                            )
{        
    __half* fp16_user_ptr = (__half*)p[0];
    float* fp32_user_ptr = (float*)p[1];

    __half* fp16_item_ptr = (__half*)q[0];
    float* fp32_item_ptr = (float*)q[1];

    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        int lane_id = threadIdx.x%32;

        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            unsigned char user_prec = 0;
            unsigned char item_prec = 0;

            if (u >= fp16_user_num) {
                user_prec = 1;
                u = u - fp16_user_num;
            }

            if (v >= fp16_item_num) {
                item_prec = 1;
                v = v - fp16_item_num;
            }

            int base_p = u*k;
            int base_q = v*k;

            // unsigned char user_prec = user_group_prec_s[user_group];
            // unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half r_h = __float2half(r);
                // __half* tmp_p_ptr = (__half*)p_s[0];
                // __half* tmp_q_ptr = (__half*)q_s[0];

                const __half tmp_p1 = fp16_user_ptr[base_p + lane_id];
                const __half tmp_q1 = fp16_item_ptr[base_q + lane_id];
                const __half tmp_p2 = fp16_user_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = fp16_item_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = fp16_user_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = fp16_item_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = fp16_user_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = fp16_item_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r_h - tmp_product;

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                fp16_user_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                fp16_item_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                fp16_user_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                fp16_item_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                fp16_user_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                fp16_item_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                fp16_user_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                fp16_item_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);  
            }
            //! user half item single
            else if (!user_prec && item_prec){
                // __half* tmp_p_ptr = (__half*)p_s[0];
                // float* tmp_q_ptr = (float*)q_s[1];

                const __half tmp_p1 = fp16_user_ptr[base_p + lane_id];
                const float tmp_q1_f = fp32_item_ptr[base_q + lane_id];
                const __half tmp_p2 = fp16_user_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = fp32_item_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = fp16_user_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = fp32_item_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = fp16_user_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = fp32_item_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                fp16_user_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                fp32_item_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                fp16_user_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                fp32_item_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                fp16_user_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                fp32_item_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                fp16_user_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                fp32_item_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);     
            }
            //! user single item half
            else if (user_prec && !item_prec){
                // float* tmp_p_ptr = (float*)p_s[1];
                // __half* tmp_q_ptr = (__half*)q_s[0];

                const float tmp_p1_f = fp32_user_ptr[base_p + lane_id];
                const __half tmp_q1 = fp16_item_ptr[base_q + lane_id];
                const float tmp_p2_f = fp32_user_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = fp16_item_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = fp32_user_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = fp16_item_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = fp32_user_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = fp16_item_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                fp32_user_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                fp16_item_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                fp32_user_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                fp16_item_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                fp32_user_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                fp16_item_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                fp32_user_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                fp16_item_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));
            }
            //! user single item single
            else{
                // float* tmp_p_ptr = (float*)p_s[1];
                // float* tmp_q_ptr = (float*)q_s[1];

                float tmp_p1 = fp32_user_ptr[base_p + lane_id];
                float tmp_q1 = fp32_item_ptr[base_q + lane_id];
                float tmp_p2 = fp32_user_ptr[base_p + lane_id + 32];
                float tmp_q2 = fp32_item_ptr[base_q + lane_id + 32];
                float tmp_p3 = fp32_user_ptr[base_p + lane_id + 64];
                float tmp_q3 = fp32_item_ptr[base_q + lane_id + 64];
                float tmp_p4 = fp32_user_ptr[base_p + lane_id + 96];
                float tmp_q4 = fp32_item_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;
                
                fp32_user_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                fp32_item_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                fp32_user_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                fp32_item_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                fp32_user_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                fp32_item_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                fp32_user_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                fp32_item_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
        }
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_random_fp16_comp_fp32(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            int user_group_num,
                            int item_group_num,
                            unsigned int fp16_user_num,
                            unsigned int fp16_item_num
                            )
{        
    __half* fp16_user_ptr = (__half*)p[0];
    float* fp32_user_ptr = (float*)p[1];

    __half* fp16_item_ptr = (__half*)q[0];
    float* fp32_item_ptr = (float*)q[1];

    __half lrate = __float2half(lrate_f);
    __half lambda = __float2half(lambda_f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        int lane_id = threadIdx.x%32;

        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;

            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            unsigned char user_prec = 0;
            unsigned char item_prec = 0;

            if (u >= fp16_user_num) {
                user_prec = 1;
                u = u - fp16_user_num;
            }

            if (v >= fp16_item_num) {
                item_prec = 1;
                v = v - fp16_item_num;
            }

            int base_p = u*k;
            int base_q = v*k;

            // unsigned char user_prec = user_group_prec_s[user_group];
            // unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                // __half r_h = __float2half(r);
                // __half* tmp_p_ptr = (__half*)p_s[0];
                // __half* tmp_q_ptr = (__half*)q_s[0];

                const __half tmp_p1 = fp16_user_ptr[base_p + lane_id];
                const __half tmp_q1 = fp16_item_ptr[base_q + lane_id];
                const __half tmp_p2 = fp16_user_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = fp16_item_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = fp16_user_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = fp16_item_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = fp16_user_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = fp16_item_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_p4_f = __half2float(tmp_p4);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                // fp16_user_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                // fp16_item_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                // fp16_user_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                // fp16_item_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                // fp16_user_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                // fp16_item_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                // fp16_user_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                // fp16_item_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);  

                fp16_user_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                fp16_item_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));

                fp16_user_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                fp16_item_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));

                fp16_user_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                fp16_item_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));

                fp16_user_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                fp16_item_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));
            }
            //! user half item single
            else if (!user_prec && item_prec){
                // __half* tmp_p_ptr = (__half*)p_s[0];
                // float* tmp_q_ptr = (float*)q_s[1];

                const __half tmp_p1 = fp16_user_ptr[base_p + lane_id];
                const float tmp_q1_f = fp32_item_ptr[base_q + lane_id];
                const __half tmp_p2 = fp16_user_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = fp32_item_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = fp16_user_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = fp32_item_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = fp16_user_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = fp32_item_ptr[base_q + lane_id + 96];

                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_p1_updated_val = (ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                const float tmp_p2_updated_val = (ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                const float tmp_p3_updated_val = (ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                const float tmp_p4_updated_val = (ruv*tmp_q4_f - lambda_f*tmp_p4_f);

                fp16_user_ptr[base_p + lane_id] = __float2half(fmaf(lrate_f, tmp_p1_updated_val, tmp_p1_f));
                fp32_item_ptr[base_q + lane_id] = fmaf(lrate_f, (ruv*tmp_p1_f - lambda_f*tmp_q1_f), tmp_q1_f);     
                fp16_user_ptr[base_p + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_p2_updated_val, tmp_p2_f));
                fp32_item_ptr[base_q + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_p2_f - lambda_f*tmp_q2_f), tmp_q2_f);
                fp16_user_ptr[base_p + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_p3_updated_val, tmp_p3_f));
                fp32_item_ptr[base_q + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_p3_f - lambda_f*tmp_q3_f), tmp_q3_f);
                fp16_user_ptr[base_p + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_p4_updated_val, tmp_p4_f));
                fp32_item_ptr[base_q + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_p4_f - lambda_f*tmp_q4_f), tmp_q4_f);     
            }
            //! user single item half
            else if (user_prec && !item_prec){
                // float* tmp_p_ptr = (float*)p_s[1];
                // __half* tmp_q_ptr = (__half*)q_s[0];

                const float tmp_p1_f = fp32_user_ptr[base_p + lane_id];
                const __half tmp_q1 = fp16_item_ptr[base_q + lane_id];
                const float tmp_p2_f = fp32_user_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = fp16_item_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = fp32_user_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = fp16_item_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = fp32_user_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = fp16_item_ptr[base_q + lane_id + 96];
                
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                const float tmp_q1_updated_val = (ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                const float tmp_q2_updated_val = (ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                const float tmp_q3_updated_val = (ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                const float tmp_q4_updated_val = (ruv*tmp_p4_f - lambda_f*tmp_q4_f);
                
                fp32_user_ptr[base_p + lane_id] = fmaf(lrate_f, (ruv*tmp_q1_f) - (lambda_f*tmp_p1_f), tmp_p1_f);
                fp16_item_ptr[base_q + lane_id] = __float2half(fmaf(lrate_f, tmp_q1_updated_val, tmp_q1_f));
                fp32_user_ptr[base_p + lane_id + 32] = fmaf(lrate_f, (ruv*tmp_q2_f) - (lambda_f*tmp_p2_f), tmp_p2_f);
                fp16_item_ptr[base_q + lane_id + 32] = __float2half(fmaf(lrate_f, tmp_q2_updated_val, tmp_q2_f));
                fp32_user_ptr[base_p + lane_id + 64] = fmaf(lrate_f, (ruv*tmp_q3_f) - (lambda_f*tmp_p3_f), tmp_p3_f);
                fp16_item_ptr[base_q + lane_id + 64] = __float2half(fmaf(lrate_f, tmp_q3_updated_val, tmp_q3_f));
                fp32_user_ptr[base_p + lane_id + 96] = fmaf(lrate_f, (ruv*tmp_q4_f) - (lambda_f*tmp_p4_f), tmp_p4_f);
                fp16_item_ptr[base_q + lane_id + 96] = __float2half(fmaf(lrate_f, tmp_q4_updated_val, tmp_q4_f));
            }
            //! user single item single
            else{
                // float* tmp_p_ptr = (float*)p_s[1];
                // float* tmp_q_ptr = (float*)q_s[1];

                float tmp_p1 = fp32_user_ptr[base_p + lane_id];
                float tmp_q1 = fp32_item_ptr[base_q + lane_id];
                float tmp_p2 = fp32_user_ptr[base_p + lane_id + 32];
                float tmp_q2 = fp32_item_ptr[base_q + lane_id + 32];
                float tmp_p3 = fp32_user_ptr[base_p + lane_id + 64];
                float tmp_q3 = fp32_item_ptr[base_q + lane_id + 64];
                float tmp_p4 = fp32_user_ptr[base_p + lane_id + 96];
                float tmp_q4 = fp32_item_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = r - tmp_product;
                
                fp32_user_ptr[base_p + lane_id] =  fmaf(lrate_f, (ruv*tmp_q1 - lambda_f*tmp_p1), tmp_p1);
                fp32_item_ptr[base_q + lane_id] =  fmaf(lrate_f, (ruv*tmp_p1 - lambda_f*tmp_q1), tmp_q1);

                fp32_user_ptr[base_p + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_q2 - lambda_f*tmp_p2), tmp_p2);
                fp32_item_ptr[base_q + lane_id + 32] =  fmaf(lrate_f, (ruv*tmp_p2 - lambda_f*tmp_q2), tmp_q2);

                fp32_user_ptr[base_p + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_q3 - lambda_f*tmp_p3), tmp_p3);
                fp32_item_ptr[base_q + lane_id + 64] =  fmaf(lrate_f, (ruv*tmp_p3 - lambda_f*tmp_q3), tmp_q3);

                fp32_user_ptr[base_p + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_q4 - lambda_f*tmp_p4), tmp_p4);
                fp32_item_ptr[base_q + lane_id + 96] =  fmaf(lrate_f, (ruv*tmp_p4 - lambda_f*tmp_q4), tmp_q4);
            }
        }
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_range_based_indexing_error_grad_diversity_grouped_cache_timing_overhead(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            // Index_info_node* user_index_info,
                            // Index_info_node* item_index_info,
                            unsigned int* user_group_end_idx,
                            unsigned int* item_group_end_idx,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num,
                            unsigned int uncached_user_num,
                            unsigned int uncached_item_num,
                            long long int* time_per_block_arr
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_updated_val_s = (float*)&q_s[item_group_num]; 
    float* item_group_sum_updated_val_s = (float*)&user_group_sum_updated_val_s[(user_group_num - uncached_user_num) * k];
    float* user_group_sum_norms_s = (float*)&item_group_sum_updated_val_s[(item_group_num - uncached_item_num) * k];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned int* user_group_end_idx_s = (unsigned int*)&item_group_sum_norms_s[item_group_num]; 
    unsigned int* item_group_end_idx_s = (unsigned int*)&user_group_end_idx_s[user_group_num + 1];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_end_idx[item_group_num + 1];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int tid = threadIdx.x;

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_sum_norms_s[threadIdx.x] = 0.f;
        user_group_end_idx_s[threadIdx.x+1] = user_group_end_idx[threadIdx.x];    
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_sum_norms_s[threadIdx.x] = 0.f;
        item_group_end_idx_s[threadIdx.x+1] = item_group_end_idx[threadIdx.x];
    }
    
    if (threadIdx.x == 0){
        user_group_end_idx_s[0] = -1;
        item_group_end_idx_s[0] = -1;      
    }
    
    // for (;tid<(user_group_num - uncached_user_num)*128;tid+=blockDim.x) user_group_sum_updated_val_s[tid] = 0.f;
    // tid = threadIdx.x;
    // for (;tid<(item_group_num - uncached_item_num)*128;tid+=blockDim.x) item_group_sum_updated_val_s[tid] = 0.f;

    // 0으로 초기화
    __syncthreads();

    unsigned int processed_cnt = 0;
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);
    __half zero = __float2half_rn(0.f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);
            
            int user_group = 0;
            int item_group = 0;

            for (int t = 0; t < user_group_num; t+=32){
                int val = 0;
                if (t + lane_id < user_group_num){
                    int from = user_group_end_idx_s[t+lane_id];
                    int to = user_group_end_idx_s[t+lane_id+1];
                    val = (u > from && u <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    user_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }

            if (user_group != 0)
                u = u-user_group_end_idx_s[user_group]-1;

            for (int t = 0; t < item_group_num; t+=32){
                int val = 0;
                if (t + lane_id < item_group_num){
                    int from = item_group_end_idx_s[t+lane_id];
                    int to = item_group_end_idx_s[t+lane_id+1];
                    val = (v > from && v <= to) * (t + lane_id);
                }
                unsigned bitpack = __ballot_sync(0xffffffff, val);
                if (bitpack != 0){
                    item_group = __shfl_sync(0xffffffff,val ,__ffs(bitpack)-1);
                    break;
                }
            }
            
            if (item_group != 0)
                v = v-item_group_end_idx_s[item_group]-1;

            int base_p = u*k;
            int base_q = v*k;

            // __half r = __float2half_rn(__ldg(&R[offset].r));
            // int orig_u = __ldg(&R[offset].u);
            // int orig_v = __ldg(&R[offset].i);
            
            // int user_group = user_index_info[orig_u].g;
            // int u = user_index_info[orig_u].v;
            // int item_group = item_index_info[orig_v].g;
            // int v = item_index_info[orig_v].v;
            // int base_p = u*k;
            // int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                // const __half tmp_p1_updated_val = lrate*(ruv*tmp_q1 - lambda*tmp_p1);
                // const __half tmp_q1_updated_val = lrate*(ruv*tmp_p1 - lambda*tmp_q1);
                // const __half tmp_p2_updated_val = lrate*(ruv*tmp_q2 - lambda*tmp_p2);
                // const __half tmp_q2_updated_val = lrate*(ruv*tmp_p2 - lambda*tmp_q2);
                // const __half tmp_p3_updated_val = lrate*(ruv*tmp_q3 - lambda*tmp_p3);
                // const __half tmp_q3_updated_val = lrate*(ruv*tmp_p3 - lambda*tmp_q3);
                // const __half tmp_p4_updated_val = lrate*(ruv*tmp_q4 - lambda*tmp_p4);
                // const __half tmp_q4_updated_val = lrate*(ruv*tmp_p4 - lambda*tmp_q4);

                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);

                //! ============================ Error computation area ============================               
                if (threadIdx.x == 0) time_per_block_arr[blockIdx.x] = clock64();

                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);
                
                if (processed_cnt >= first_sample_rating_idx){
                    unsigned int user_group_base;
                    unsigned int item_group_base;

                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    // float norm_p = __powf(tmp_p1_updated_val_f,2) + __powf(tmp_p2_updated_val_f,2) + __powf(tmp_p3_updated_val_f,2) + __powf(tmp_p4_updated_val_f,2); // longer 
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += norm_p;
                        item_group_sum_norms_s[item_group] += norm_q;
                    }

                    if (user_group >= uncached_user_num) {
                        user_group_base = (user_group - uncached_user_num) * k;
                        // user_group_sum_updated_val_s[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                        // user_group_sum_updated_val_s[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);

                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);                        
                    } 
                    else {
                        user_group_base = user_group * k;
                        // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                        // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                        
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    }

                    if (item_group >= uncached_item_num) {
                        item_group_base = (item_group - uncached_item_num) * k;
                        // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    else {
                        item_group_base = item_group * k;
                        // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                        
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                }
                __syncthreads();

                //! sync 꼭 있어야?
                if (threadIdx.x == 0) time_per_block_arr[blockIdx.x + gridDim.x] += (clock64() - time_per_block_arr[blockIdx.x]);
                // if (threadIdx.x == 0) atomicAdd(time_per_block_arr + (blockIdx.x + gridDim.x), (clock64() - time_per_block_arr[blockIdx.x]));

                // if (threadIdx.x == 0) time_per_block_arr[blockIdx.x + gridDim.x] = blockIdx.x + gridDim.x;
        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

                const __half tmp_q1 = __float2half_rn(tmp_q1_f);
                const __half tmp_q2 = __float2half_rn(tmp_q2_f);
                const __half tmp_q3 = __float2half_rn(tmp_q3_f);
                const __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                const __half tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
                const __half tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
                const __half tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
                const __half tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));

                tmp_p_ptr[base_p + lane_id] = __hfma(lrate, tmp_p1_updated_val, tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = __hfma(lrate, tmp_p2_updated_val, tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = __hfma(lrate, tmp_p3_updated_val, tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = __hfma(lrate, tmp_p4_updated_val, tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);


                //! Error computation area
                if (threadIdx.x == 0) time_per_block_arr[blockIdx.x] = clock64();

                const float tmp_p1_updated_val_f = __half2float(tmp_p1_updated_val);
                const float tmp_p2_updated_val_f = __half2float(tmp_p2_updated_val);
                const float tmp_p3_updated_val_f = __half2float(tmp_p3_updated_val);
                const float tmp_p4_updated_val_f = __half2float(tmp_p4_updated_val);
                
                if (processed_cnt >= first_sample_rating_idx){
                    // unsigned int user_group_base = user_group * k;
                    unsigned int user_group_base;
                    float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                    
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                    norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

                    if (lane_id == 0){
                        user_group_sum_norms_s[user_group] += (((norm_p)));
                    }

                    if (user_group >= uncached_user_num) {
                        user_group_base = (user_group - uncached_user_num) * k;
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);                        
                    } 
                    else {
                        user_group_base = user_group * k;
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                        atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);
                    }
                }

                if (threadIdx.x == 0) time_per_block_arr[blockIdx.x + blockDim.x] += clock64() - time_per_block_arr[blockIdx.x];
        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                const __half tmp_p1 = __float2half_rn(tmp_p1_f);
                const __half tmp_p2 = __float2half_rn(tmp_p2_f);
                const __half tmp_p3 = __float2half_rn(tmp_p3_f);
                const __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                const __half tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
                const __half tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
                const __half tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
                const __half tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = __hfma(lrate, tmp_q1_updated_val, tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = __hfma(lrate, tmp_q2_updated_val, tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = __hfma(lrate, tmp_q3_updated_val, tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = __hfma(lrate, tmp_q4_updated_val, tmp_q4);    

                //! Error computation area
                if (threadIdx.x == 0) time_per_block_arr[blockIdx.x] = clock64();

                const float tmp_q1_updated_val_f = __half2float(tmp_q1_updated_val);
                const float tmp_q2_updated_val_f = __half2float(tmp_q2_updated_val);
                const float tmp_q3_updated_val_f = __half2float(tmp_q3_updated_val);
                const float tmp_q4_updated_val_f = __half2float(tmp_q4_updated_val);

                if (processed_cnt >= first_sample_rating_idx){
                    // unsigned int item_group_base = item_group * k;
                    unsigned int item_group_base;
                    float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                    
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                    norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                    if (lane_id == 0){
                        item_group_sum_norms_s[item_group] += (((norm_q)));
                    }

                    if (item_group >= uncached_item_num) {
                        item_group_base = (item_group - uncached_item_num) * k;
                        // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    else {
                        item_group_base = item_group * k;
                        // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                        // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                        
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                        atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                    }
                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                }

                if (threadIdx.x == 0) time_per_block_arr[blockIdx.x + blockDim.x] += clock64() - time_per_block_arr[blockIdx.x];
     
            }
            //! user single item single
            else{
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }

            processed_cnt += 1;
        }
    }

    __syncthreads();

    int local_wid = threadIdx.x/32;
    int local_w_num = blockDim.x/32;
    int lane_id = threadIdx.x%32;

    // for (; local_wid < user_group_num; local_wid += local_w_num){
    //     int base_user_group = local_wid * k;
    //     if (local_wid >= uncached_user_num){
    //         int converted_user_group = (local_wid - uncached_user_num) * k;
    //         grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val_s[converted_user_group + lane_id];
    //         grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val_s[converted_user_group + lane_id + 32];
    //         grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val_s[converted_user_group + lane_id + 64];
    //         grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val_s[converted_user_group + lane_id + 96];
    //     }
    //     // else{
    //     //     // grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val[base_user_group + lane_id];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val[base_user_group + lane_id + 32];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val[base_user_group + lane_id + 64];
    //     //     // grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val[base_user_group + lane_id + 96];

    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id, user_group_sum_updated_val[base_user_group + lane_id]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 32, user_group_sum_updated_val[base_user_group + lane_id + 32]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 64, user_group_sum_updated_val[base_user_group + lane_id + 64]);
    //     //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 96, user_group_sum_updated_val[base_user_group + lane_id + 96]);
    //     // }
    // }

    // local_wid = threadIdx.x/32;
    
    // for (; local_wid < item_group_num; local_wid += local_w_num){
    //     int base_item_group = local_wid * k;
    //     if (local_wid >= uncached_item_num){
    //         int converted_item_group = (local_wid - uncached_item_num) * k;
    //         grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val_s[converted_item_group + lane_id];
    //         grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val_s[converted_item_group + lane_id + 32];
    //         grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val_s[converted_item_group + lane_id + 64];
    //         grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val_s[converted_item_group + lane_id + 96];
    //     }
    //     // else {
    //     //     // grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val[base_item_group + lane_id];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val[base_item_group + lane_id + 32];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val[base_item_group + lane_id + 64];
    //     //     // grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val[base_item_group + lane_id + 96];
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id, item_group_sum_updated_val[base_item_group + lane_id]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 32, item_group_sum_updated_val[base_item_group + lane_id + 32]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 64, item_group_sum_updated_val[base_item_group + lane_id + 64]);
    //     //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 96, item_group_sum_updated_val[base_item_group + lane_id + 96]);
    //     // }
    // }

    if (threadIdx.x < user_group_num) {
        norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    }

    if (threadIdx.x < item_group_num) {
        norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    }
}


__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_grouped_cache_fp32_verison(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            float* user_group_sum_updated_val,
                            float* item_group_sum_updated_val,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num,
                            unsigned int uncached_user_num,
                            unsigned int uncached_item_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    float* user_group_sum_updated_val_s = (float*)&q_s[item_group_num]; 
    float* item_group_sum_updated_val_s = (float*)&user_group_sum_updated_val_s[(user_group_num - uncached_user_num) * k];
    float* user_group_sum_norms_s = (float*)&item_group_sum_updated_val_s[(item_group_num - uncached_item_num) * k];
    float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_sum_norms_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int tid = threadIdx.x;

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_sum_norms_s[threadIdx.x] = 0.f;
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_sum_norms_s[threadIdx.x] = 0.f;
    }

    for (;tid<(user_group_num - uncached_user_num)*128;tid+=blockDim.x) user_group_sum_updated_val_s[tid] = 0.f;
    tid = threadIdx.x;
    for (;tid<(item_group_num - uncached_item_num)*128;tid+=blockDim.x) item_group_sum_updated_val_s[tid] = 0.f;

    // 0으로 초기화
    __syncthreads();

    unsigned int processed_cnt = 0;

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            float r = __ldg(&R[offset].r);
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            float* tmp_p_ptr = (float*)p_s[user_group];
            float* tmp_q_ptr = (float*)q_s[item_group];

            const float tmp_p1 = tmp_p_ptr[base_p + lane_id];
            const float tmp_q1 = tmp_q_ptr[base_q + lane_id];
            const float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            const float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            const float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            const float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            const float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            const float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

            float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            const float ruv = r - tmp_product;
            
            // const __half tmp_p1_updated_val = lrate*(ruv*tmp_q1 - lambda*tmp_p1);
            // const __half tmp_q1_updated_val = lrate*(ruv*tmp_p1 - lambda*tmp_q1);
            // const __half tmp_p2_updated_val = lrate*(ruv*tmp_q2 - lambda*tmp_p2);
            // const __half tmp_q2_updated_val = lrate*(ruv*tmp_p2 - lambda*tmp_q2);
            // const __half tmp_p3_updated_val = lrate*(ruv*tmp_q3 - lambda*tmp_p3);
            // const __half tmp_q3_updated_val = lrate*(ruv*tmp_p3 - lambda*tmp_q3);
            // const __half tmp_p4_updated_val = lrate*(ruv*tmp_q4 - lambda*tmp_p4);
            // const __half tmp_q4_updated_val = lrate*(ruv*tmp_p4 - lambda*tmp_q4);

            const float tmp_p1_updated_val = ((ruv*tmp_q1) - (lambda*tmp_p1));
            const float tmp_q1_updated_val = ((ruv*tmp_p1) - (lambda*tmp_q1));
            const float tmp_p2_updated_val = ((ruv*tmp_q2) - (lambda*tmp_p2));
            const float tmp_q2_updated_val = ((ruv*tmp_p2) - (lambda*tmp_q2));
            const float tmp_p3_updated_val = ((ruv*tmp_q3) - (lambda*tmp_p3));
            const float tmp_q3_updated_val = ((ruv*tmp_p3) - (lambda*tmp_q3));
            const float tmp_p4_updated_val = ((ruv*tmp_q4) - (lambda*tmp_p4));
            const float tmp_q4_updated_val = ((ruv*tmp_p4) - (lambda*tmp_q4));

            tmp_p_ptr[base_p + lane_id] = __fmaf_rn(lrate, tmp_p1_updated_val, tmp_p1);
            tmp_q_ptr[base_q + lane_id] = __fmaf_rn(lrate, tmp_q1_updated_val, tmp_q1);
            tmp_p_ptr[base_p + lane_id + 32] = __fmaf_rn(lrate, tmp_p2_updated_val, tmp_p2);
            tmp_q_ptr[base_q + lane_id + 32] = __fmaf_rn(lrate, tmp_q2_updated_val, tmp_q2);
            tmp_p_ptr[base_p + lane_id + 64] = __fmaf_rn(lrate, tmp_p3_updated_val, tmp_p3);
            tmp_q_ptr[base_q + lane_id + 64] = __fmaf_rn(lrate, tmp_q3_updated_val, tmp_q3);
            tmp_p_ptr[base_p + lane_id + 96] = __fmaf_rn(lrate, tmp_p4_updated_val, tmp_p4);
            tmp_q_ptr[base_q + lane_id + 96] = __fmaf_rn(lrate, tmp_q4_updated_val, tmp_q4);
            
            const float tmp_p1_updated_val_f = tmp_p1_updated_val;
            const float tmp_q1_updated_val_f = tmp_q1_updated_val;
            const float tmp_p2_updated_val_f = tmp_p2_updated_val;
            const float tmp_q2_updated_val_f = tmp_q2_updated_val;
            const float tmp_p3_updated_val_f = tmp_p3_updated_val;
            const float tmp_q3_updated_val_f = tmp_q3_updated_val;
            const float tmp_p4_updated_val_f = tmp_p4_updated_val;
            const float tmp_q4_updated_val_f = tmp_q4_updated_val;

            if (processed_cnt >= first_sample_rating_idx){
                unsigned int user_group_base;
                unsigned int item_group_base;

                float norm_p = ((tmp_p1_updated_val_f*tmp_p1_updated_val_f) + (tmp_p2_updated_val_f*tmp_p2_updated_val_f)) + ((tmp_p3_updated_val_f*tmp_p3_updated_val_f) + (tmp_p4_updated_val_f*tmp_p4_updated_val_f));
                // float norm_p = __powf(tmp_p1_updated_val_f,2) + __powf(tmp_p2_updated_val_f,2) + __powf(tmp_p3_updated_val_f,2) + __powf(tmp_p4_updated_val_f,2); // longer 
                float norm_q = ((tmp_q1_updated_val_f*tmp_q1_updated_val_f) + (tmp_q2_updated_val_f*tmp_q2_updated_val_f)) + ((tmp_q3_updated_val_f*tmp_q3_updated_val_f) + (tmp_q4_updated_val_f*tmp_q4_updated_val_f));
                
                norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
                norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
                norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
                norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
                norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                
                norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
                norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
                norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
                norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
                norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

                if (lane_id == 0){
                    user_group_sum_norms_s[user_group] += norm_p;
                    item_group_sum_norms_s[item_group] += norm_q;
                }

                if (user_group >= uncached_user_num) {
                    user_group_base = (user_group - uncached_user_num) * k;
                    // user_group_sum_updated_val_s[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val_s[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);

                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val_s + user_group_base + lane_id + 96, tmp_p4_updated_val_f);                        
                } 
                else {
                    user_group_base = user_group * k;
                    // user_group_sum_updated_val[user_group_base + lane_id] += (tmp_p1_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 32] += (tmp_p2_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 64] += (tmp_p3_updated_val_f);
                    // user_group_sum_updated_val[user_group_base + lane_id + 96] += (tmp_p4_updated_val_f);
                    
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(user_group_sum_updated_val + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                }

                if (item_group >= uncached_item_num) {
                    item_group_base = (item_group - uncached_item_num) * k;
                    // item_group_sum_updated_val_s[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val_s[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);

                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val_s + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }
                else {
                    item_group_base = item_group * k;
                    // item_group_sum_updated_val[item_group_base + lane_id] += (tmp_q1_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 32] += (tmp_q2_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 64] += (tmp_q3_updated_val_f);
                    // item_group_sum_updated_val[item_group_base + lane_id + 96] += (tmp_q4_updated_val_f);
                    
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(item_group_sum_updated_val + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }
            }        
            processed_cnt += 1;
        }
    }

    __syncthreads();

    int local_wid = threadIdx.x/32;
    int local_w_num = blockDim.x/32;
    int lane_id = threadIdx.x%32;

    for (; local_wid < user_group_num; local_wid += local_w_num){
        int base_user_group = local_wid * k;
        if (local_wid >= uncached_user_num){
            int converted_user_group = (local_wid - uncached_user_num) * k;
            grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val_s[converted_user_group + lane_id];
            grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val_s[converted_user_group + lane_id + 32];
            grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val_s[converted_user_group + lane_id + 64];
            grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val_s[converted_user_group + lane_id + 96];
        }
        // else{
        //     // grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val[base_user_group + lane_id];
        //     // grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val[base_user_group + lane_id + 32];
        //     // grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val[base_user_group + lane_id + 64];
        //     // grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val[base_user_group + lane_id + 96];

        //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id, user_group_sum_updated_val[base_user_group + lane_id]);
        //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 32, user_group_sum_updated_val[base_user_group + lane_id + 32]);
        //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 64, user_group_sum_updated_val[base_user_group + lane_id + 64]);
        //     // atomicAdd(grad_sum_norm_p + base_user_group + lane_id + 96, user_group_sum_updated_val[base_user_group + lane_id + 96]);
        // }
    }

    local_wid = threadIdx.x/32;
    
    for (; local_wid < item_group_num; local_wid += local_w_num){
        int base_item_group = local_wid * k;
        if (local_wid >= uncached_item_num){
            int converted_item_group = (local_wid - uncached_item_num) * k;
            grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val_s[converted_item_group + lane_id];
            grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val_s[converted_item_group + lane_id + 32];
            grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val_s[converted_item_group + lane_id + 64];
            grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val_s[converted_item_group + lane_id + 96];
        }
        // else {
        //     // grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val[base_item_group + lane_id];
        //     // grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val[base_item_group + lane_id + 32];
        //     // grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val[base_item_group + lane_id + 64];
        //     // grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val[base_item_group + lane_id + 96];
        //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id, item_group_sum_updated_val[base_item_group + lane_id]);
        //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 32, item_group_sum_updated_val[base_item_group + lane_id + 32]);
        //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 64, item_group_sum_updated_val[base_item_group + lane_id + 64]);
        //     atomicAdd(grad_sum_norm_q + base_item_group + lane_id + 96, item_group_sum_updated_val[base_item_group + lane_id + 96]);
        // }
    }

    if (threadIdx.x < user_group_num) {
        norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    }

    if (threadIdx.x < item_group_num) {
        norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_fp32_version(
                            const Node *R,
                            unsigned int nnz,
                            float** p,
                            float** q,
                            curandState *state,
                            float lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            float* grad_sum_norm_p,
                            float* grad_sum_norm_q,
                            float* norm_sum_p,
                            float* norm_sum_q,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
    }

    // 0으로 초기화
    __syncthreads();

    unsigned int processed_cnt = 0;

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            float r = __float2half_rn(__ldg(&R[offset].r));
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);

            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            //! both precisions are half
            float* tmp_p_ptr = (float*)p_s[user_group];
            float* tmp_q_ptr = (float*)q_s[item_group];

            const float tmp_p1 = tmp_p_ptr[base_p + lane_id];
            const float tmp_q1 = tmp_q_ptr[base_q + lane_id];
            const float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            const float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            const float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            const float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            const float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            const float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

            float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            const float ruv = r - tmp_product;
            
            //! Original version
            // const float tmp_p1_updated_val = lrate*(ruv*tmp_q1 - lambda*tmp_p1);
            // const float tmp_q1_updated_val = lrate*(ruv*tmp_p1 - lambda*tmp_q1);
            // const float tmp_p2_updated_val = lrate*(ruv*tmp_q2 - lambda*tmp_p2);
            // const float tmp_q2_updated_val = lrate*(ruv*tmp_p2 - lambda*tmp_q2);
            // const float tmp_p3_updated_val = lrate*(ruv*tmp_q3 - lambda*tmp_p3);
            // const float tmp_q3_updated_val = lrate*(ruv*tmp_p3 - lambda*tmp_q3);
            // const float tmp_p4_updated_val = lrate*(ruv*tmp_q4 - lambda*tmp_p4);
            // const float tmp_q4_updated_val = lrate*(ruv*tmp_p4 - lambda*tmp_q4);

            // const float tmp_p1_updated_val = (ruv*tmp_q1 - lambda*tmp_p1);
            // const float tmp_q1_updated_val = (ruv*tmp_p1 - lambda*tmp_q1);
            // const float tmp_p2_updated_val = (ruv*tmp_q2 - lambda*tmp_p2);
            // const float tmp_q2_updated_val = (ruv*tmp_p2 - lambda*tmp_q2);
            // const float tmp_p3_updated_val = (ruv*tmp_q3 - lambda*tmp_p3);
            // const float tmp_q3_updated_val = (ruv*tmp_p3 - lambda*tmp_q3);
            // const float tmp_p4_updated_val = (ruv*tmp_q4 - lambda*tmp_p4);
            // const float tmp_q4_updated_val = (ruv*tmp_p4 - lambda*tmp_q4);

            tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1);
            tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1);
            tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2);
            tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2);
            tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3);
            tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3);
            tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4);
            tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4);   

            // if (processed_cnt >= first_sample_rating_idx){
            //     const unsigned int user_group_base = user_group * k;
            //     const unsigned int item_group_base = item_group * k;
            //     float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
            //     float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                
            //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
            //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
            //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
            //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
            //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                
            //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
            //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
            //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
            //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
            //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

            //     if (lane_id == 0){
            //         user_group_sum_norms_s[user_group] += norm_p;
            //         item_group_sum_norms_s[item_group] += norm_q;
            //     }

            //     user_group_sum_updated_val_s[user_group_base + lane_id] += tmp_p1_updated_val;
            //     item_group_sum_updated_val_s[item_group_base + lane_id] += tmp_q1_updated_val;
            //     user_group_sum_updated_val_s[user_group_base + lane_id + 32] += tmp_p2_updated_val;
            //     item_group_sum_updated_val_s[item_group_base + lane_id + 32] += tmp_q2_updated_val;
            //     user_group_sum_updated_val_s[user_group_base + lane_id + 64] += tmp_p3_updated_val;
            //     item_group_sum_updated_val_s[item_group_base + lane_id + 64] += tmp_q3_updated_val;
            //     user_group_sum_updated_val_s[user_group_base + lane_id + 96] += tmp_p4_updated_val;
            //     item_group_sum_updated_val_s[item_group_base + lane_id + 96] += tmp_q4_updated_val;
            // }        
            // processed_cnt += 1;
        }
    }

    // int local_wid = threadIdx.x/32;
    // int local_w_num = blockDim.x/32;
    // int lane_id = threadIdx.x%32;

    // for (; local_wid < user_group_num; local_wid += local_w_num){
    //     int base_user_group = local_wid * k;

    //     grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val_s[base_user_group + lane_id];
    //     grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val_s[base_user_group + lane_id + 32];
    //     grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val_s[base_user_group + lane_id + 64];
    //     grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val_s[base_user_group + lane_id + 96];
    // }

    // local_wid = threadIdx.x/32;
    
    // for (; local_wid < item_group_num; local_wid += local_w_num){
    //     int base_item_group = local_wid * k;

    //     grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val_s[base_item_group + lane_id];
    //     grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val_s[base_item_group + lane_id + 32];
    //     grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val_s[base_item_group + lane_id + 64];
    //     grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val_s[base_item_group + lane_id + 96];
    // }

    // if (threadIdx.x < user_group_num) {
    //     norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
    // }

    // if (threadIdx.x < item_group_num) {
    //     norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
    // }
}

// __global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_error_grad_diversity_fp32_version(
//                             const Node *R,
//                             unsigned int nnz,
//                             float** p,
//                             float** q,
//                             curandState *state,
//                             float lrate,
//                             int k,
//                             int num_iters,
//                             int current_iter,
//                             int update_count_this_block,
//                             int update_vector_size,
//                             float lambda,
//                             Index_info_node* user_index_info,
//                             Index_info_node* item_index_info,
//                             float* grad_sum_norm_p,
//                             float* grad_sum_norm_q,
//                             float* norm_sum_p,
//                             float* norm_sum_q,
//                             unsigned int first_sample_rating_idx,
//                             unsigned int user_group_num,
//                             unsigned int item_group_num
//                             )
// {    
    
//     extern __shared__ float array[];
//     void** p_s = (void**)array;
//     void** q_s = (void**)&p_s[user_group_num];
//     // unsigned int* user_group_update_cnt_s = (unsigned int*)&q_s[item_group_num];
//     // unsigned int* item_group_update_cnt_s = (unsigned int*)&user_group_update_cnt_s[user_group_num];
//     // float* user_group_sum_updated_val_s = (float*)&item_group_update_cnt_s[item_group_num]; 
//     // float* item_group_sum_updated_val_s = (float*)&user_group_sum_updated_val_s[user_group_num * k];
//     // float* user_group_sum_norms_s = (float*)&item_group_sum_updated_val_s[item_group_num * k];
//     // float* item_group_sum_norms_s = (float*)&user_group_sum_norms_s[user_group_num];   
//         //    float* tmp_p_ptr = (float*)p[0];
//         //    float* tmp_q_ptr = (float*)q[0];
//     // int tid = threadIdx.x;

//     if (threadIdx.x < user_group_num){
//         p_s[threadIdx.x] = p[threadIdx.x];
//     //     // user_group_sum_norms_s[threadIdx.x] = 0.f;
//     //     // user_group_update_cnt_s[threadIdx.x] = 0;
//     }
//     if (threadIdx.x < item_group_num){
//         q_s[threadIdx.x] = q[threadIdx.x];
//     //     // item_group_sum_norms_s[threadIdx.x] = 0.f;
//     //     // item_group_update_cnt_s[threadIdx.x] = 0;
//     }

//     // for (;tid<user_group_num*128;tid+=blockDim.x) user_group_sum_updated_val_s[tid] = 0.f;
//     // tid = threadIdx.x;
//     // for (;tid<item_group_num*128;tid+=blockDim.x) item_group_sum_updated_val_s[tid] = 0.f;

//     // 0으로 초기화
//     __syncthreads();

//     unsigned int processed_cnt = 0;

//     for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
//     {
//         int lane_id = threadIdx.x%32;
//         int local_wid = threadIdx.x/32;
//         int local_w_num = blockDim.x/32;
//         int wid = local_w_num*blockIdx.x + local_wid;  
        
//         unsigned int start_id = 0;
//         if(lane_id == 0)
//         {
//             unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
//             start_id = origin%nnz;
//         }

//         start_id = __shfl_sync(0xffffffff,start_id, 0);
        
//         for(int i = 0;i < update_vector_size;i++)
//         {
//             int offset = (start_id + i)%nnz;
            
//             float r = __float2half_rn(__ldg(&R[offset].r));
//             int orig_u = __ldg(&R[offset].u);
//             int orig_v = __ldg(&R[offset].i);
            
//             // int user_group = 0;
//             // int item_group = 0;

//             int u = orig_u;
//             int v = orig_v;

//             // int user_group = user_index_info[orig_u].g;
//             // int u = user_index_info[orig_u].v;
//             // int item_group = item_index_info[orig_v].g;
//             // int v = item_index_info[orig_v].v;
//             int base_p = u*k;
//             int base_q = v*k;

//             //! both precisions are half
//             // float* tmp_p_ptr = (float*)p_s[user_group];
//             // float* tmp_q_ptr = (float*)q_s[item_group];
//             float* tmp_p_ptr = (float*)p_s[0];
//             float* tmp_q_ptr = (float*)q_s[0];
//             const float tmp_p1 = tmp_p_ptr[base_p + lane_id];
//             const float tmp_q1 = tmp_q_ptr[base_q + lane_id];
//             const float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//             const float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//             const float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//             const float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//             const float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//             const float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

//             float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
//             tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//             const float ruv = r - tmp_product;
            
//             //! Original version
//             // const float tmp_p1_updated_val = lrate*(ruv*tmp_q1 - lambda*tmp_p1);
//             // const float tmp_q1_updated_val = lrate*(ruv*tmp_p1 - lambda*tmp_q1);
//             // const float tmp_p2_updated_val = lrate*(ruv*tmp_q2 - lambda*tmp_p2);
//             // const float tmp_q2_updated_val = lrate*(ruv*tmp_p2 - lambda*tmp_q2);
//             // const float tmp_p3_updated_val = lrate*(ruv*tmp_q3 - lambda*tmp_p3);
//             // const float tmp_q3_updated_val = lrate*(ruv*tmp_p3 - lambda*tmp_q3);
//             // const float tmp_p4_updated_val = lrate*(ruv*tmp_q4 - lambda*tmp_p4);
//             // const float tmp_q4_updated_val = lrate*(ruv*tmp_p4 - lambda*tmp_q4);

//             // const float tmp_p1_updated_val = (ruv*tmp_q1 - lambda*tmp_p1);
//             // const float tmp_q1_updated_val = (ruv*tmp_p1 - lambda*tmp_q1);
//             // const float tmp_p2_updated_val = (ruv*tmp_q2 - lambda*tmp_p2);
//             // const float tmp_q2_updated_val = (ruv*tmp_p2 - lambda*tmp_q2);
//             // const float tmp_p3_updated_val = (ruv*tmp_q3 - lambda*tmp_p3);
//             // const float tmp_q3_updated_val = (ruv*tmp_p3 - lambda*tmp_q3);
//             // const float tmp_p4_updated_val = (ruv*tmp_q4 - lambda*tmp_p4);
//             // const float tmp_q4_updated_val = (ruv*tmp_p4 - lambda*tmp_q4);

//             tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1);
//             tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1);
//             tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2);
//             tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2);
//             tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3);
//             tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3);
//             tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4);
//             tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4);   

//             // if (processed_cnt >= first_sample_rating_idx){
//             //     const unsigned int user_group_base = user_group * k;
//             //     const unsigned int item_group_base = item_group * k;
//             //     float norm_p = ((tmp_p1_updated_val*tmp_p1_updated_val) + (tmp_p2_updated_val*tmp_p2_updated_val)) + ((tmp_p3_updated_val*tmp_p3_updated_val) + (tmp_p4_updated_val*tmp_p4_updated_val));
//             //     float norm_q = ((tmp_q1_updated_val*tmp_q1_updated_val) + (tmp_q2_updated_val*tmp_q2_updated_val)) + ((tmp_q3_updated_val*tmp_q3_updated_val) + (tmp_q4_updated_val*tmp_q4_updated_val));
                
//             //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
//             //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
//             //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
//             //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
//             //     norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);
                
//             //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
//             //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
//             //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
//             //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
//             //     norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);

//             //     if (lane_id == 0){
//             //         user_group_sum_norms_s[user_group] += norm_p;
//             //         item_group_sum_norms_s[item_group] += norm_q;
//             //     }

//             //     user_group_sum_updated_val_s[user_group_base + lane_id] += tmp_p1_updated_val;
//             //     item_group_sum_updated_val_s[item_group_base + lane_id] += tmp_q1_updated_val;
//             //     user_group_sum_updated_val_s[user_group_base + lane_id + 32] += tmp_p2_updated_val;
//             //     item_group_sum_updated_val_s[item_group_base + lane_id + 32] += tmp_q2_updated_val;
//             //     user_group_sum_updated_val_s[user_group_base + lane_id + 64] += tmp_p3_updated_val;
//             //     item_group_sum_updated_val_s[item_group_base + lane_id + 64] += tmp_q3_updated_val;
//             //     user_group_sum_updated_val_s[user_group_base + lane_id + 96] += tmp_p4_updated_val;
//             //     item_group_sum_updated_val_s[item_group_base + lane_id + 96] += tmp_q4_updated_val;
//             // }        
//             // processed_cnt += 1;
//         }
//     }

//     // int local_wid = threadIdx.x/32;
//     // int local_w_num = blockDim.x/32;
//     // int lane_id = threadIdx.x%32;

//     // for (; local_wid < user_group_num; local_wid += local_w_num){
//     //     int base_user_group = local_wid * k;

//     //     grad_sum_norm_p[base_user_group + lane_id] += user_group_sum_updated_val_s[base_user_group + lane_id];
//     //     grad_sum_norm_p[base_user_group + lane_id + 32] += user_group_sum_updated_val_s[base_user_group + lane_id + 32];
//     //     grad_sum_norm_p[base_user_group + lane_id + 64] += user_group_sum_updated_val_s[base_user_group + lane_id + 64];
//     //     grad_sum_norm_p[base_user_group + lane_id + 96] += user_group_sum_updated_val_s[base_user_group + lane_id + 96];
//     // }

//     // local_wid = threadIdx.x/32;
    
//     // for (; local_wid < item_group_num; local_wid += local_w_num){
//     //     int base_item_group = local_wid * k;

//     //     grad_sum_norm_q[base_item_group + lane_id] += item_group_sum_updated_val_s[base_item_group + lane_id];
//     //     grad_sum_norm_q[base_item_group + lane_id + 32] += item_group_sum_updated_val_s[base_item_group + lane_id + 32];
//     //     grad_sum_norm_q[base_item_group + lane_id + 64] += item_group_sum_updated_val_s[base_item_group + lane_id + 64];
//     //     grad_sum_norm_q[base_item_group + lane_id + 96] += item_group_sum_updated_val_s[base_item_group + lane_id + 96];
//     // }

//     // if (threadIdx.x < user_group_num) {
//     //     norm_sum_p[gridDim.x * threadIdx.x + blockIdx.x] = user_group_sum_norms_s[threadIdx.x];
//     // }

//     // if (threadIdx.x < item_group_num) {
//     //     norm_sum_q[gridDim.x * threadIdx.x + blockIdx.x] = item_group_sum_norms_s[threadIdx.x];
//     // }
// }
__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_division_based_indexing(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            unsigned int* acc_user_group_error,
                            unsigned int* acc_item_group_error,
                            unsigned int* user_group_update_cnt,
                            unsigned int* item_group_update_cnt,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_size,
                            unsigned int item_group_size,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    // __shared__ void* p_s[NUM_USER_GROUPS];
    // __shared__ void* q_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
    // __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_update_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_update_cnt_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_zero_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_zero_cnt_s[NUM_ITEM_GROUPS];

    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_update_cnt_s = (unsigned int*)&q_s[item_group_num];
    unsigned int* item_group_update_cnt_s = (unsigned int*)&user_group_update_cnt_s[user_group_num];
    unsigned int* user_group_zero_cnt_s = (unsigned int*)&item_group_update_cnt_s[item_group_num];
    unsigned int* item_group_zero_cnt_s = (unsigned int*)&user_group_zero_cnt_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_zero_cnt_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_zero_cnt_s[threadIdx.x] = 0;
        user_group_update_cnt_s[threadIdx.x] = 0;
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_zero_cnt_s[threadIdx.x] = 0;
        item_group_update_cnt_s[threadIdx.x] = 0;
    }
    __syncthreads();
    unsigned int processed_cnt = 0;
    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);
    __half zero = __float2half_rn(0.f);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = orig_u/user_group_size;
            int u = orig_u%user_group_size;
            int item_group = orig_v/item_group_size;
            int v = orig_v%item_group_size;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half
            if (!user_prec && !item_prec){
                unsigned int user_zero_cnt = 0;
                unsigned int item_zero_cnt = 0;
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;
                
                __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
                __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
                __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
                __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
                __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
                __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
                __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
                __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);
                
                if (processed_cnt >= first_sample_rating_idx){
                    if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;

                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
                        item_group_zero_cnt_s[item_group] += item_zero_cnt;
                        user_group_update_cnt_s[user_group] += 1;
                        item_group_update_cnt_s[item_group] += 1;
                    }
                }        
            }
            //! user half item single
            else if (!user_prec && item_prec){
                unsigned int user_zero_cnt = 0;
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];
                __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];

                __half tmp_q1 = __float2half_rn(tmp_q1_f);
                __half tmp_q2 = __float2half_rn(tmp_q2_f);
                __half tmp_q3 = __float2half_rn(tmp_q3_f);
                __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;

                __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
                __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
                __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
                __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
                
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
                
                if (processed_cnt >= first_sample_rating_idx){
                    if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
                    if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;

                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
                    user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
                        user_group_update_cnt_s[user_group] += 1;
                    }
                }        
            }
            //! user single item half
            else if (user_prec && !item_prec){
                unsigned int item_zero_cnt = 0;
                // if (lane_id == 0)
                // printf("%d ",processed_cnt);
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
                
                __half tmp_p1 = __float2half_rn(tmp_p1_f);
                __half tmp_p2 = __float2half_rn(tmp_p2_f);
                __half tmp_p3 = __float2half_rn(tmp_p3_f);
                __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                __half ruv = r - tmp_product;
                
                __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
                __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
                __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
                __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);    
                
                if (processed_cnt >= first_sample_rating_idx){
                    if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
                    if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;
                    
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
                    item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                    
                    if (lane_id == 0){
                        item_group_zero_cnt_s[item_group] += item_zero_cnt;
                        item_group_update_cnt_s[item_group] += 1;
                    }
                }     
            }
            //! user single item single
            else{
                // if (lane_id == 0)
                // printf("%d ",processed_cnt);
                float* tmp_p_ptr = (float*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                float tmp_p1 = tmp_p_ptr[base_p + lane_id];
                float tmp_q1 = tmp_q_ptr[base_q + lane_id];
                float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                float ruv = __half2float(r) - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            }

            processed_cnt += 1;
        }
    }

    if (threadIdx.x < user_group_num) {
        acc_user_group_error[gridDim.x * threadIdx.x + blockIdx.x] = user_group_zero_cnt_s[threadIdx.x];
        user_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = user_group_update_cnt_s[threadIdx.x];
    };
    if (threadIdx.x < item_group_num) {
        acc_item_group_error[gridDim.x * threadIdx.x + blockIdx.x] = item_group_zero_cnt_s[threadIdx.x];
        item_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = item_group_update_cnt_s[threadIdx.x];
    }
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_division_based_indexing_fp32_version(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            unsigned int* acc_user_group_error,
                            unsigned int* acc_item_group_error,
                            unsigned int* user_group_update_cnt,
                            unsigned int* item_group_update_cnt,
                            unsigned int first_sample_rating_idx,
                            unsigned int user_group_size,
                            unsigned int item_group_size,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    // __shared__ void* p_s[NUM_USER_GROUPS];
    // __shared__ void* q_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned char user_group_prec_s[NUM_USER_GROUPS];
    // __shared__ unsigned char item_group_prec_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_update_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_update_cnt_s[NUM_ITEM_GROUPS];
    // __shared__ unsigned int user_group_zero_cnt_s[NUM_USER_GROUPS];
    // __shared__ unsigned int item_group_zero_cnt_s[NUM_ITEM_GROUPS];

    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned int* user_group_update_cnt_s = (unsigned int*)&q_s[item_group_num];
    unsigned int* item_group_update_cnt_s = (unsigned int*)&user_group_update_cnt_s[user_group_num];
    unsigned int* user_group_zero_cnt_s = (unsigned int*)&item_group_update_cnt_s[item_group_num];
    unsigned int* item_group_zero_cnt_s = (unsigned int*)&user_group_zero_cnt_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&item_group_zero_cnt_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
        user_group_zero_cnt_s[threadIdx.x] = 0;
        user_group_update_cnt_s[threadIdx.x] = 0;
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
        item_group_zero_cnt_s[threadIdx.x] = 0;
        item_group_update_cnt_s[threadIdx.x] = 0;
    }
    __syncthreads();
    unsigned int processed_cnt = 0;

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            float r = __ldg(&R[offset].r);
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = orig_u/user_group_size;
            int u = orig_u%user_group_size;
            int item_group = orig_v/item_group_size;
            int v = orig_v%item_group_size;
            int base_p = u*k;
            int base_q = v*k;

            //! both precisions are half
            unsigned int user_zero_cnt = 0;
            unsigned int item_zero_cnt = 0;
            float* tmp_p_ptr = (float*)p_s[user_group];
            float* tmp_q_ptr = (float*)q_s[item_group];

            float tmp_p1 = tmp_p_ptr[base_p + lane_id];
            float tmp_q1 = tmp_q_ptr[base_q + lane_id];
            float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

            float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            float ruv = r - tmp_product;
            
            float tmp_p1_grad = (ruv*tmp_q1 - lambda*tmp_p1);
            float tmp_q1_grad = (ruv*tmp_p1 - lambda*tmp_q1);
            float tmp_p2_grad = (ruv*tmp_q2 - lambda*tmp_p2);
            float tmp_q2_grad = (ruv*tmp_p2 - lambda*tmp_q2);
            float tmp_p3_grad = (ruv*tmp_q3 - lambda*tmp_p3);
            float tmp_q3_grad = (ruv*tmp_p3 - lambda*tmp_q3);
            float tmp_p4_grad = (ruv*tmp_q4 - lambda*tmp_p4);
            float tmp_q4_grad = (ruv*tmp_p4 - lambda*tmp_q4);

            tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(tmp_p1_grad);
            tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(tmp_q1_grad);
            tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
            tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
            tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
            tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
            tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
            tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);
            
            // if (processed_cnt >= first_sample_rating_idx){
            //     if (lrate*tmp_p1_grad == zero) user_zero_cnt += 1;
            //     if (lrate*tmp_q1_grad == zero) item_zero_cnt += 1;
            //     if (lrate*tmp_p2_grad == zero) user_zero_cnt += 1;
            //     if (lrate*tmp_q2_grad == zero) item_zero_cnt += 1;
            //     if (lrate*tmp_p3_grad == zero) user_zero_cnt += 1;
            //     if (lrate*tmp_q3_grad == zero) item_zero_cnt += 1;
            //     if (lrate*tmp_p4_grad == zero) user_zero_cnt += 1;
            //     if (lrate*tmp_q4_grad == zero) item_zero_cnt += 1;

            //     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 16);
            //     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 8);
            //     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 4);
            //     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 2);
            //     user_zero_cnt += __shfl_down_sync(0xffffffff, user_zero_cnt, 1);
                
            //     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 16);
            //     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 8);
            //     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 4);
            //     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 2);
            //     item_zero_cnt += __shfl_down_sync(0xffffffff, item_zero_cnt, 1);
                
            //     if (lane_id == 0){
            //         user_group_zero_cnt_s[user_group] += user_zero_cnt;                   
            //         item_group_zero_cnt_s[item_group] += item_zero_cnt;
            //         user_group_update_cnt_s[user_group] += 1;
            //         item_group_update_cnt_s[item_group] += 1;
            //     }
            // }        
            

            processed_cnt += 1;
        }
    }

    if (threadIdx.x < user_group_num) {
        acc_user_group_error[gridDim.x * threadIdx.x + blockIdx.x] = user_group_zero_cnt_s[threadIdx.x];
        user_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = user_group_update_cnt_s[threadIdx.x];
    };
    if (threadIdx.x < item_group_num) {
        acc_item_group_error[gridDim.x * threadIdx.x + blockIdx.x] = item_group_zero_cnt_s[threadIdx.x];
        item_group_update_cnt[gridDim.x * threadIdx.x + blockIdx.x] = item_group_update_cnt_s[threadIdx.x];
    }
}
// #endif

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_fp16_grouped_eqaul_size_indexing_eval_eval_compute_precision(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            Index_info_node* user_index_info,
                            Index_info_node* item_index_info,
                            unsigned char* user_group_prec_info,
                            unsigned char* item_group_prec_info,
                            unsigned int user_group_num,
                            unsigned int item_group_num
                            )
{    
    extern __shared__ float array[];
    void** p_s = (void**)array;
    void** q_s = (void**)&p_s[user_group_num];
    unsigned char* user_group_prec_s = (unsigned char*)&q_s[item_group_num];
    unsigned char* item_group_prec_s = (unsigned char*)&user_group_prec_s[user_group_num];    

    int tid = threadIdx.x;

    if (threadIdx.x < user_group_num){
        p_s[threadIdx.x] = p[threadIdx.x];
        user_group_prec_s[threadIdx.x] = user_group_prec_info[threadIdx.x];
    }
    if (threadIdx.x < item_group_num){
        q_s[threadIdx.x] = q[threadIdx.x];
        item_group_prec_s[threadIdx.x] = item_group_prec_info[threadIdx.x];
    }

    __syncthreads();

    float lrate_f = __half2float(lrate);
    float lambda_f = __half2float(lambda);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
#ifdef HALF_COMP
            __half r = __float2half_rn(__ldg(&R[offset].r));
#endif
#ifdef SINGLE_COMP
            float r = __ldg(&R[offset].r);
#endif
            int orig_u = __ldg(&R[offset].u);
            int orig_v = __ldg(&R[offset].i);
            
            int user_group = user_index_info[orig_u].g;
            int u = user_index_info[orig_u].v;
            int item_group = item_index_info[orig_v].g;
            int v = item_index_info[orig_v].v;
            int base_p = u*k;
            int base_q = v*k;

            unsigned char user_prec = user_group_prec_s[user_group];
            unsigned char item_prec = item_group_prec_s[item_group];

            //! both precisions are half

            //! user half item single
            if (!user_prec && item_prec){
                __half* tmp_p_ptr = (__half*)p_s[user_group];
                float* tmp_q_ptr = (float*)q_s[item_group];

                const __half tmp_p1 = tmp_p_ptr[base_p + lane_id];
                const float tmp_q1_f = tmp_q_ptr[base_q + lane_id];
                const __half tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
                const float tmp_q2_f = tmp_q_ptr[base_q + lane_id + 32];
                const __half tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
                const float tmp_q3_f = tmp_q_ptr[base_q + lane_id + 64];
                const __half tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
                const float tmp_q4_f = tmp_q_ptr[base_q + lane_id + 96];

#ifdef HALF_COMP
                //! Converting to half
                const __half tmp_q1 = __float2half_rn(tmp_q1_f);
                const __half tmp_q2 = __float2half_rn(tmp_q2_f);
                const __half tmp_q3 = __float2half_rn(tmp_q3_f);
                const __half tmp_q4 = __float2half_rn(tmp_q4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;
                
                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*__half2float(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*__half2float(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*__half2float(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*__half2float(ruv*tmp_p4 - lambda*tmp_q4);  
#endif
#ifdef SINGLE_COMP
                //! Converting to single
                const float tmp_p1_f = __half2float(tmp_p1);
                const float tmp_p2_f = __half2float(tmp_p2);
                const float tmp_p3_f = __half2float(tmp_p3);
                const float tmp_p4_f = __half2float(tmp_p4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate*__float2half_rn(ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = tmp_q1_f + lrate_f*(ruv*tmp_p1_f - lambda_f*tmp_q1_f);

                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate*__float2half_rn(ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2_f + lrate_f*(ruv*tmp_p2_f - lambda_f*tmp_q2_f);

                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate*__float2half_rn(ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3_f + lrate_f*(ruv*tmp_p3_f - lambda_f*tmp_q3_f);

                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate*__float2half_rn(ruv*tmp_q4_f - lambda_f*tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4_f + lrate_f*(ruv*tmp_p4_f - lambda_f*tmp_q4_f);
#endif
            }
            //! user single item half
            else if (user_prec && !item_prec){
                float* tmp_p_ptr = (float*)p_s[user_group];
                __half* tmp_q_ptr = (__half*)q_s[item_group];

                const float tmp_p1_f = tmp_p_ptr[base_p + lane_id];
                const __half tmp_q1 = tmp_q_ptr[base_q + lane_id];
                const float tmp_p2_f = tmp_p_ptr[base_p + lane_id + 32];
                const __half tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
                const float tmp_p3_f = tmp_p_ptr[base_p + lane_id + 64];
                const __half tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
                const float tmp_p4_f = tmp_p_ptr[base_p + lane_id + 96];
                const __half tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

#ifdef HALF_COMP
                //! Converting to half
                const __half tmp_p1 = __float2half_rn(tmp_p1_f);
                const __half tmp_p2 = __float2half_rn(tmp_p2_f);
                const __half tmp_p3 = __float2half_rn(tmp_p3_f);
                const __half tmp_p4 = __float2half_rn(tmp_p4_f);

                __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const __half ruv = r - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*__half2float(ruv*tmp_q1 - lambda*tmp_p1);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*__half2float(ruv*tmp_q2 - lambda*tmp_p2);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*__half2float(ruv*tmp_q3 - lambda*tmp_p3);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*__half2float(ruv*tmp_q4 - lambda*tmp_p4);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4);  
#endif
#ifdef SINGLE_COMP
                //! Converting to single
                const float tmp_q1_f = __half2float(tmp_q1);
                const float tmp_q2_f = __half2float(tmp_q2);
                const float tmp_q3_f = __half2float(tmp_q3);
                const float tmp_q4_f = __half2float(tmp_q4);

                float tmp_product = ((tmp_p1_f*tmp_q1_f) + (tmp_p2_f*tmp_q2_f)) + ((tmp_p3_f*tmp_q3_f) + (tmp_p4_f*tmp_q4_f));

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                const float ruv = r - tmp_product;

                tmp_p_ptr[base_p + lane_id] = tmp_p1_f + lrate_f*(ruv*tmp_q1_f - lambda_f*tmp_p1_f);
                tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate*__float2half_rn(ruv*tmp_p1_f - lambda_f*tmp_q1_f);
                tmp_p_ptr[base_p + lane_id + 32] = tmp_p2_f + lrate_f*(ruv*tmp_q2_f - lambda_f*tmp_p2_f);
                tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate*__float2half_rn(ruv*tmp_p2_f - lambda_f*tmp_q2_f);
                tmp_p_ptr[base_p + lane_id + 64] = tmp_p3_f + lrate_f*(ruv*tmp_q3_f - lambda_f*tmp_p3_f);
                tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate*__float2half_rn(ruv*tmp_p3_f - lambda_f*tmp_q3_f);
                tmp_p_ptr[base_p + lane_id + 96] = tmp_p4_f + lrate_f*(ruv*tmp_q4_f - lambda_f*tmp_p4_f);
                tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate*__float2half_rn(ruv*tmp_p4_f - lambda_f*tmp_q4_f);  
#endif
            }
            //! user single item single
            // else{
            //     float* tmp_p_ptr = (float*)p_s[user_group];
            //     float* tmp_q_ptr = (float*)q_s[item_group];

            //     float tmp_p1 = tmp_p_ptr[base_p + lane_id];
            //     float tmp_q1 = tmp_q_ptr[base_q + lane_id];
            //     float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            //     float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            //     float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            //     float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            //     float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            //     float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];

            //     float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            //     tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            //     tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            //     tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            //     tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            //     tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
            //     tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            //     float ruv = __half2float(r) - tmp_product;

            //     tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(ruv*tmp_q1 - lambda_f*tmp_p1);
            //     tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(ruv*tmp_p1 - lambda_f*tmp_q1);

            //     tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(ruv*tmp_q2 - lambda_f*tmp_p2);
            //     tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(ruv*tmp_p2 - lambda_f*tmp_q2);

            //     tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(ruv*tmp_q3 - lambda_f*tmp_p3);
            //     tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(ruv*tmp_p3 - lambda_f*tmp_q3);

            //     tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(ruv*tmp_q4 - lambda_f*tmp_p4);
            //     tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(ruv*tmp_p4 - lambda_f*tmp_q4);
            // }
        }
    }
}

__global__ void rev_origin_sgd_k128_kernel_hogwild_warp32_lrate_loss_scaling(
                            const Node *R,
                            unsigned int nnz,
                            float *p,
                            float *q,
                            curandState *state,
                            float lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            float scaling_factor
                            )
{    
    float scaling_factor_rev = 1/scaling_factor;
    __half scaling_factor_half = __float2half_rn(scaling_factor);

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {
        int lane_id = threadIdx.x%32;
        int local_wid = threadIdx.x/32;
        int local_w_num = blockDim.x/32;
        int wid = local_w_num*blockIdx.x + local_wid;  
        
        unsigned int start_id = 0;
        if(lane_id == 0)
        {
            unsigned int origin = (unsigned int)(curand_uniform(&state[wid])*nnz);
            start_id = origin%nnz;
        }

        // All threads read x from laneid 0
        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            //read the p & q into register file.
            int base_p = u*k;
            int base_q = v*k;

            float tmp_p1_f = (p[base_p + lane_id]);
            float tmp_q1_f = (q[base_q + lane_id]);

            float tmp_p2_f = (p[base_p + lane_id + 32]);
            float tmp_q2_f = (q[base_q + lane_id + 32]);

            float tmp_p3_f = (p[base_p + lane_id + 64]);
            float tmp_q3_f = (q[base_q + lane_id + 64]);

            float tmp_p4_f = (p[base_p + lane_id + 96]);
            float tmp_q4_f = (q[base_q + lane_id + 96]);
            
            __half tmp_p1 = __float2half(tmp_p1_f);
            __half tmp_q1 = __float2half(tmp_q1_f);

            __half tmp_p2 = __float2half(tmp_p2_f);
            __half tmp_q2 = __float2half(tmp_q2_f);

            __half tmp_p3 = __float2half(tmp_p3_f);
            __half tmp_q3 = __float2half(tmp_q3_f);

            __half tmp_p4 = __float2half(tmp_p4_f);
            __half tmp_q4 = __float2half(tmp_q4_f);

            __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            
            __half ruv = scaling_factor_half * (r-tmp_product);

            p[base_p + lane_id +  0] = tmp_p1_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_q1 - lambda*tmp_p1)));
            q[base_q + lane_id +  0] = tmp_q1_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_p1 - lambda*tmp_q1)));
            
            p[base_p + lane_id + 32] = tmp_p2_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_q2 - lambda*tmp_p2)));
            q[base_q + lane_id + 32] = tmp_q2_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_p2 - lambda*tmp_q2)));
            
            p[base_p + lane_id + 64] = tmp_p3_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_q3 - lambda*tmp_p3)));
            q[base_q + lane_id + 64] = tmp_q3_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_p3 - lambda*tmp_q3)));
            
            p[base_p + lane_id + 96] = tmp_p4_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_q4 - lambda*tmp_p4)));
            q[base_q + lane_id + 96] = tmp_q4_f + (lrate * (scaling_factor_rev * __half2float(ruv*tmp_p4 - lambda*tmp_q4)));
        }    
    }
}