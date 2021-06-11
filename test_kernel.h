__global__ void half_sgd_k128_kernel_hogwild_warp32_lrate_grad_mean_user_item(
                            const Node *R,
                            unsigned int nnz,
                            __half *p,
                            __half *q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            float* grad_mean_array_user,
                            float* grad_mean_array_item,
                            unsigned int* update_cnt_array_user,
                            unsigned int* update_cnt_array_item,
                            unsigned int first_sample_rating_idx
                            )
{    
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
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            int base_p = u*k;
            int base_q = v*k;

            __half tmp_p1 = p[base_p + lane_id];
            __half tmp_q1 = q[base_q + lane_id];

            __half tmp_p2 = p[base_p + lane_id + 32];
            __half tmp_q2 = q[base_q + lane_id + 32];

            __half tmp_p3 = p[base_p + lane_id + 64];
            __half tmp_q3 = q[base_q + lane_id + 64];

            __half tmp_p4 = p[base_p + lane_id + 96];
            __half tmp_q4 = q[base_q + lane_id + 96];

            __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);

            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);

            __half ruv = r-tmp_product;

            __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
            __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
            __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
            __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
            __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
            __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
            __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
            __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;

            float tmp_p_sum = fabs(__half2float(tmp_p1_grad)) + fabs(__half2float(tmp_p2_grad)) + fabs(__half2float(tmp_p3_grad)) + fabs(__half2float(tmp_p4_grad));
            float tmp_q_sum = fabs(__half2float(tmp_q1_grad)) + fabs(__half2float(tmp_q2_grad)) + fabs(__half2float(tmp_q3_grad)) + fabs(__half2float(tmp_q4_grad));
            
            tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 16);
            tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 8);
            tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 4);
            tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 2);
            tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 1);
            
            tmp_p_sum = __shfl_sync(0xffffffff,tmp_p_sum,0);
            
            if (processed_cnt >= first_sample_rating_idx && lane_id == 0){
                // atomicAdd(grad_mean_array_user + u, (tmp_p_sum));
                // atomicAdd(grad_mean_array_item + v, (tmp_q_sum));
                // atomicAdd(update_cnt_array_user + u, 1);
                // atomicAdd(update_cnt_array_item + v, 1);

                grad_mean_array_user[u] += tmp_p_sum;
                grad_mean_array_item[v] += tmp_q_sum;
                update_cnt_array_user[u] += 1;
                update_cnt_array_item[v] += 1;
            }

            tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 16);
            tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 8);
            tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 4);
            tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 2);
            tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 1);
            
            tmp_q_sum = __shfl_sync(0xffffffff,tmp_q_sum,0);

            p[base_p + lane_id +  0] = tmp_p1 + lrate*(tmp_p1_grad);
            q[base_q + lane_id +  0] = tmp_q1 + lrate*(tmp_q1_grad);
            p[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
            q[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
            p[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
            q[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
            p[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
            q[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);

            if (lane_id == 0) processed_cnt+=1;
        }    
    }
}

// __global__ void half_sgd_k128_kernel_hogwild_warp32_lrate_grad_zero_user_item(
//                             const Node *R,
//                             unsigned int nnz,
//                             __half *p,
//                             __half *q,
//                             curandState *state,
//                             __half lrate,
//                             int k,
//                             int num_iters,
//                             int current_iter,
//                             int update_count_this_block,
//                             int update_vector_size,
//                             __half lambda,
//                             float* grad_mean_array_user,
//                             float* grad_mean_array_item,
//                             unsigned int* update_cnt_array_user,
//                             unsigned int* update_cnt_array_item,
//                             unsigned int first_sample_rating_idx
//                             )
// {    
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
            
//             __half r = __float2half_rn(__ldg(&R[offset].r));
//             int u = __ldg(&R[offset].u);
//             int v = __ldg(&R[offset].i);

//             int base_p = u*k;
//             int base_q = v*k;

//             __half tmp_p1 = p[base_p + lane_id];
//             __half tmp_q1 = q[base_q + lane_id];

//             __half tmp_p2 = p[base_p + lane_id + 32];
//             __half tmp_q2 = q[base_q + lane_id + 32];

//             __half tmp_p3 = p[base_p + lane_id + 64];
//             __half tmp_q3 = q[base_q + lane_id + 64];

//             __half tmp_p4 = p[base_p + lane_id + 96];
//             __half tmp_q4 = q[base_q + lane_id + 96];

//             __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//             tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);

//             tmp_product = __shfl_sync(0xffffffff,tmp_product,0);

//             __half ruv = r-tmp_product;

//             __half tmp_p1_grad = ruv*tmp_q1 - lambda*tmp_p1;
//             __half tmp_q1_grad = ruv*tmp_p1 - lambda*tmp_q1;
//             __half tmp_p2_grad = ruv*tmp_q2 - lambda*tmp_p2;
//             __half tmp_q2_grad = ruv*tmp_p2 - lambda*tmp_q2;
//             __half tmp_p3_grad = ruv*tmp_q3 - lambda*tmp_p3;
//             __half tmp_q3_grad = ruv*tmp_p3 - lambda*tmp_q3;
//             __half tmp_p4_grad = ruv*tmp_q4 - lambda*tmp_p4;
//             __half tmp_q4_grad = ruv*tmp_p4 - lambda*tmp_q4;

//             float tmp_p_sum = fabs(__half2float(tmp_p1_grad)) + fabs(__half2float(tmp_p2_grad)) + fabs(__half2float(tmp_p3_grad)) + fabs(__half2float(tmp_p4_grad));
//             float tmp_q_sum = fabs(__half2float(tmp_q1_grad)) + fabs(__half2float(tmp_q2_grad)) + fabs(__half2float(tmp_q3_grad)) + fabs(__half2float(tmp_q4_grad));
            
//             tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 16);
//             tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 8);
//             tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 4);
//             tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 2);
//             tmp_p_sum += __shfl_down_sync(0xffffffff, tmp_p_sum, 1);
            
//             tmp_p_sum = __shfl_sync(0xffffffff,tmp_p_sum,0);
            
//             if (processed_cnt >= first_sample_rating_idx && lane_id == 0){
//                 // atomicAdd(grad_mean_array_user + u, (tmp_p_sum));
//                 // atomicAdd(grad_mean_array_item + v, (tmp_q_sum));
//                 // atomicAdd(update_cnt_array_user + u, 1);
//                 // atomicAdd(update_cnt_array_item + v, 1);

//                 grad_mean_array_user[u] += tmp_p_sum;
//                 grad_mean_array_item[v] += tmp_q_sum;
//                 update_cnt_array_user[u] += 1;
//                 update_cnt_array_item[v] += 1;
//             }

//             tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 16);
//             tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 8);
//             tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 4);
//             tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 2);
//             tmp_q_sum += __shfl_down_sync(0xffffffff, tmp_q_sum, 1);
            
//             tmp_q_sum = __shfl_sync(0xffffffff,tmp_q_sum,0);

//             p[base_p + lane_id +  0] = tmp_p1 + lrate*(tmp_p1_grad);
//             q[base_q + lane_id +  0] = tmp_q1 + lrate*(tmp_q1_grad);
//             p[base_p + lane_id + 32] = tmp_p2 + lrate*(tmp_p2_grad);
//             q[base_q + lane_id + 32] = tmp_q2 + lrate*(tmp_q2_grad);
//             p[base_p + lane_id + 64] = tmp_p3 + lrate*(tmp_p3_grad);
//             q[base_q + lane_id + 64] = tmp_q3 + lrate*(tmp_q3_grad);
//             p[base_p + lane_id + 96] = tmp_p4 + lrate*(tmp_p4_grad);
//             q[base_q + lane_id + 96] = tmp_q4 + lrate*(tmp_q4_grad);

//             if (lane_id == 0) processed_cnt+=1;
//         }    
//     }
// }

__global__ void half_sgd_k128_kernel_hogwild_warp32_lrate(
                            const Node *R,
                            unsigned int nnz,
                            __half *p,
                            __half *q,
                            curandState *state,
                            __half lrate,
                            int k,
                            int num_iters,
                            int current_iter,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda
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
            // unsigned int origin = (unsigned int) update_ite * update_vector_size;

            start_id = origin%nnz;
        }

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            int base_p = u*k;
            int base_q = v*k;

            __half tmp_p1 = p[base_p + lane_id];
            __half tmp_q1 = q[base_q + lane_id];

            __half tmp_p2 = p[base_p + lane_id + 32];
            __half tmp_q2 = q[base_q + lane_id + 32];

            __half tmp_p3 = p[base_p + lane_id + 64];
            __half tmp_q3 = q[base_q + lane_id + 64];

            __half tmp_p4 = p[base_p + lane_id + 96];
            __half tmp_q4 = q[base_q + lane_id + 96];


            __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);

            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);

            __half ruv = r-tmp_product;

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