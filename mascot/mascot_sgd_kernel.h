__global__ void mascot_sgd_k128_hogwild_kernel(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
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

                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;

                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
                }        
            }
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
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 96, tmp_p4_updated_val);
                }        
            }
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
                    
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 96, tmp_q4_updated_val);
                }     
            }
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
}


__global__ void naive_mascot_sgd_k128_hogwild_kernel(
                            const Node *R,
                            unsigned int nnz,
                            void** p,
                            void** q,
                            curandState *state,
                            float lrate_f,
                            int k,
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
 
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id, tmp_p1_updated_val_f);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 32, tmp_p2_updated_val_f);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 64, tmp_p3_updated_val_f);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 96, tmp_p4_updated_val_f);

                    item_group_base = item_group * k;

                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id, tmp_q1_updated_val_f);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 32, tmp_q2_updated_val_f);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 64, tmp_q3_updated_val_f);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 96, tmp_q4_updated_val_f);
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
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id, tmp_p1_updated_val);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 32, tmp_p2_updated_val);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 64, tmp_p3_updated_val);
                    atomicAdd(grad_sum_norm_p + user_group_base + lane_id + 96, tmp_p4_updated_val);
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
                    
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id, tmp_q1_updated_val);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 32, tmp_q2_updated_val);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 64, tmp_q3_updated_val);
                    atomicAdd(grad_sum_norm_q + item_group_base + lane_id + 96, tmp_q4_updated_val);
                }     
            }
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
}