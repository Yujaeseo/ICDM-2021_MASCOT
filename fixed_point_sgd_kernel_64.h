__global__ void sgd_k64_kernel_hogwild_warp32_lrate_grad_diversity_muppet(
                            const Node *R,
                            unsigned int nnz,
                            float* p,
                            float* q,
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
                            float* sum_norms,
                            unsigned char forward_prop_precision,
                            unsigned char backward_prop_precision
                            )
{
    float scaling_factor_p;
    float scaling_factor_q;
    float scaling_factor_back_prop;
    unsigned int scaling_factor_p_index;
    unsigned int scaling_factor_q_index;
    unsigned int back_prop_index;
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
            
            float* tmp_p_ptr = (float*)p;
            float* tmp_q_ptr = (float*)q;

            float tmp_p1 = tmp_p_ptr[base_p + lane_id];
            float tmp_q1 = tmp_q_ptr[base_q + lane_id];
            float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
            float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
            
            float p_min_val;
            float p_max_val;
            float q_min_val;
            float q_max_val;
            float pred;

            if (forward_prop_precision != (unsigned char)32){
                p_min_val = fminf(tmp_p1, tmp_p2);
                p_max_val = fmaxf(tmp_p1, tmp_p2);
                q_min_val = fminf(tmp_q1, tmp_q2);
                q_max_val = fmaxf(tmp_q1, tmp_q2);
                
                p_min_val = reduce_min_input(p_min_val);
                q_min_val = reduce_min_input(q_min_val);
                p_max_val = reduce_max_input(p_max_val);
                q_max_val = reduce_max_input(q_max_val);

                scaling_factor_p_index = get_only_scaling_factor(forward_prop_precision, p_max_val, p_min_val);
                scaling_factor_q_index = get_only_scaling_factor(forward_prop_precision, q_max_val, q_min_val);

                scaling_factor_p = exp2f(scaling_factor_p_index);
                scaling_factor_q = exp2f(scaling_factor_q_index);
            }

            if (forward_prop_precision == (unsigned char)8){
                char4 p_int8_4 = make_char4(roundf(tmp_p1 * scaling_factor_p),
                                            roundf(tmp_p2 * scaling_factor_p),
                                            0,
                                            0);
                char4 q_int8_4 = make_char4(roundf(tmp_q1 * scaling_factor_q),
                                            roundf(tmp_q2 * scaling_factor_q),
                                            0,
                                            0);

                int tmp_product = 0;
                tmp_product = __dp4a(p_int8_4, q_int8_4, tmp_product);
                
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                pred = tmp_product / (scaling_factor_p * scaling_factor_q);

            }else if (forward_prop_precision == (unsigned char)32){
                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2));
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                pred = __shfl_sync(0xffffffff,tmp_product,0);
            }else{
                __half2 p_half2_1, q_half2_1;
                float scaling_factor_p_rev = 1/scaling_factor_p;
                float scaling_factor_q_rev = 1/scaling_factor_q;
                //! 왜 half로 하면 안될지 생각해보기
                // p_half2_1.x = hrint(__float2half(tmp_p1 * scaling_factor_p));
                // p_half2_1.y = hrint(__float2half(tmp_p2 * scaling_factor_p));
                // p_half2_2.x = hrint(__float2half(tmp_p3 * scaling_factor_p));
                // p_half2_2.y = hrint(__float2half(tmp_p4 * scaling_factor_p));
                // q_half2_1.x = hrint(__float2half(tmp_q1 * scaling_factor_q));
                // q_half2_1.y = hrint(__float2half(tmp_q2 * scaling_factor_q));
                // q_half2_2.x = hrint(__float2half(tmp_q3 * scaling_factor_q));
                // q_half2_2.y = hrint(__float2half(tmp_q4 * scaling_factor_q));  
                // p_half2_1 = __h2div(p_half2_1,__half2half2(__float2half(scaling_factor_p)));
                // p_half2_2 = __h2div(p_half2_2,__half2half2(__float2half(scaling_factor_p)));
                // q_half2_1 = __h2div(q_half2_1,__half2half2(__float2half(scaling_factor_q)));
                // q_half2_2 = __h2div(q_half2_2,__half2half2(__float2half(scaling_factor_q)));

                p_half2_1.x = __float2half(roundf(tmp_p1 * scaling_factor_p)*scaling_factor_p_rev);
                p_half2_1.y = __float2half(roundf(tmp_p2 * scaling_factor_p)*scaling_factor_p_rev);
                // p_half2_2.x = __float2half(roundf(tmp_p3 * scaling_factor_p)*scaling_factor_p_rev);
                // p_half2_2.y = __float2half(roundf(tmp_p4 * scaling_factor_p)*scaling_factor_p_rev);
                q_half2_1.x = __float2half(roundf(tmp_q1 * scaling_factor_q)*scaling_factor_q_rev);
                q_half2_1.y = __float2half(roundf(tmp_q2 * scaling_factor_q)*scaling_factor_q_rev);
                // q_half2_2.x = __float2half(roundf(tmp_q3 * scaling_factor_q)*scaling_factor_q_rev);
                // q_half2_2.y = __float2half(roundf(tmp_q4 * scaling_factor_q)*scaling_factor_q_rev);
                // p_half2_1 = __h2div(p_half2_1,__half2half2(__float2half(scaling_factor_p)));
                // p_half2_2 = __h2div(p_half2_2,__half2half2(__float2half(scaling_factor_p)));
                // q_half2_1 = __h2div(q_half2_1,__half2half2(__float2half(scaling_factor_q)));
                // q_half2_2 = __h2div(q_half2_2,__half2half2(__float2half(scaling_factor_q)));
                __half2 tmp_product = __hmul2(p_half2_1,q_half2_1);

                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);

                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                pred = __half2float(tmp_product.x+tmp_product.y);             
            }

            float ruv = r - pred;
            //! Back propagation fixed point 연산 어떻게 구현하는지 muppet 코드 찾아보기
            float back_prop_min;
            float back_prop_max;

            if (backward_prop_precision != (unsigned char)32){
                back_prop_min = fminf(fminf(p_min_val, q_min_val), fminf(-1*lambda_f, ruv));
                back_prop_max = fmaxf(fmaxf(p_max_val, q_max_val), fmaxf(-1*lambda_f, ruv));
                back_prop_index = get_only_scaling_factor(backward_prop_precision, back_prop_max, back_prop_min);
                scaling_factor_back_prop = exp2f(back_prop_index);
            }

            float p1_grad_val;
            float q1_grad_val;
            float p2_grad_val;
            float q2_grad_val;

            if (backward_prop_precision == (unsigned char)8){
                float scaling_factor_square_rev = 1/(scaling_factor_back_prop*scaling_factor_back_prop);
                char4 int_val2 = make_char4(roundf(ruv * scaling_factor_back_prop),
                                            roundf(-1.0f*lambda_f * scaling_factor_back_prop),
                                            0,
                                            0);

                p1_grad_val = comp_grad_int8(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
                q1_grad_val = comp_grad_int8(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
                p2_grad_val = comp_grad_int8(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
                q2_grad_val = comp_grad_int8(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);

                // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(p1_grad_val);
                // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(q1_grad_val);
                // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(p2_grad_val);
                // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(q2_grad_val);
                // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(p3_grad_val);
                // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(q3_grad_val);
                // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(p4_grad_val);
                // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(q4_grad_val);

            }else if(backward_prop_precision == (unsigned char)32){
                p1_grad_val = (ruv*tmp_q1 - lambda_f*tmp_p1);
                q1_grad_val = (ruv*tmp_p1 - lambda_f*tmp_q1);
                p2_grad_val = (ruv*tmp_q2 - lambda_f*tmp_p2);
                q2_grad_val = (ruv*tmp_p2 - lambda_f*tmp_q2);
            }else{
                // __half2 half2_tmp2;
                // half2_tmp2.x = hrint(__float2half(ruv * scaling_factor_back_prop));
                // half2_tmp2.y = hrint(__float2half(-1*lambda_f * scaling_factor_back_prop));
                // half2_tmp2 = __h2div(half2_tmp2,__half2half2(__float2half(scaling_factor_back_prop)));

                // p1_grad_val = comp_grad_fp16(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, half2_tmp2);
                // q1_grad_val = comp_grad_fp16(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, half2_tmp2);
                // p2_grad_val = comp_grad_fp16(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, half2_tmp2);
                // q2_grad_val = comp_grad_fp16(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, half2_tmp2);
                // p3_grad_val = comp_grad_fp16(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, half2_tmp2);
                // q3_grad_val = comp_grad_fp16(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, half2_tmp2);
                // p4_grad_val = comp_grad_fp16(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, half2_tmp2);
                // q4_grad_val = comp_grad_fp16(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, half2_tmp2);

                __half2 half2_tmp2;
                float scaling_factor_back_prop_rev = 1/scaling_factor_back_prop;
                half2_tmp2.x = __float2half(roundf(ruv * scaling_factor_back_prop)*scaling_factor_back_prop_rev);
                half2_tmp2.y = __float2half(roundf(-1*lambda_f * scaling_factor_back_prop)*scaling_factor_back_prop_rev);
                // half2_tmp2 = __h2div(half2_tmp2,__half2half2(__float2half(scaling_factor_back_prop)));

                p1_grad_val = comp_grad_fp16(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
                q1_grad_val = comp_grad_fp16(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
                p2_grad_val = comp_grad_fp16(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
                q2_grad_val = comp_grad_fp16(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
            }

            tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(p1_grad_val);
            tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(q1_grad_val);
            tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(p2_grad_val);
            tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(q2_grad_val);
            
            if (forward_prop_precision != (unsigned char)32 && backward_prop_precision != (unsigned char)32 && processed_cnt >= first_sample_rating_idx){
                // float sum_norms_reg = 0;
                float norm_p = ((p1_grad_val*p1_grad_val) + (p2_grad_val*p2_grad_val));
                float norm_q = ((q1_grad_val*q1_grad_val) + (q2_grad_val*q2_grad_val));
                float norm_pq = norm_p + norm_q; 
        
                norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 16);
                norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 8);
                norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 4);
                norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 2);
                norm_pq += __shfl_down_sync(0xffffffff, norm_pq, 1);
                
                if (lane_id == 0){
                    sum_norms_reg += norm_pq;
                }

                atomicAdd(sum_updated_val + lane_id, p1_grad_val + q1_grad_val);
                atomicAdd(sum_updated_val + lane_id + 32, p2_grad_val + q2_grad_val);
            }

            processed_cnt += 1;
        }    
    }

    if (lane_id == 0) sum_norms[wid] = sum_norms_reg;
}