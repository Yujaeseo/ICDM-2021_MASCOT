__forceinline__ __device__ float reduce_min_input(float input){
    float tmp_min = input;
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 16));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 8));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 4));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 2));
    tmp_min = fminf(tmp_min,__shfl_down_sync(0xffffffff, tmp_min, 1));
    tmp_min = __shfl_sync(0xffffffff,tmp_min,0);

    return tmp_min;
}

__forceinline__ __device__ float reduce_max_input(float input){
    float tmp_max = input;
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 16));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 8));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 4));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 2));
    tmp_max = fmaxf(tmp_max,__shfl_down_sync(0xffffffff, tmp_max, 1));
    tmp_max = __shfl_sync(0xffffffff,tmp_max,0);

    return tmp_max;
}

__forceinline__ __device__ float stochastic_round(float val, curandState* state){
    int wid = (threadIdx.x%32);
    float added_val = curand_uniform(&state[wid])-0.5f;
    return roundf(val + added_val);
}


//! ICML
__forceinline__ __device__ float get_only_scaling_factor(unsigned int bitwidth, float max_scaled_val, float min_scaled_val){
    if (max_scaled_val == 0 && min_scaled_val == 0)
        return 0;
    
    float val = 1 << (bitwidth-1);
    float max_val = (val - 1) + 0.5f;
    float min_val = (-val) - 0.5f; 

    // float max_abs_val = fmaxf(fabsf(max_scaled_val), fabsf(min_scaled_val));
    // float scaling_factor = fabsf(ceilf(log2f(max_abs_val/(exp2f((bitwidth-1))-1))));
    float range_best = fminf(fabsf(max_val/max_scaled_val), fabsf(min_val/min_scaled_val));
    float scaling_factor = floorf(log2f(range_best));
    // printf("%f ",scaling_factor);
    return scaling_factor;
}


// __forceinline__  __device__ float comp_grad_int8(float f_val1, float f_val2, float ruv, float lambda, float scaling_factor){
//     char4 int_val1 = make_char4((int)nearbyintf(f_val1 * scaling_factor),
//                                 (int)nearbyintf(f_val2 * scaling_factor),
//                                 0,
//                                 0);
//     char4 int_val2 = make_char4((int)nearbyintf(ruv * scaling_factor),
//                                 (int)nearbyintf(-1.0f*lambda * scaling_factor),
//                                 0,
//                                 0);
//     // printf("%f ", (updated_val)/(scaling_factor * scaling_factor));
//     return (__dp4a(int_val1, int_val2, 0))/(scaling_factor * scaling_factor);
// }

__forceinline__  __device__ float comp_grad_int8(float f_val1, float f_val2, float ruv, float scaling_factor, float scaling_factor_square_rev ,char4 int_val2){
    char4 int_val1 = make_char4(roundf(f_val1 * scaling_factor),
                                roundf(f_val2 * scaling_factor),
                                0,
                                0);
    // char4 int_val2 = make_char4((int)nearbyintf(ruv * scaling_factor),
    //                             (int)nearbyintf(-1.0f*lambda * scaling_factor),
    //                             0,
    //                             0);
    // printf("%f ", (updated_val)/(scaling_factor * scaling_factor));
    return (__dp4a(int_val1, int_val2, 0))*scaling_factor_square_rev;
}

__forceinline__ __device__ float comp_grad_fp16(float f_val1, float f_val2, float ruv, float scaling_factor, float scaling_factor_rev ,__half2 half2_tmp2){
    // __half2 half2_tmp1;
    // half2_tmp1.x = hrint(__float2half(f_val1 * scaling_factor));
    // half2_tmp1.y = hrint(__float2half(f_val2 * scaling_factor));
    __half2 half2_tmp1;
    half2_tmp1.x = __float2half(roundf(f_val1 * scaling_factor)*scaling_factor_rev);
    half2_tmp1.y = __float2half(roundf(f_val2 * scaling_factor)*scaling_factor_rev);
    // half2_tmp1 = __h2div(half2_tmp1,__half2half2(__float2half(scaling_factor)));

    __half2 result = __hmul2(half2_tmp1, half2_tmp2);
    return __half2float(result.x + result.y);
}

__forceinline__  __device__ float comp_grad_int8_stochastic_rounding(float f_val1, float f_val2, float ruv, float scaling_factor, float scaling_factor_square_rev ,char4 int_val2, curandState* state){
    char4 int_val1 = make_char4(stochastic_round(f_val1 * scaling_factor, state),
                                stochastic_round(f_val2 * scaling_factor, state),
                                0,
                                0);
    return (__dp4a(int_val1, int_val2, 0))*scaling_factor_square_rev;
}

__forceinline__ __device__ float comp_grad_fp16_stochastic_rounding(float f_val1, float f_val2, float ruv, float scaling_factor, float scaling_factor_rev, __half2 half2_tmp2, curandState* state){
    __half2 half2_tmp1;
    half2_tmp1.x = __float2half(stochastic_round(f_val1 * scaling_factor, state)*scaling_factor_rev);
    half2_tmp1.y = __float2half(stochastic_round(f_val2 * scaling_factor, state)*scaling_factor_rev);
    __half2 result = __hmul2(half2_tmp1, half2_tmp2);
    return __half2float(result.x + result.y);
}

//! CVPR
// __device__ float get_only_scaling_factor(unsigned int bitwidth, float max_scaled_val, float min_scaled_val){
//     if (max_scaled_val == 0 && min_scaled_val == 0)
//         return 0;
    
//     float val = 1 << (bitwidth-1);
//     // float max_val = (val - 1) + 0.5f;
//     // float min_val = (-val) - 0.5f; 

//     float max_abs_val = fmaxf(fabsf(max_scaled_val), fabsf(min_scaled_val));
//     float scaling_factor = fabsf(ceilf(log2f(max_abs_val/(exp2f((bitwidth-1))-1))));
//     // printf("%f ",scaling_factor);
//     return scaling_factor;
// }

//! Final version of muppet 
__global__ void sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet(
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
            float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
            
            float p_min_val;
            float p_max_val;
            float q_min_val;
            float q_max_val;
            float pred;

            if (forward_prop_precision != (unsigned char)32){
                p_min_val = fminf(fminf(tmp_p1, tmp_p2), fminf(tmp_p3, tmp_p4));
                p_max_val = fmaxf(fmaxf(tmp_p1, tmp_p2), fmaxf(tmp_p3, tmp_p4));
                q_min_val = fminf(fminf(tmp_q1, tmp_q2), fminf(tmp_q3, tmp_q4));
                q_max_val = fmaxf(fmaxf(tmp_q1, tmp_q2), fmaxf(tmp_q3, tmp_q4));
                
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
                                            roundf(tmp_p3 * scaling_factor_p),
                                            roundf(tmp_p4 * scaling_factor_p));
                char4 q_int8_4 = make_char4(roundf(tmp_q1 * scaling_factor_q),
                                            roundf(tmp_q2 * scaling_factor_q),
                                            roundf(tmp_q3 * scaling_factor_q),
                                            roundf(tmp_q4 * scaling_factor_q));

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
                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                pred = __shfl_sync(0xffffffff,tmp_product,0);
            }else{
                __half2 p_half2_1, p_half2_2, q_half2_1, q_half2_2;
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
                p_half2_2.x = __float2half(roundf(tmp_p3 * scaling_factor_p)*scaling_factor_p_rev);
                p_half2_2.y = __float2half(roundf(tmp_p4 * scaling_factor_p)*scaling_factor_p_rev);
                q_half2_1.x = __float2half(roundf(tmp_q1 * scaling_factor_q)*scaling_factor_q_rev);
                q_half2_1.y = __float2half(roundf(tmp_q2 * scaling_factor_q)*scaling_factor_q_rev);
                q_half2_2.x = __float2half(roundf(tmp_q3 * scaling_factor_q)*scaling_factor_q_rev);
                q_half2_2.y = __float2half(roundf(tmp_q4 * scaling_factor_q)*scaling_factor_q_rev);
                // p_half2_1 = __h2div(p_half2_1,__half2half2(__float2half(scaling_factor_p)));
                // p_half2_2 = __h2div(p_half2_2,__half2half2(__float2half(scaling_factor_p)));
                // q_half2_1 = __h2div(q_half2_1,__half2half2(__float2half(scaling_factor_q)));
                // q_half2_2 = __h2div(q_half2_2,__half2half2(__float2half(scaling_factor_q)));
                __half2 tmp_product = __hmul2(p_half2_1,q_half2_1) + __hmul2(p_half2_2,q_half2_2);

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
            float p3_grad_val;
            float q3_grad_val;
            float p4_grad_val;
            float q4_grad_val;

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
                p3_grad_val = comp_grad_int8(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
                q3_grad_val = comp_grad_int8(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
                p4_grad_val = comp_grad_int8(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
                q4_grad_val = comp_grad_int8(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);

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
                p3_grad_val = (ruv*tmp_q3 - lambda_f*tmp_p3);
                q3_grad_val = (ruv*tmp_p3 - lambda_f*tmp_q3);
                p4_grad_val = (ruv*tmp_q4 - lambda_f*tmp_p4);
                q4_grad_val = (ruv*tmp_p4 - lambda_f*tmp_q4);
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
                p3_grad_val = comp_grad_fp16(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
                q3_grad_val = comp_grad_fp16(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
                p4_grad_val = comp_grad_fp16(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
                q4_grad_val = comp_grad_fp16(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
            }

            tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(p1_grad_val);
            tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(q1_grad_val);
            tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(p2_grad_val);
            tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(q2_grad_val);
            tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(p3_grad_val);
            tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(q3_grad_val);
            tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(p4_grad_val);
            tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(q4_grad_val);
            
            if (forward_prop_precision != (unsigned char)32 && backward_prop_precision != (unsigned char)32 && processed_cnt >= first_sample_rating_idx){
                // float sum_norms_reg = 0;
                float norm_p = ((p1_grad_val*p1_grad_val) + (p2_grad_val*p2_grad_val)) + ((p3_grad_val*p3_grad_val) + (p4_grad_val*p4_grad_val));
                float norm_q = ((q1_grad_val*q1_grad_val) + (q2_grad_val*q2_grad_val)) + ((q3_grad_val*q3_grad_val) + (q4_grad_val*q4_grad_val));
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
                atomicAdd(sum_updated_val + lane_id + 64, p3_grad_val + q3_grad_val);
                atomicAdd(sum_updated_val + lane_id + 96, p4_grad_val + q4_grad_val);
            }

            processed_cnt += 1;
        }    
    }

    if (lane_id == 0) sum_norms[wid] = sum_norms_reg;
}

//! Average of user,item gradient diversity version 
// __global__ void sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet(
//                             const Node *R,
//                             unsigned int nnz,
//                             float* p,
//                             float* q,
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
//                             float* sum_norms_item,
//                             unsigned char forward_prop_precision,
//                             unsigned char backward_prop_precision
//                             )
// {
//     float scaling_factor_p;
//     float scaling_factor_q;
//     float scaling_factor_back_prop;
//     unsigned int scaling_factor_p_index;
//     unsigned int scaling_factor_q_index;
//     unsigned int back_prop_index;
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
            
//             float* tmp_p_ptr = (float*)p;
//             float* tmp_q_ptr = (float*)q;

//             float tmp_p1 = tmp_p_ptr[base_p + lane_id];
//             float tmp_q1 = tmp_q_ptr[base_q + lane_id];
//             float tmp_p2 = tmp_p_ptr[base_p + lane_id + 32];
//             float tmp_q2 = tmp_q_ptr[base_q + lane_id + 32];
//             float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
//             float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
//             float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
//             float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
            
//             float p_min_val;
//             float p_max_val;
//             float q_min_val;
//             float q_max_val;
//             float pred;

//             if (forward_prop_precision != (unsigned char)32){
//                 p_min_val = fminf(fminf(tmp_p1, tmp_p2), fminf(tmp_p3, tmp_p4));
//                 p_max_val = fmaxf(fmaxf(tmp_p1, tmp_p2), fmaxf(tmp_p3, tmp_p4));
//                 q_min_val = fminf(fminf(tmp_q1, tmp_q2), fminf(tmp_q3, tmp_q4));
//                 q_max_val = fmaxf(fmaxf(tmp_q1, tmp_q2), fmaxf(tmp_q3, tmp_q4));
                
//                 p_min_val = reduce_min_input(p_min_val);
//                 q_min_val = reduce_min_input(q_min_val);
//                 p_max_val = reduce_max_input(p_max_val);
//                 q_max_val = reduce_max_input(q_max_val);

//                 scaling_factor_p_index = get_only_scaling_factor(forward_prop_precision, p_max_val, p_min_val);
//                 scaling_factor_q_index = get_only_scaling_factor(forward_prop_precision, q_max_val, q_min_val);

//                 scaling_factor_p = exp2f(scaling_factor_p_index);
//                 scaling_factor_q = exp2f(scaling_factor_q_index);
//             }

//             if (forward_prop_precision == (unsigned char)8){
//                 char4 p_int8_4 = make_char4(roundf(tmp_p1 * scaling_factor_p),
//                                             roundf(tmp_p2 * scaling_factor_p),
//                                             roundf(tmp_p3 * scaling_factor_p),
//                                             roundf(tmp_p4 * scaling_factor_p));
//                 char4 q_int8_4 = make_char4(roundf(tmp_q1 * scaling_factor_q),
//                                             roundf(tmp_q2 * scaling_factor_q),
//                                             roundf(tmp_q3 * scaling_factor_q),
//                                             roundf(tmp_q4 * scaling_factor_q));

//                 int tmp_product = 0;
//                 tmp_product = __dp4a(p_int8_4, q_int8_4, tmp_product);
                
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 pred = tmp_product / (scaling_factor_p * scaling_factor_q);

//             }else if (forward_prop_precision == (unsigned char)32){
//                 float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
//                 pred = __shfl_sync(0xffffffff,tmp_product,0);
//             }else{
//                 __half2 p_half2_1, p_half2_2, q_half2_1, q_half2_2;
//                 float scaling_factor_p_rev = 1/scaling_factor_p;
//                 float scaling_factor_q_rev = 1/scaling_factor_q;
//                 //! 왜 half로 하면 안될지 생각해보기
//                 // p_half2_1.x = hrint(__float2half(tmp_p1 * scaling_factor_p));
//                 // p_half2_1.y = hrint(__float2half(tmp_p2 * scaling_factor_p));
//                 // p_half2_2.x = hrint(__float2half(tmp_p3 * scaling_factor_p));
//                 // p_half2_2.y = hrint(__float2half(tmp_p4 * scaling_factor_p));
//                 // q_half2_1.x = hrint(__float2half(tmp_q1 * scaling_factor_q));
//                 // q_half2_1.y = hrint(__float2half(tmp_q2 * scaling_factor_q));
//                 // q_half2_2.x = hrint(__float2half(tmp_q3 * scaling_factor_q));
//                 // q_half2_2.y = hrint(__float2half(tmp_q4 * scaling_factor_q));  
//                 // p_half2_1 = __h2div(p_half2_1,__half2half2(__float2half(scaling_factor_p)));
//                 // p_half2_2 = __h2div(p_half2_2,__half2half2(__float2half(scaling_factor_p)));
//                 // q_half2_1 = __h2div(q_half2_1,__half2half2(__float2half(scaling_factor_q)));
//                 // q_half2_2 = __h2div(q_half2_2,__half2half2(__float2half(scaling_factor_q)));

//                 p_half2_1.x = __float2half(roundf(tmp_p1 * scaling_factor_p)*scaling_factor_p_rev);
//                 p_half2_1.y = __float2half(roundf(tmp_p2 * scaling_factor_p)*scaling_factor_p_rev);
//                 p_half2_2.x = __float2half(roundf(tmp_p3 * scaling_factor_p)*scaling_factor_p_rev);
//                 p_half2_2.y = __float2half(roundf(tmp_p4 * scaling_factor_p)*scaling_factor_p_rev);
//                 q_half2_1.x = __float2half(roundf(tmp_q1 * scaling_factor_q)*scaling_factor_q_rev);
//                 q_half2_1.y = __float2half(roundf(tmp_q2 * scaling_factor_q)*scaling_factor_q_rev);
//                 q_half2_2.x = __float2half(roundf(tmp_q3 * scaling_factor_q)*scaling_factor_q_rev);
//                 q_half2_2.y = __float2half(roundf(tmp_q4 * scaling_factor_q)*scaling_factor_q_rev);
//                 // p_half2_1 = __h2div(p_half2_1,__half2half2(__float2half(scaling_factor_p)));
//                 // p_half2_2 = __h2div(p_half2_2,__half2half2(__float2half(scaling_factor_p)));
//                 // q_half2_1 = __h2div(q_half2_1,__half2half2(__float2half(scaling_factor_q)));
//                 // q_half2_2 = __h2div(q_half2_2,__half2half2(__float2half(scaling_factor_q)));
//                 __half2 tmp_product = __hmul2(p_half2_1,q_half2_1) + __hmul2(p_half2_2,q_half2_2);

//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
//                 tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);

//                 tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
//                 pred = __half2float(tmp_product.x+tmp_product.y);             
//             }

//             float ruv = r - pred;
//             //! Back propagation fixed point 연산 어떻게 구현하는지 muppet 코드 찾아보기
//             float back_prop_min;
//             float back_prop_max;

//             if (backward_prop_precision != (unsigned char)32){
//                 back_prop_min = fminf(fminf(p_min_val, q_min_val), fminf(-1*lambda_f, ruv));
//                 back_prop_max = fmaxf(fmaxf(p_max_val, q_max_val), fmaxf(-1*lambda_f, ruv));
//                 back_prop_index = get_only_scaling_factor(backward_prop_precision, back_prop_max, back_prop_min);
//                 scaling_factor_back_prop = exp2f(back_prop_index);
//             }

//             float p1_grad_val;
//             float q1_grad_val;
//             float p2_grad_val;
//             float q2_grad_val;
//             float p3_grad_val;
//             float q3_grad_val;
//             float p4_grad_val;
//             float q4_grad_val;

//             if (backward_prop_precision == (unsigned char)8){
//                 float scaling_factor_square_rev = 1/(scaling_factor_back_prop*scaling_factor_back_prop);
//                 char4 int_val2 = make_char4(roundf(ruv * scaling_factor_back_prop),
//                                             roundf(-1.0f*lambda_f * scaling_factor_back_prop),
//                                             0,
//                                             0);

//                 p1_grad_val = comp_grad_int8(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
//                 q1_grad_val = comp_grad_int8(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
//                 p2_grad_val = comp_grad_int8(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
//                 q2_grad_val = comp_grad_int8(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
//                 p3_grad_val = comp_grad_int8(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
//                 q3_grad_val = comp_grad_int8(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
//                 p4_grad_val = comp_grad_int8(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);
//                 q4_grad_val = comp_grad_int8(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2);

//                 // tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(p1_grad_val);
//                 // tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(q1_grad_val);
//                 // tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(p2_grad_val);
//                 // tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(q2_grad_val);
//                 // tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(p3_grad_val);
//                 // tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(q3_grad_val);
//                 // tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(p4_grad_val);
//                 // tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(q4_grad_val);

//             }else if(backward_prop_precision == (unsigned char)32){
//                 p1_grad_val = (ruv*tmp_q1 - lambda_f*tmp_p1);
//                 q1_grad_val = (ruv*tmp_p1 - lambda_f*tmp_q1);
//                 p2_grad_val = (ruv*tmp_q2 - lambda_f*tmp_p2);
//                 q2_grad_val = (ruv*tmp_p2 - lambda_f*tmp_q2);
//                 p3_grad_val = (ruv*tmp_q3 - lambda_f*tmp_p3);
//                 q3_grad_val = (ruv*tmp_p3 - lambda_f*tmp_q3);
//                 p4_grad_val = (ruv*tmp_q4 - lambda_f*tmp_p4);
//                 q4_grad_val = (ruv*tmp_p4 - lambda_f*tmp_q4);
//             }else{
//                 // __half2 half2_tmp2;
//                 // half2_tmp2.x = hrint(__float2half(ruv * scaling_factor_back_prop));
//                 // half2_tmp2.y = hrint(__float2half(-1*lambda_f * scaling_factor_back_prop));
//                 // half2_tmp2 = __h2div(half2_tmp2,__half2half2(__float2half(scaling_factor_back_prop)));

//                 // p1_grad_val = comp_grad_fp16(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, half2_tmp2);
//                 // q1_grad_val = comp_grad_fp16(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, half2_tmp2);
//                 // p2_grad_val = comp_grad_fp16(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, half2_tmp2);
//                 // q2_grad_val = comp_grad_fp16(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, half2_tmp2);
//                 // p3_grad_val = comp_grad_fp16(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, half2_tmp2);
//                 // q3_grad_val = comp_grad_fp16(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, half2_tmp2);
//                 // p4_grad_val = comp_grad_fp16(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, half2_tmp2);
//                 // q4_grad_val = comp_grad_fp16(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, half2_tmp2);

//                 __half2 half2_tmp2;
//                 float scaling_factor_back_prop_rev = 1/scaling_factor_back_prop;
//                 half2_tmp2.x = __float2half(roundf(ruv * scaling_factor_back_prop)*scaling_factor_back_prop_rev);
//                 half2_tmp2.y = __float2half(roundf(-1*lambda_f * scaling_factor_back_prop)*scaling_factor_back_prop_rev);
//                 // half2_tmp2 = __h2div(half2_tmp2,__half2half2(__float2half(scaling_factor_back_prop)));

//                 p1_grad_val = comp_grad_fp16(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//                 q1_grad_val = comp_grad_fp16(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//                 p2_grad_val = comp_grad_fp16(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//                 q2_grad_val = comp_grad_fp16(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//                 p3_grad_val = comp_grad_fp16(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//                 q3_grad_val = comp_grad_fp16(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//                 p4_grad_val = comp_grad_fp16(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//                 q4_grad_val = comp_grad_fp16(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev, half2_tmp2);
//             }

//             tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(p1_grad_val);
//             tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(q1_grad_val);
//             tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(p2_grad_val);
//             tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(q2_grad_val);
//             tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(p3_grad_val);
//             tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(q3_grad_val);
//             tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(p4_grad_val);
//             tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(q4_grad_val);
            
//             if (forward_prop_precision != 32 && backward_prop_precision != 32 && processed_cnt >= first_sample_rating_idx){
//                 // float sum_norms_reg = 0;
//                 float norm_p = ((p1_grad_val*p1_grad_val) + (p2_grad_val*p2_grad_val)) + ((p3_grad_val*p3_grad_val) + (p4_grad_val*p4_grad_val));
//                 float norm_q = ((q1_grad_val*q1_grad_val) + (q2_grad_val*q2_grad_val)) + ((q3_grad_val*q3_grad_val) + (q4_grad_val*q4_grad_val));
        
//                 norm_p += __shfl_down_sync(0xffffffff, norm_p, 16);
//                 norm_p += __shfl_down_sync(0xffffffff, norm_p, 8);
//                 norm_p += __shfl_down_sync(0xffffffff, norm_p, 4);
//                 norm_p += __shfl_down_sync(0xffffffff, norm_p, 2);
//                 norm_p += __shfl_down_sync(0xffffffff, norm_p, 1);

//                 norm_q += __shfl_down_sync(0xffffffff, norm_q, 16);
//                 norm_q += __shfl_down_sync(0xffffffff, norm_q, 8);
//                 norm_q += __shfl_down_sync(0xffffffff, norm_q, 4);
//                 norm_q += __shfl_down_sync(0xffffffff, norm_q, 2);
//                 norm_q += __shfl_down_sync(0xffffffff, norm_q, 1);
                
//                 if (lane_id == 0){
//                     sum_norms_reg += norm_p;
//                     sum_norms_reg_item += norm_q;
//                 }

//                 atomicAdd(sum_updated_val + lane_id, p1_grad_val);
//                 atomicAdd(sum_updated_val + lane_id + 32, p2_grad_val);
//                 atomicAdd(sum_updated_val + lane_id + 64, p3_grad_val);
//                 atomicAdd(sum_updated_val + lane_id + 96, p4_grad_val);

//                 atomicAdd(sum_updated_val_item + lane_id, q1_grad_val);
//                 atomicAdd(sum_updated_val_item + lane_id + 32, q2_grad_val);
//                 atomicAdd(sum_updated_val_item + lane_id + 64, q3_grad_val);
//                 atomicAdd(sum_updated_val_item + lane_id + 96, q4_grad_val);


//             }

//             processed_cnt += 1;
//         }    
//     }

//     if (lane_id == 0) {
//         sum_norms[wid] = sum_norms_reg;
//         sum_norms_item[wid] = sum_norms_reg_item;
//     }
// }

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet_stochastic_rounding(
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
            float tmp_p3 = tmp_p_ptr[base_p + lane_id + 64];
            float tmp_q3 = tmp_q_ptr[base_q + lane_id + 64];
            float tmp_p4 = tmp_p_ptr[base_p + lane_id + 96];
            float tmp_q4 = tmp_q_ptr[base_q + lane_id + 96];
            
            float p_min_val;
            float p_max_val;
            float q_min_val;
            float q_max_val;
            float pred;

            if (forward_prop_precision != (unsigned char)32){
                p_min_val = fminf(fminf(tmp_p1, tmp_p2), fminf(tmp_p3, tmp_p4));
                p_max_val = fmaxf(fmaxf(tmp_p1, tmp_p2), fmaxf(tmp_p3, tmp_p4));
                q_min_val = fminf(fminf(tmp_q1, tmp_q2), fminf(tmp_q3, tmp_q4));
                q_max_val = fmaxf(fmaxf(tmp_q1, tmp_q2), fmaxf(tmp_q3, tmp_q4));
                
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
                char4 p_int8_4 = make_char4(stochastic_round(tmp_p1 * scaling_factor_p, state),
                                            stochastic_round(tmp_p2 * scaling_factor_p, state),
                                            stochastic_round(tmp_p3 * scaling_factor_p, state),
                                            stochastic_round(tmp_p4 * scaling_factor_p, state));
                char4 q_int8_4 = make_char4(stochastic_round(tmp_q1 * scaling_factor_q, state),
                                            stochastic_round(tmp_q2 * scaling_factor_q, state),
                                            stochastic_round(tmp_q3 * scaling_factor_q, state),
                                            stochastic_round(tmp_q4 * scaling_factor_q, state));
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
                float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                pred = __shfl_sync(0xffffffff,tmp_product,0);
            }else{
                __half2 p_half2_1, p_half2_2, q_half2_1, q_half2_2;
                float scaling_factor_p_rev = 1/scaling_factor_p;
                float scaling_factor_q_rev = 1/scaling_factor_q;

                p_half2_1.x = __float2half(stochastic_round(tmp_p1 * scaling_factor_p, state)*scaling_factor_p_rev);
                p_half2_1.y = __float2half(stochastic_round(tmp_p2 * scaling_factor_p, state)*scaling_factor_p_rev);
                p_half2_2.x = __float2half(stochastic_round(tmp_p3 * scaling_factor_p, state)*scaling_factor_p_rev);
                p_half2_2.y = __float2half(stochastic_round(tmp_p4 * scaling_factor_p, state)*scaling_factor_p_rev);
                q_half2_1.x = __float2half(stochastic_round(tmp_q1 * scaling_factor_q, state)*scaling_factor_q_rev);
                q_half2_1.y = __float2half(stochastic_round(tmp_q2 * scaling_factor_q, state)*scaling_factor_q_rev);
                q_half2_2.x = __float2half(stochastic_round(tmp_q3 * scaling_factor_q, state)*scaling_factor_q_rev);
                q_half2_2.y = __float2half(stochastic_round(tmp_q4 * scaling_factor_q, state)*scaling_factor_q_rev);
                __half2 tmp_product = __hmul2(p_half2_1,q_half2_1) + __hmul2(p_half2_2,q_half2_2);

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
            float p3_grad_val;
            float q3_grad_val;
            float p4_grad_val;
            float q4_grad_val;

            if (backward_prop_precision == (unsigned char)8){
                float scaling_factor_square_rev = 1/(scaling_factor_back_prop*scaling_factor_back_prop);
                char4 int_val2 = make_char4(stochastic_round(ruv * scaling_factor_back_prop, state),
                                            stochastic_round(-1.0f*lambda_f * scaling_factor_back_prop, state),
                                            0,
                                            0);

                p1_grad_val = comp_grad_int8_stochastic_rounding(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);
                q1_grad_val = comp_grad_int8_stochastic_rounding(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);
                p2_grad_val = comp_grad_int8_stochastic_rounding(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);
                q2_grad_val = comp_grad_int8_stochastic_rounding(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);
                p3_grad_val = comp_grad_int8_stochastic_rounding(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);
                q3_grad_val = comp_grad_int8_stochastic_rounding(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);
                p4_grad_val = comp_grad_int8_stochastic_rounding(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);
                q4_grad_val = comp_grad_int8_stochastic_rounding(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, scaling_factor_square_rev, int_val2, state);

            }else if(backward_prop_precision == (unsigned char)32){
                p1_grad_val = (ruv*tmp_q1 - lambda_f*tmp_p1);
                q1_grad_val = (ruv*tmp_p1 - lambda_f*tmp_q1);
                p2_grad_val = (ruv*tmp_q2 - lambda_f*tmp_p2);
                q2_grad_val = (ruv*tmp_p2 - lambda_f*tmp_q2);
                p3_grad_val = (ruv*tmp_q3 - lambda_f*tmp_p3);
                q3_grad_val = (ruv*tmp_p3 - lambda_f*tmp_q3);
                p4_grad_val = (ruv*tmp_q4 - lambda_f*tmp_p4);
                q4_grad_val = (ruv*tmp_p4 - lambda_f*tmp_q4);
            }else{

                __half2 half2_tmp2;
                float scaling_factor_back_prop_rev = 1/scaling_factor_back_prop;
                half2_tmp2.x = __float2half(stochastic_round(ruv * scaling_factor_back_prop, state)*scaling_factor_back_prop_rev);
                half2_tmp2.y = __float2half(stochastic_round(-1*lambda_f * scaling_factor_back_prop, state)*scaling_factor_back_prop_rev);

                p1_grad_val = comp_grad_fp16_stochastic_rounding(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
                q1_grad_val = comp_grad_fp16_stochastic_rounding(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
                p2_grad_val = comp_grad_fp16_stochastic_rounding(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
                q2_grad_val = comp_grad_fp16_stochastic_rounding(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
                p3_grad_val = comp_grad_fp16_stochastic_rounding(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
                q3_grad_val = comp_grad_fp16_stochastic_rounding(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
                p4_grad_val = comp_grad_fp16_stochastic_rounding(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
                q4_grad_val = comp_grad_fp16_stochastic_rounding(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, scaling_factor_back_prop_rev ,half2_tmp2, state);
            }

            tmp_p_ptr[base_p + lane_id] = tmp_p1 + lrate_f*(p1_grad_val);
            tmp_q_ptr[base_q + lane_id] = tmp_q1 + lrate_f*(q1_grad_val);
            tmp_p_ptr[base_p + lane_id + 32] = tmp_p2 + lrate_f*(p2_grad_val);
            tmp_q_ptr[base_q + lane_id + 32] = tmp_q2 + lrate_f*(q2_grad_val);
            tmp_p_ptr[base_p + lane_id + 64] = tmp_p3 + lrate_f*(p3_grad_val);
            tmp_q_ptr[base_q + lane_id + 64] = tmp_q3 + lrate_f*(q3_grad_val);
            tmp_p_ptr[base_p + lane_id + 96] = tmp_p4 + lrate_f*(p4_grad_val);
            tmp_q_ptr[base_q + lane_id + 96] = tmp_q4 + lrate_f*(q4_grad_val);
            
            if (forward_prop_precision != 32 && backward_prop_precision != 32 && processed_cnt >= first_sample_rating_idx){
                float norm_p = ((p1_grad_val*p1_grad_val) + (p2_grad_val*p2_grad_val)) + ((p3_grad_val*p3_grad_val) + (p4_grad_val*p4_grad_val));
                float norm_q = ((q1_grad_val*q1_grad_val) + (q2_grad_val*q2_grad_val)) + ((q3_grad_val*q3_grad_val) + (q4_grad_val*q4_grad_val));
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
                atomicAdd(sum_updated_val + lane_id + 64, p3_grad_val + q3_grad_val);
                atomicAdd(sum_updated_val + lane_id + 96, p4_grad_val + q4_grad_val);
            }

            processed_cnt += 1;
        }    
    }
    if (lane_id == 0) sum_norms[wid] = sum_norms_reg;
}

__global__ void sgd_k128_kernel_hogwild_warp32_lrate_grad_diversity_muppet_parameter_precision_switching(
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
            if (cur_precision == (unsigned char)16){
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