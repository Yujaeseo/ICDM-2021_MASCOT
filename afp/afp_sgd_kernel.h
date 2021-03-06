struct Quantization_params{
    int r;
    int n;
    int itv;
    float mov_avg_range;
};

__forceinline__ __device__ float get_only_scaling_factor_cvpr(unsigned int bitwidth, float max_scaled_val){
    if (max_scaled_val == 0)
        return 0;
    
    float val = 1 << (bitwidth-1);
    float scaling_factor = fabsf(ceilf(log2f(max_scaled_val/(exp2f((bitwidth-1))-1))));

    return scaling_factor;
}

__forceinline__  __device__ float comp_grad_int8_cvpr(char f_val1, char f_val2, float scaling_factor ,char4 int_val2){
    char4 int_val1 = make_char4(f_val1, f_val2, 0, 0);
    return (__dp4a(int_val1, int_val2, 0))*(scaling_factor);
}

__forceinline__ __device__ float comp_grad_fp16_cvpr(float f_val1, float f_val2, float ruv, float scaling_factor, float scaling_factor_rev ,__half2 half2_tmp2){
    __half2 half2_tmp1;
    half2_tmp1.x = __float2half(roundf(f_val1 * scaling_factor)*scaling_factor_rev);
    half2_tmp1.y = __float2half(roundf(f_val2 * scaling_factor)*scaling_factor_rev);
    __half2 result = __hmul2(half2_tmp1, half2_tmp2);

    return __half2float(result.x + result.y);
}

__forceinline__ __device__ float QEM(float val1, float val2, float val3, float val4, float max_val){    
    unsigned int scaling_factor_index = get_only_scaling_factor_cvpr(8, max_val);
    float scaling_factor = exp2f(scaling_factor_index);
    float scaling_factor_rev = 1/scaling_factor;

    float val1_q = roundf(val1 * scaling_factor)*scaling_factor_rev;
    float val2_q = roundf(val2 * scaling_factor)*scaling_factor_rev;
    float val3_q = roundf(val3 * scaling_factor)*scaling_factor_rev;
    float val4_q = roundf(val4 * scaling_factor)*scaling_factor_rev;
    
    float val_sum_q = fabsf(val1_q) + fabsf(val2_q) + fabsf(val3_q) + fabsf(val4_q); 
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 16);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 8);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 4);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 2);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 1);
    val_sum_q = __shfl_sync(0xffffffff,val_sum_q,0);

    float val_sum = fabsf(val1) + fabsf(val2) + fabsf(val3) + fabsf(val4);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 16);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 8);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 4);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 2);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 1);
    val_sum = __shfl_sync(0xffffffff,val_sum,0);

    return log2f(fabsf((val_sum - val_sum_q)/val_sum) + 1);
}

__forceinline__ __device__ float QEM_BP(float val1, float val2, float val3, float val4,
                                        float val5, float val6, float val7, float val8,
                                        float val9, float val10, float max_val){    
    unsigned int scaling_factor_index = get_only_scaling_factor_cvpr(8, max_val);
    float scaling_factor = exp2f(scaling_factor_index);
    float scaling_factor_rev = 1/scaling_factor;

    float val1_q = roundf(val1 * scaling_factor)*scaling_factor_rev;
    float val2_q = roundf(val2 * scaling_factor)*scaling_factor_rev;
    float val3_q = roundf(val3 * scaling_factor)*scaling_factor_rev;
    float val4_q = roundf(val4 * scaling_factor)*scaling_factor_rev;
    float val5_q = roundf(val5 * scaling_factor)*scaling_factor_rev;
    float val6_q = roundf(val6 * scaling_factor)*scaling_factor_rev;
    float val7_q = roundf(val7 * scaling_factor)*scaling_factor_rev;
    float val8_q = roundf(val8 * scaling_factor)*scaling_factor_rev;
    float val9_q = roundf(val9 * scaling_factor)*scaling_factor_rev;
    float val10_q = roundf(val10 * scaling_factor)*scaling_factor_rev;

    float val_sum_q = fabsf(val1_q) + fabsf(val2_q) + fabsf(val3_q) + fabsf(val4_q)
                    + fabsf(val5_q) + fabsf(val6_q) + fabsf(val7_q) + fabsf(val8_q)
                    + fabsf(val9_q) + fabsf(val10_q);

    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 16);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 8);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 4);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 2);
    val_sum_q += __shfl_down_sync(0xffffffff, val_sum_q, 1);
    val_sum_q = __shfl_sync(0xffffffff,val_sum_q,0);

    float val_sum = fabsf(val1) + fabsf(val2) + fabsf(val3) + fabsf(val4)
                    + fabsf(val5) + fabsf(val6) + fabsf(val7) + fabsf(val8)
                    + fabsf(val9) + fabsf(val10);
    
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 16);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 8);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 4);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 2);
    val_sum += __shfl_down_sync(0xffffffff, val_sum, 1);
    val_sum = __shfl_sync(0xffffffff,val_sum,0);

    return log2f(fabsf((val_sum - val_sum_q)/val_sum) + 1);
}

__forceinline__ __device__ Quantization_params QPA(float diff, float range, float prev_mov_avg_range){    
    Quantization_params params;
    float threshold = 0.0005;
    float alpha = 0.04;
    float beta = 0.1;
    float delta = 100;
    float gamma = 2;

    if (diff <= threshold){
        params.r = fabsf(ceilf(log2f(range/(exp2f((8-1))-1))));
        params.n = 8;
    }else{
        params.r = fabsf(ceilf(log2f(range/(exp2f((16-1))-1))));
        params.n = 16;
    }
    params.mov_avg_range = alpha * range + (1.0f - alpha) * prev_mov_avg_range;
    float i1 = delta * diff * diff;
    float i2 = fabsf(params.mov_avg_range - prev_mov_avg_range);
    params.itv = (beta / fmaxf(i1, i2)) - gamma;

    return params;
}

__global__ void afp_sgd_k128_hogwild_kernel(
                            const Node *R,
                            unsigned int nnz,
                            float* p,
                            float* q,
                            curandState *state,
                            float lrate_f,
                            int k,
                            int epoch,
                            int update_count_this_block,
                            int update_vector_size,
                            float lambda_f
                            )
{
    int lane_id = threadIdx.x%32;
    int local_wid = threadIdx.x/32;
    int local_w_num = blockDim.x/32;
    int wid = local_w_num*blockIdx.x + local_wid;  
    int initialization_threshold = (float)(update_count_this_block * update_vector_size)/10.0f;
    Quantization_params user_quantization_params;
    Quantization_params item_quantization_params;
    Quantization_params back_ward_quantization_params;    
    
    int update_user_params_iter = 0;
    int update_item_params_iter = 0;
    int update_backward_iter = 0;
    int cur_iter = 0;

    float scaling_factor_p;
    float scaling_factor_q;
    float scaling_factor_back_prop;

    for(int update_ite = 0; update_ite < update_count_this_block; update_ite ++)
    {    
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

            int base_p = u*k;
            int base_q = v*k;
            
            float* tmp_p_ptr = (float*)p;
            float* tmp_q_ptr = (float*)q;

            float tmp_p1 = p[base_p + lane_id];
            float tmp_q1 = q[base_q + lane_id];
            float tmp_p2 = p[base_p + lane_id + 32];
            float tmp_q2 = q[base_q + lane_id + 32];
            float tmp_p3 = p[base_p + lane_id + 64];
            float tmp_q3 = q[base_q + lane_id + 64];
            float tmp_p4 = p[base_p + lane_id + 96];
            float tmp_q4 = q[base_q + lane_id + 96];

            float p_max_val;    
            float q_max_val;
            float pq_max_val;

            if (cur_iter == update_user_params_iter){
                p_max_val = fmaxf(fmaxf(fabsf(tmp_p1), fabsf(tmp_p2)), fmaxf(fabsf(tmp_p3), fabsf(tmp_p4)));
                p_max_val = reduce_max_input(p_max_val);
                float diff = QEM(tmp_p1, tmp_p2, tmp_p3, tmp_p4, p_max_val);
                if (cur_iter == 0) user_quantization_params.mov_avg_range = p_max_val;
                user_quantization_params = QPA(diff, p_max_val, user_quantization_params.mov_avg_range);
                scaling_factor_p = exp2f(user_quantization_params.r);
                if (epoch == 0 && cur_iter < initialization_threshold) user_quantization_params.itv = 1;
                update_user_params_iter += user_quantization_params.itv;
            }

            if (cur_iter == update_item_params_iter){
                q_max_val = fmaxf(fmaxf(fabsf(tmp_q1), fabsf(tmp_q2)), fmaxf(fabsf(tmp_q3), fabsf(tmp_q4)));
                q_max_val = reduce_max_input(q_max_val);
                float diff = QEM(tmp_q1, tmp_q2, tmp_q3, tmp_q4, q_max_val);
                if (cur_iter == 0) item_quantization_params.mov_avg_range = q_max_val;
                item_quantization_params = QPA(diff, q_max_val, item_quantization_params.mov_avg_range);
                scaling_factor_q = exp2f(item_quantization_params.r);
                if (epoch == 0 && cur_iter < initialization_threshold) item_quantization_params.itv = 1;
                update_item_params_iter += item_quantization_params.itv;
            }

            float pred; 

            if (user_quantization_params.n == 8 && item_quantization_params.n == 8){
                float tmp_int1, tmp_int2, tmp_int3, tmp_int4;
                char val1, val2, val3, val4;

                tmp_int1 = roundf(tmp_p1 * scaling_factor_p);
                tmp_int2 = roundf(tmp_p2 * scaling_factor_p);
                tmp_int3 = roundf(tmp_p3 * scaling_factor_p);
                tmp_int4 = roundf(tmp_p4 * scaling_factor_p);
                
                val1 = (char)(tmp_int1 > 127.0f ? 127.0f : (tmp_int1 < -128.0f)? -128.0f : tmp_int1);
                val2 = (char)(tmp_int2 > 127.0f ? 127.0f : (tmp_int2 < -128.0f)? -128.0f : tmp_int2);
                val3 = (char)(tmp_int3 > 127.0f ? 127.0f : (tmp_int3 < -128.0f)? -128.0f : tmp_int3);                
                val4 = (char)(tmp_int4 > 127.0f ? 127.0f : (tmp_int4 < -128.0f)? -128.0f : tmp_int4);
                
                char4 p_int8_4 = make_char4(val1, val2, val3, val4);
                tmp_int1 = roundf(tmp_q1 * scaling_factor_q);
                tmp_int2 = roundf(tmp_q2 * scaling_factor_q);
                tmp_int3 = roundf(tmp_q3 * scaling_factor_q);
                tmp_int4 = roundf(tmp_q4 * scaling_factor_q);
                
                val1 = (char)(tmp_int1 > 127.0f ? 127.0f : (tmp_int1 < -128.0f)? -128.0f : tmp_int1);
                val2 = (char)(tmp_int2 > 127.0f ? 127.0f : (tmp_int2 < -128.0f)? -128.0f : tmp_int2);
                val3 = (char)(tmp_int3 > 127.0f ? 127.0f : (tmp_int3 < -128.0f)? -128.0f : tmp_int3);                
                val4 = (char)(tmp_int4 > 127.0f ? 127.0f : (tmp_int4 < -128.0f)? -128.0f : tmp_int4);

                char4 q_int8_4 = make_char4(val1, val2, val3, val4);

                int tmp_product = 0;
                tmp_product = __dp4a(p_int8_4, q_int8_4, tmp_product);
                
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
                tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
                
                tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
                pred = tmp_product / (scaling_factor_p * scaling_factor_q);
            }else{
                __half2 p_half2_1, p_half2_2, q_half2_1, q_half2_2;
                float scaling_factor_p_rev = 1/scaling_factor_p;
                float scaling_factor_q_rev = 1/scaling_factor_q;
                
                p_half2_1.x = __float2half(roundf(tmp_p1 * scaling_factor_p)*scaling_factor_p_rev);
                p_half2_1.y = __float2half(roundf(tmp_p2 * scaling_factor_p)*scaling_factor_p_rev);
                p_half2_2.x = __float2half(roundf(tmp_p3 * scaling_factor_p)*scaling_factor_p_rev);
                p_half2_2.y = __float2half(roundf(tmp_p4 * scaling_factor_p)*scaling_factor_p_rev);
                q_half2_1.x = __float2half(roundf(tmp_q1 * scaling_factor_q)*scaling_factor_q_rev);
                q_half2_1.y = __float2half(roundf(tmp_q2 * scaling_factor_q)*scaling_factor_q_rev);
                q_half2_2.x = __float2half(roundf(tmp_q3 * scaling_factor_q)*scaling_factor_q_rev);
                q_half2_2.y = __float2half(roundf(tmp_q4 * scaling_factor_q)*scaling_factor_q_rev);

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

            if (cur_iter == update_backward_iter){
                p_max_val = fmaxf(fmaxf(fabsf(tmp_p1), fabsf(tmp_p2)), fmaxf(fabsf(tmp_p3), fabsf(tmp_p4)));
                q_max_val = fmaxf(fmaxf(fabsf(tmp_q1), fabsf(tmp_q2)), fmaxf(fabsf(tmp_q3), fabsf(tmp_q4)));
                pq_max_val = fmaxf(p_max_val, q_max_val);
                pq_max_val = fmaxf(fmaxf(pq_max_val, fabsf(lambda_f)), fabsf(ruv));
                pq_max_val = reduce_max_input(pq_max_val);

                float diff = QEM_BP(tmp_p1, tmp_p2, tmp_p3, tmp_p4, tmp_q1, tmp_q2, tmp_q3, tmp_q4, lambda_f, ruv, pq_max_val);

                if (cur_iter == 0) back_ward_quantization_params.mov_avg_range = pq_max_val;
                back_ward_quantization_params = QPA(diff, pq_max_val, back_ward_quantization_params.mov_avg_range);
                scaling_factor_back_prop = exp2f(back_ward_quantization_params.r);
                if (epoch == 0 && cur_iter < initialization_threshold) back_ward_quantization_params.itv = 1;
                update_backward_iter += back_ward_quantization_params.itv;
            }

            float p1_grad_val;
            float q1_grad_val;
            float p2_grad_val;
            float q2_grad_val;
            float p3_grad_val;
            float q3_grad_val;
            float p4_grad_val;
            float q4_grad_val;

            if (back_ward_quantization_params.n == 8){
                float tmp_int_p1, tmp_int_p2, tmp_int_p3, tmp_int_p4, tmp_int_q1, tmp_int_q2, tmp_int_q3, tmp_int_q4, tmp_int_lambda, tmp_int_ruv;
                char val_p1, val_p2, val_p3, val_p4, val_q1, val_q2, val_q3, val_q4, val_lambda, val_ruv;

                tmp_int_p1 = roundf(tmp_p1 * scaling_factor_back_prop);
                tmp_int_p2 = roundf(tmp_p2 * scaling_factor_back_prop);
                tmp_int_p3 = roundf(tmp_p3 * scaling_factor_back_prop);
                tmp_int_p4 = roundf(tmp_p4 * scaling_factor_back_prop);
                tmp_int_q1 = roundf(tmp_q1 * scaling_factor_back_prop);
                tmp_int_q2 = roundf(tmp_q2 * scaling_factor_back_prop);
                tmp_int_q3 = roundf(tmp_q3 * scaling_factor_back_prop);
                tmp_int_q4 = roundf(tmp_q4 * scaling_factor_back_prop);
                tmp_int_lambda = roundf(-1.0f * lambda_f * scaling_factor_back_prop);
                tmp_int_ruv = roundf(ruv * scaling_factor_back_prop);

                val_p1 = (char)(tmp_int_p1 > 127.0f ? 127.0f : (tmp_int_p1 < -128.0f)? -128.0f : tmp_int_p1);
                val_p2 = (char)(tmp_int_p2 > 127.0f ? 127.0f : (tmp_int_p2 < -128.0f)? -128.0f : tmp_int_p2);
                val_p3 = (char)(tmp_int_p3 > 127.0f ? 127.0f : (tmp_int_p3 < -128.0f)? -128.0f : tmp_int_p3);                
                val_p4 = (char)(tmp_int_p4 > 127.0f ? 127.0f : (tmp_int_p4 < -128.0f)? -128.0f : tmp_int_p4);
                val_q1 = (char)(tmp_int_q1 > 127.0f ? 127.0f : (tmp_int_q1 < -128.0f)? -128.0f : tmp_int_q1);
                val_q2 = (char)(tmp_int_q2 > 127.0f ? 127.0f : (tmp_int_q2 < -128.0f)? -128.0f : tmp_int_q2);
                val_q3 = (char)(tmp_int_q3 > 127.0f ? 127.0f : (tmp_int_q3 < -128.0f)? -128.0f : tmp_int_q3);                
                val_q4 = (char)(tmp_int_q4 > 127.0f ? 127.0f : (tmp_int_q4 < -128.0f)? -128.0f : tmp_int_q4);
                val_lambda = (char)(tmp_int_lambda > 127.0f ? 127.0f : (tmp_int_lambda < -128.0f)? -128.0f : tmp_int_lambda);
                val_ruv = (char)(tmp_int_ruv > 127.0f ? 127.0f : (tmp_int_ruv < -128.0f)? -128.0f : tmp_int_ruv);

                char4 int_val2 = make_char4(val_ruv, val_lambda, 0, 0);
                float scaling_square_rev = 1/(scaling_factor_back_prop * scaling_factor_back_prop);
                
                p1_grad_val = comp_grad_int8_cvpr(val_q1, val_p1,  scaling_square_rev, int_val2);
                q1_grad_val = comp_grad_int8_cvpr(val_p1, val_q1,  scaling_square_rev, int_val2);
                p2_grad_val = comp_grad_int8_cvpr(val_q2, val_p2,  scaling_square_rev, int_val2);
                q2_grad_val = comp_grad_int8_cvpr(val_p2, val_q2,  scaling_square_rev, int_val2);
                p3_grad_val = comp_grad_int8_cvpr(val_q3, val_p3,  scaling_square_rev, int_val2);
                q3_grad_val = comp_grad_int8_cvpr(val_p3, val_q3,  scaling_square_rev, int_val2);
                p4_grad_val = comp_grad_int8_cvpr(val_q4, val_p4,  scaling_square_rev, int_val2);
                q4_grad_val = comp_grad_int8_cvpr(val_p4, val_q4,  scaling_square_rev, int_val2);
            }else {
                __half2 half2_tmp2;
                float scaling_factor_rev = 1/scaling_factor_back_prop;
                half2_tmp2.x = __float2half(roundf(ruv * scaling_factor_back_prop)*scaling_factor_rev);
                half2_tmp2.y = __float2half(roundf(-1.0f*lambda_f * scaling_factor_back_prop)*scaling_factor_rev);

                p1_grad_val = comp_grad_fp16_cvpr(tmp_q1, tmp_p1, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
                q1_grad_val = comp_grad_fp16_cvpr(tmp_p1, tmp_q1, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
                p2_grad_val = comp_grad_fp16_cvpr(tmp_q2, tmp_p2, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
                q2_grad_val = comp_grad_fp16_cvpr(tmp_p2, tmp_q2, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
                p3_grad_val = comp_grad_fp16_cvpr(tmp_q3, tmp_p3, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
                q3_grad_val = comp_grad_fp16_cvpr(tmp_p3, tmp_q3, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
                p4_grad_val = comp_grad_fp16_cvpr(tmp_q4, tmp_p4, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
                q4_grad_val = comp_grad_fp16_cvpr(tmp_p4, tmp_q4, ruv, scaling_factor_back_prop, scaling_factor_rev ,half2_tmp2);
            }
            
            p[base_p + lane_id] = tmp_p1 + lrate_f*p1_grad_val;
            q[base_q + lane_id] = tmp_q1 + lrate_f*q1_grad_val;
            p[base_p + lane_id + 32] = tmp_p2 + lrate_f*p2_grad_val;
            q[base_q + lane_id + 32] = tmp_q2 + lrate_f*q2_grad_val;
            p[base_p + lane_id + 64] = tmp_p3 + lrate_f*p3_grad_val;
            q[base_q + lane_id + 64] = tmp_q3 + lrate_f*q3_grad_val;
            p[base_p + lane_id + 96] = tmp_p4 + lrate_f*p4_grad_val;
            q[base_q + lane_id + 96] = tmp_q4 + lrate_f*q4_grad_val;

            cur_iter += 1;
        }    
    }
}