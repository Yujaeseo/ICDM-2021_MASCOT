__global__ void single_sgd_k128_hogwild_kernel(
                            const Node *R,
                            unsigned int nnz,
                            float *p,
                            float *q,
                            curandState *state,
                            float lrate,
                            int k,
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

            int base_p = u*k;
            int base_q = v*k;

            const float tmp_p1 = p[base_p + lane_id];
            const float tmp_q1 = q[base_q + lane_id];

            const float tmp_p2 = p[base_p + lane_id + 32];
            const float tmp_q2 = q[base_q + lane_id + 32];

            const float tmp_p3 = p[base_p + lane_id + 64];
            const float tmp_q3 = q[base_q + lane_id + 64];

            const float tmp_p4 = p[base_p + lane_id + 96];
            const float tmp_q4 = q[base_q + lane_id + 96];

        
            float tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2)) + ((tmp_p3*tmp_q3) + (tmp_p4*tmp_q4));

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

__global__ void mem_quant_sgd_k128_hogwild_kernel(
                            const Node *R,
                            unsigned int nnz,
                            half *p,
                            half *q,
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

            float tmp_p1 = __half2float(p[base_p + lane_id]);
            float tmp_q1 = __half2float(q[base_q + lane_id]);
            
            float tmp_p2 = __half2float(p[base_p + lane_id + 32]);
            float tmp_q2 = __half2float(q[base_q + lane_id + 32]);
        
            float tmp_p3 = __half2float(p[base_p + lane_id + 64]);
            float tmp_q3 = __half2float(q[base_q + lane_id + 64]);
            
            float tmp_p4 = __half2float(p[base_p + lane_id + 96]);
            float tmp_q4 = __half2float(q[base_q + lane_id + 96]);

            float tmp_product = tmp_p1*tmp_q1 + tmp_p2*tmp_q2 + tmp_p3*tmp_q3 + tmp_p4*tmp_q4;

            //get dot product.
            tmp_product += __shfl_down_sync(0xffffffff,tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff,tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff,tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff,tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff,tmp_product, 1);

            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);

            float ruv = r - tmp_product;

            //update
            //only works for k=blockDim.x=128
            p[base_p + lane_id +  0] = __float2half(tmp_p1 + lrate*(ruv*tmp_q1 - lambda*tmp_p1));
            q[base_q + lane_id +  0] = __float2half(tmp_q1 + lrate*(ruv*tmp_p1 - lambda*tmp_q1));
            
            p[base_p + lane_id + 32] = __float2half(tmp_p2 + lrate*(ruv*tmp_q2 - lambda*tmp_p2));
            q[base_q + lane_id + 32] = __float2half(tmp_q2 + lrate*(ruv*tmp_p2 - lambda*tmp_q2));

            p[base_p + lane_id + 64] = __float2half(tmp_p3 + lrate*(ruv*tmp_q3 - lambda*tmp_p3));
            q[base_q + lane_id + 64] = __float2half(tmp_q3 + lrate*(ruv*tmp_p3 - lambda*tmp_q3));

            p[base_p + lane_id + 96] = __float2half(tmp_p4 + lrate*(ruv*tmp_q4 - lambda*tmp_p4));
            q[base_q + lane_id + 96] = __float2half(tmp_q4 + lrate*(ruv*tmp_p4 - lambda*tmp_q4));
        }    
    }
}

__global__ void switching_only_sgd_k128_hogwild_kernel(
                            const Node *R,
                            unsigned int nnz,
                            void* p,
                            void* q,
                            curandState *state,
                            float lrate_f,
                            int k,
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

        start_id = __shfl_sync(0xffffffff,start_id, 0);

        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            float r = __ldg(&R[offset].r);
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            int base_p = u*k;
            int base_q = v*k;
            
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

                    atomicAdd(sum_updated_val + lane_id, tmp_p1_updated_val_f + tmp_q1_updated_val_f);
                    atomicAdd(sum_updated_val + lane_id + 32, tmp_p2_updated_val_f + tmp_q2_updated_val_f);
                    atomicAdd(sum_updated_val + lane_id + 64, tmp_p3_updated_val_f + tmp_q3_updated_val_f);
                    atomicAdd(sum_updated_val + lane_id + 96, tmp_p4_updated_val_f + tmp_q4_updated_val_f);
                }
            }
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