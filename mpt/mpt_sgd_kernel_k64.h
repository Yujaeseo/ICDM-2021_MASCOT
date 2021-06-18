__global__ void mpt_sgd_k64_hogwild_kernel(
                            const Node *R,
                            unsigned int nnz,
                            float *p,
                            float *q,
                            curandState *state,
                            float lrate,
                            int k,
                            int update_count_this_block,
                            int update_vector_size,
                            __half lambda,
                            float scaling_factor
                            )
{    
    float scaling_factor2 = 1/scaling_factor;
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

        start_id = __shfl_sync(0xffffffff,start_id, 0);
        
        for(int i = 0;i < update_vector_size;i++)
        {
            int offset = (start_id + i)%nnz;
            
            __half r = __float2half_rn(__ldg(&R[offset].r));
            int u = __ldg(&R[offset].u);
            int v = __ldg(&R[offset].i);

            int base_p = u*k;
            int base_q = v*k;

            float tmp_p1_f = (p[base_p + lane_id]);
            float tmp_q1_f = (q[base_q + lane_id]);

            float tmp_p2_f = (p[base_p + lane_id + 32]);
            float tmp_q2_f = (q[base_q + lane_id + 32]);
            
            __half tmp_p1 = __float2half_rn(tmp_p1_f);
            __half tmp_q1 = __float2half_rn(tmp_q1_f);

            __half tmp_p2 = __float2half_rn(tmp_p2_f);
            __half tmp_q2 = __float2half_rn(tmp_q2_f);

            __half tmp_product = ((tmp_p1*tmp_q1) + (tmp_p2*tmp_q2));

            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 16);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 8);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 4);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 2);
            tmp_product += __shfl_down_sync(0xffffffff, tmp_product, 1);
            
            tmp_product = __shfl_sync(0xffffffff,tmp_product,0);
            
            __half ruv = scaling_factor_half * (r-tmp_product);

            p[base_p + lane_id +  0] = tmp_p1_f + (lrate * (scaling_factor2 * __half2float(ruv*tmp_q1 - lambda*tmp_p1)));
            q[base_q + lane_id +  0] = tmp_q1_f + (lrate * (scaling_factor2 * __half2float(ruv*tmp_p1 - lambda*tmp_q1)));
            
            p[base_p + lane_id + 32] = tmp_p2_f + (lrate * (scaling_factor2 * __half2float(ruv*tmp_q2 - lambda*tmp_p2)));
            q[base_q + lane_id + 32] = tmp_q2_f + (lrate * (scaling_factor2 * __half2float(ruv*tmp_p2 - lambda*tmp_q2)));
        }    
    }
}