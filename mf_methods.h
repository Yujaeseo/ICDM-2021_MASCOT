#ifndef _SGD_GPU
#define _SGD_GPU
using namespace std;
void mascot_training_mf(Mf_info* mf_info, SGD* sgd_info);
void adaptive_fixed_point_training_mf(Mf_info* mf_info, SGD* sgd_info);
void muppet_training_mf(Mf_info* mf_info, SGD* sgd_model);
void mixed_precision_training_mf(Mf_info* mf_info, SGD* sgd_info);
void training_single_mf(Mf_info *mf_info, SGD *sgd_info);
void mascot_training_mf_naive(Mf_info *mf_info, SGD *sgd_info);
void training_mem_quant_mf(Mf_info *mf_info, SGD *sgd_info);
void training_switching_only(Mf_info* mf_info, SGD* sgd_info);
#endif
