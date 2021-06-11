#ifndef _SGD_GPU
#define _SGD_GPU
using namespace std;

void sgd_training_single(Mf_info *mf_info, SGD *sgd_info);
void grouped_sgd_training_map_based_indexing(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_indexing_fp32_version(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_grad_diversity(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_grad_diversity_fp32_version(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_division_based_indexing(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_indexing(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_division_based_indexing_iteration(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_grad_diversity_partial_group(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_division_based_indexing_fp32_version(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_indexing_iteration(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_indexing_check_gradient(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_grad_diversity_partial_group_fp32_version(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_indexing_eval_indexing(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_grad_diversity_eval_indexing(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_map_based_eval_compute_precision(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_grad_diversity_partial_group(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_grad_diversity_partial_group_timing_overhead(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_grad_diversity_not_group_only_switching(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_grad_diversity_not_switching_only_grouping(Mf_info* mf_info, SGD* sgd_info);
void mixed_precision_training_mf(Mf_info* mf_info, SGD* sgd_info);
void muppet_training_mf(Mf_info* mf_info, SGD* sgd_info);
void muppet_training_mf_parameter_switching_version(Mf_info* mf_info, SGD* sgd_info);
void adaptive_fixed_point_training_mf(Mf_info* mf_info, SGD* sgd_info);
void adaptive_fixed_point_training_mf_log(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_grad_diversity_partial_group_time_check_per_area(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_grad_diversity_partial_group_cur_version_eval_indexing(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_random_select_user_item(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_grad_diversity_partial_group_naive_version(Mf_info* mf_info, SGD* sgd_info);
void grouped_sgd_training_comparison_based_grad_diversity_partial_group_naive_version_device_mem(Mf_info* mf_info, SGD* sgd_info);
#endif
