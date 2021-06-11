#include <iostream>
#include <sys/stat.h>
#include "common_struct.h"
#include "io_utils.h"
#include "model_init.h"
#include "sgd.h"
#define SINGLE_PREC_SGD_BASELINE 1
using namespace std;

// Helper functions to check if file exists
bool exists (const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

int main (int argc, const char* argv[]){
    string infile = "";
    string outfile = "";
    string testfile = "None";
    // float lambda = 0.015;
    float lambda = 0.015;
    float alpha = 0.01f;
    float decay = 0.1f;
    float threshold = 0.0;
    float sample_ratio = 0.01f;
    float error_threshold = 0.001f;
    unsigned int k = 128;
    unsigned int iteration = 50;
    unsigned int work_groups = 2048;
    unsigned int thread_block_size = 128;
    unsigned int user_group_num = 10;
    unsigned int item_group_num = 10;
    unsigned int grouping_method = 4;
    unsigned int version = 7;
    unsigned int precision = 0;
    unsigned int interval = 1;
    unsigned int fp32_user_last_n_group = 10;
    unsigned int fp32_item_last_n_group = 10;
    float fp16_user_ratio = 0.01f;
    float fp16_item_ratio = 0.01f;

    if(argc < 2){
        cout << argv[0] << " [-t <threads> -p <predictions/user> -o <output-tsv>] <input-tsv>" << endl;
        return(0);
    }else if(argc == 2){    
        infile = string(argv[1]);
    }else{
        for(int i = 0; i < argc; i++){
        
            if(string(argv[i]) == "-i" && i < argc-1){
                infile = string(argv[i+1]);
            }
            if(string(argv[i]) == "-o" && i < argc-1){
                outfile = string(argv[i+1]);
            }
            if(string(argv[i]) == "-y" && i < argc-1){
                testfile = string(argv[i+1]);
            }
            if(string(argv[i]) == "-l" && i < argc-1){
                iteration = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-k" && i < argc-1){
                k = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-b" && i < argc-1){
                lambda = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-a" && i < argc-1){
                alpha = atof(argv[i+1]);
            }           
            if(string(argv[i]) == "-wg" && i < argc-1){
                work_groups = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-d" && i < argc-1){
                decay = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-th" && i < argc-1){
                threshold = atof(argv[i+1]);
            }
            if(string(argv[i]) == "-bl" && i < argc-1){
                thread_block_size = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-v" && i < argc-1){
                version = atoi(argv[i+1]);
            }      
            if(string(argv[i]) == "-r" && i < argc-1){
                sample_ratio = atof(argv[i+1]);
            }    
            if(string(argv[i]) == "-ug" && i < argc-1){
                user_group_num = atoi(argv[i+1]);
            }
            if(string(argv[i]) == "-ig" && i < argc-1){
                item_group_num = atoi(argv[i+1]);
            }      
            if(string(argv[i]) == "-g" && i < argc-1){
                grouping_method = atoi(argv[i+1]);
            }    
            if(string(argv[i]) == "-e" && i < argc-1){
                error_threshold = atof(argv[i+1]);
            }    
            if(string(argv[i]) == "-s" && i < argc-1){
                sample_ratio = atof(argv[i+1]);
            }          
            if(string(argv[i]) == "-p" && i < argc-1){
                precision = atoi(argv[i+1]);
            }    
            if(string(argv[i]) == "-it" && i < argc-1){
                interval = atoi(argv[i+1]);
            }    
            if(string(argv[i]) == "-uls" && i < argc-1){
                fp32_user_last_n_group = atoi(argv[i+1]);
            }       
            if(string(argv[i]) == "-ils" && i < argc-1){
                fp32_item_last_n_group = atoi(argv[i+1]);
            }             
            if(string(argv[i]) == "-hun" && i < argc-1){
                fp16_user_ratio = atof(argv[i+1]);
            }    
            if(string(argv[i]) == "-hin" && i < argc-1){
                fp16_item_ratio = atof(argv[i+1]);
            }    
            if(string(argv[i]) == "-h"){
                cout << argv[0] << " [-t <threads> -p <predictions/user> -o <output-tsv>] <input-tsv>" << endl;
                return(0);
            }
        }
    }

    if(!exists(infile)){
        cout << infile << " doesn't exist!" << endl;
        return(0);
    }

    cout << "\nInput file                : " << infile << endl;
    cout << "Test file                   : " << testfile << endl;
    cout << "Output file                 : " << outfile << endl;
    cout << "Latent features             : " << k << endl;
    cout << "Learning rate               : " << alpha << endl;
    cout << "Decay                       : " << decay << endl;
    cout << "Lambda                      : " << lambda << endl;
	cout << "Iteration                   : " << iteration << endl;
    cout << "Num of workers              : " << work_groups << endl;
    cout << "Thread block size           : " << thread_block_size << endl;
    cout << "Sample ratio                : " << sample_ratio << endl;
    cout << "Error threshold             : " << error_threshold << endl;
    cout << "Grouping method             : " << grouping_method << endl;
    cout << "Interval                    : " << interval << endl;

    if (precision == 1) cout << "Precision                   : FP32" << endl;
    else cout << "Precision                   : FP16" << endl;
    cout << " (";
    if (grouping_method == 1) cout << "Strict equal size";
    else if (grouping_method == 2) cout << "Equal rating range";
    else if (grouping_method == 3) cout << "Exp rating range";
    else if (grouping_method == 4) cout << "Non-strict equal size";
    cout << ")" << endl;

    if (version == 15){
        cout << "FP32 last user groups   : " << fp32_user_last_n_group << endl;
        cout << "FP32 last item groups   : " << fp32_item_last_n_group << endl;
    }
    SGD sgd_model;
    Mf_info mf_info;

    mf_info = read_training_dataset(infile);
    read_test_dataset(&mf_info, testfile);
    
    const char* data = infile.c_str();
    bool yahoo =false;
    if (strstr(data, "Yahoo") != NULL) yahoo=true;
    mf_info.is_yahoo = yahoo;
    
    cout << "The number of nonzeros      :" << mf_info.n << endl;
    cout << "The number of users         :" << mf_info.max_user << endl;
    cout << "The number of items         :" << mf_info.max_item << endl;
    cout << "The number of test nonzeros :" << mf_info.test_n << endl;

    // parameter

    mf_info.params.k = k;
    mf_info.params.learning_rate = alpha;
    mf_info.params.decay = decay;
    mf_info.params.lambda = lambda;
    mf_info.params.num_workers = work_groups;
    mf_info.params.mean_val = 0;
    mf_info.params.std_val = 0.1;
    mf_info.params.epoch = iteration; 
    mf_info.params.thread_block_size = thread_block_size;
    mf_info.user_group_num = user_group_num;
    mf_info.item_group_num = item_group_num;
    mf_info.sample_ratio = sample_ratio;
    mf_info.grouping_method = grouping_method;
    mf_info.version = version;
    mf_info.error_threshold = error_threshold;
    mf_info.out_file = outfile;
    mf_info.interval = interval;
    mf_info.fp32_user_last_n_group = fp32_user_last_n_group;
    mf_info.fp32_item_last_n_group = fp32_item_last_n_group;
    mf_info.fp16_user_ratio = fp16_user_ratio;
    mf_info.fp16_item_ratio = fp16_item_ratio;
    
    if (version != 14 && version != 16 && version != 18) init_model_single(&mf_info, &sgd_model);
    else init_model_half(&mf_info, &sgd_model);

    //! Comparison indexing 방식 최종 버전 ******
    if (version == 1) grouped_sgd_training_comparison_based_grad_diversity_partial_group(&mf_info, &sgd_model);
    else if (version == 2) mixed_precision_training_mf(&mf_info, &sgd_model);
    else if (version == 3) muppet_training_mf(&mf_info, &sgd_model); 
    else if (version == 4) adaptive_fixed_point_training_mf(&mf_info, &sgd_model);

    return 0;
}