#include <iostream>
#include <sys/stat.h>
#include "common_struct.h"
#include "io_utils.h"
#include "model_init.h"
#include "mf_methods.h"
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
    float lambda = 0.015;
    float alpha = 0.01f;
    float decay = 0.1f;
    float sample_ratio = 0.01f;
    float error_threshold = 0.001f;
    unsigned int k = 128;
    unsigned int iteration = 50;
    unsigned int work_groups = 2048;
    unsigned int thread_block_size = 128;
    unsigned int user_group_num = 10;
    unsigned int item_group_num = 10;
    unsigned int version = 7;
    unsigned int interval = 1;
    unsigned int reconst_save = 0;

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
            if(string(argv[i]) == "-e" && i < argc-1){
                error_threshold = atof(argv[i+1]);
            }    
            if(string(argv[i]) == "-s" && i < argc-1){
                sample_ratio = atof(argv[i+1]);
            }          
            if(string(argv[i]) == "-it" && i < argc-1){
                interval = atoi(argv[i+1]);
            }    
            if(string(argv[i]) == "-rc" && i < argc-1){
                reconst_save = atoi(argv[i+1]);
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

    cout << endl;
    cout << "Input file                  : " << infile << endl;
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
    cout << "Interval                    : " << interval << endl;
    
    SGD sgd_model;
    Mf_info mf_info;

    read_training_dataset(&mf_info, infile);
    read_test_dataset(&mf_info, testfile);

    cout << "The number of nonzeros      : " << mf_info.n << endl;
    cout << "The number of users         : " << mf_info.max_user << endl;
    cout << "The number of items         : " << mf_info.max_item << endl;
    cout << "The number of test nonzeros : " << mf_info.test_n << endl;

    mf_info.params.k = k;
    mf_info.params.learning_rate = alpha;
    mf_info.params.decay = decay;
    mf_info.params.lambda = lambda;
    mf_info.params.num_workers = work_groups;
    mf_info.params.mean_val = 0;
    mf_info.params.std_val = 0.1;
    mf_info.params.epoch = iteration; 
    mf_info.params.thread_block_size = thread_block_size;
    mf_info.params.user_group_num = user_group_num;
    mf_info.params.item_group_num = item_group_num;
    mf_info.params.sample_ratio = sample_ratio;
    mf_info.version = version;
    mf_info.params.error_threshold = error_threshold;
    mf_info.params.interval = interval;

    if (infile.find("Yahoo") != string::npos) mf_info.is_yahoo = true;

    if (version != 7 && version != 8 && version != 4) init_model_single(&mf_info, &sgd_model);
    else init_model_half(&mf_info, &sgd_model);

    if (version == 1) mascot_training_mf(&mf_info, &sgd_model);
    else if (version == 2) adaptive_fixed_point_training_mf(&mf_info, &sgd_model);
    else if (version == 3) muppet_training_mf(&mf_info, &sgd_model);
    else if (version == 4) mixed_precision_training_mf(&mf_info, &sgd_model);
    else if (version == 5) training_single_mf(&mf_info, &sgd_model);
    else if (version == 6) mascot_training_mf_naive(&mf_info, &sgd_model);
    else if (version == 7) training_mem_quant_mf(&mf_info, &sgd_model);
    else if (version == 8) training_switching_only(&mf_info, &sgd_model);
    if (outfile != "") {
        if (version == 1) save_trained_model_reconst(&mf_info, &sgd_model, outfile);
        else save_trained_model(&mf_info, &sgd_model, outfile);
    }

    if (version == 1 && reconst_save == 1) save_reconst_testset(&mf_info, testfile);
    
    cudaFree(sgd_model.d_p);
    cudaFree(sgd_model.d_q);
    cudaFree(sgd_model.d_half_p);
    cudaFree(sgd_model.d_half_q);
    
    return 0;
}