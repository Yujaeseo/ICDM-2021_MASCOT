#include <iostream>
#include <sys/stat.h>
#include "common.h"
#include "common_struct.h"
#include "io_utils.h"
#include "rmse.h"
using namespace std;

// Helper functions to check if file exists
bool exists (const std::string& name) {
    struct stat buffer;   
    return (stat(name.c_str(), &buffer) == 0); 
}

int main (int argc, const char* argv[]){
    string infile = "";
    string testfile = "None";
    int version = 1;

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
            if(string(argv[i]) == "-y" && i < argc-1){
                testfile = string(argv[i+1]);
            }
            if(string(argv[i]) == "-v" && i < argc-1){
                version = stoi(argv[i+1]);
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
    cout << "Version                     : " << version << endl;

    SGD sgd_model;
    Mf_info mf_info;

    vector<Node> test_set = read_testset_pretrained_model(&mf_info, testfile);
    read_trained_model(&mf_info, &sgd_model, infile);
    remove_elements(&mf_info, test_set, version ,testfile);
    cudaMalloc(&mf_info.d_test_COO, sizeof(Node) * mf_info.test_n);
    cudaMalloc(&sgd_model.d_p, sizeof(float) * mf_info.params.k * mf_info.max_user);
    cudaMalloc(&sgd_model.d_q, sizeof(float) * mf_info.params.k * mf_info.max_item);

    cudaMemcpy(mf_info.d_test_COO, mf_info.test_COO, sizeof(Node) * mf_info.test_n, cudaMemcpyHostToDevice);
    float* d_e_group;
    unsigned int error_kernel_work_groups = ceil(mf_info.test_n/(float)512);
    unsigned int group_error_size = error_kernel_work_groups;
    unsigned int iter_num = ceil(mf_info.test_n / (float) (512 * error_kernel_work_groups));
    unsigned int seg_size = 32;

    cudaMalloc(&d_e_group, sizeof(float) * group_error_size);
    double rmse = gpu_test_rmse(&mf_info, &sgd_model, mf_info.d_test_COO, d_e_group, error_kernel_work_groups, iter_num, seg_size, group_error_size);
    cout << "RMSE : " << rmse << endl;
}