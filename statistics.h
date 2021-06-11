void print_group_error_val(string outfile_path, float* err_arr, float rmse, unsigned int group_num, unsigned int epoch, unsigned int max_epoch){
    const char* pt = outfile_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }

    for (int i = 0; i < group_num; i++){
        filep << err_arr[i] << " ";
    }
    filep << rmse << endl;
    if (epoch == max_epoch) filep << "\n\n";
    filep.close();
}

void print_rmse(Mf_info* mf_info, string outfile_path, unsigned int cur_epoch ,float rmse){
    const char* pt = outfile_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }

    filep << cur_epoch + 1 << "\t" << rmse << endl;
    if (cur_epoch == mf_info->params.epoch == cur_epoch + 1) filep << "\n";
    filep.close();
}

void print_exec_time_and_rmse(string outfile_path, map<string, double> statistics_map){
    const char* pt = outfile_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }

    filep << "Preprocess\t" << statistics_map["preprocess"] << "\t" 
          << "Switching\t" << statistics_map["switching"] << "\t"
          << "Update\t" << statistics_map["update"] << "\t"
          << "Total(ms)\t" << statistics_map["total"] << "\t"
          << "RMSE\t" << statistics_map["rmse"] << endl;

    filep.close();
}

void print_exec_time_and_rmse_group_only_version(string outfile_path, map<string, double> statistics_map){
    const char* pt = outfile_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }

    filep << "Num of user groups\t" << statistics_map["total_user_groups"] << "\t"
          << "Num of item groups\t" << statistics_map["total_item_groups"] << "\t"
          << "FP32 user groups\t" << statistics_map["fp32_user_groups"] << "\t"
          << "FP32 item groups\t" << statistics_map["fp32_item_groups"] << "\t"
          << "Preprocess\t" << statistics_map["preprocess"] << "\t" 
          << "Switching\t" << statistics_map["switching"] << "\t"
          << "Update\t" << statistics_map["update"] << "\t"
          << "Total(ms)\t" << statistics_map["total"] << "\t"
          << "RMSE\t" << statistics_map["rmse"] << endl;

    filep.close();
}

void print_group_switching_log(string outfile_path, map<unsigned int, vector<unsigned int>> user_switching_log, map<unsigned int, vector<unsigned int>> item_switching_log){
    const char* pt = outfile_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }

    filep << "User groups" << endl;
    for (map<unsigned int, vector<unsigned int>>::iterator it = user_switching_log.begin(); it != user_switching_log.end(); it++){
        filep << "Epoch\t" << it->first + 1 << "\t";
        for (int i = 0; i < it->second.size(); i++){
            filep << it->second[i] + 1 << "\t";
        }
        filep << endl;
    }
    
    filep << "\nItem groups" << endl;
    for (map<unsigned int, vector<unsigned int>>::iterator it = item_switching_log.begin(); it != item_switching_log.end(); it++){
        filep << "Epoch\t" << it->first + 1 << "\t";
        for (int i = 0; i < it->second.size(); i++){
            filep << it->second[i] + 1 << "\t";
        }
        filep << endl;
    }

    filep << "\n\n";
    filep.close();
}

void print_grad_diversity_log(string outfile_path, vector<vector<double>> user_grad_diversity_log, vector<vector<double>> item_grad_diversity_log){
    const char* pt = outfile_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }
    filep << "User groups" << endl;
    for (int e = 0; e < user_grad_diversity_log.size(); e++){
        for (int g = 0; g < user_grad_diversity_log[e].size(); g++){
            filep << user_grad_diversity_log[e][g] << "\t";
        }
        filep << endl;
    }
    filep << "\nItem groups" << endl;
    for (int e = 0; e < item_grad_diversity_log.size(); e++){
        for (int g = 0; g < item_grad_diversity_log[e].size(); g++){
            filep << item_grad_diversity_log[e][g] << "\t";
        }
        filep << endl;
    }

    filep << "\n\n";
    filep.close();
}

void print_grad_diversity_log_not_grouping(string outfile_path, vector<vector<double>> grad_diversity_log){
    const char* pt = outfile_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }

    filep << "\nGrad diversity" << endl;
    for (int e = 0; e < grad_diversity_log.size(); e++){
        for (int g = 0; g < grad_diversity_log[e].size(); g++){
            filep << grad_diversity_log[e][g] << "\t";
        }
        filep << endl;
    }

    filep << "\n\n";
    filep.close();
}

void check_group_cnt(Mf_info* mf_info){
    //! ============================== print file =========================================
    string group_statistics_output_path = string("./New_statistics/Gradient diversity fp16/group_info/group_info") + mf_info->out_file + ".txt";
    const char* pt = group_statistics_output_path.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::app);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }

    // unsigned int* user_group_cnt = new unsigned int[mf_info->user_group_num];
    // for (int i = 0; i <mf_info->user_group_num; i++) user_group_cnt[i] = 0;
    // unsigned int* item_group_cnt = new unsigned int[mf_info->item_group_num];
    // for (int i = 0; i <mf_info->item_group_num; i++) item_group_cnt[i] = 0;
    // unsigned int* user_group_rating_cnt = new unsigned int[mf_info->user_group_num];
    // for (int i = 0; i <mf_info->user_group_num; i++) user_group_rating_cnt[i] = 0;
    // unsigned int* item_group_rating_cnt = new unsigned int[mf_info->item_group_num];
    // for (int i = 0; i <mf_info->item_group_num; i++) item_group_rating_cnt[i] = 0;
    // // unsigned int* grid_rating_cnt = new unsigned int[mf_info->user_group_num * mf_info->item_group_num];
    // // for (int i = 0; i <mf_info->user_group_num * mf_info->item_group_num; i++) grid_rating_cnt[i] = 0;


    // for (int i = 0; i < mf_info->n; i++) {
    //     unsigned int user_group = mf_info->user_group_idx[mf_info->R[i].u];
    //     unsigned int item_group = mf_info->item_group_idx[mf_info->R[i].i];
    //     // grid_rating_cnt[user_group * mf_info->item_group_num + item_group]++;
    //     user_group_rating_cnt[user_group]++;
    //     item_group_rating_cnt[item_group]++;
    // }

    // for (int i = 0; i < mf_info->max_user; i++){
    //     user_group_cnt[mf_info->user_group_idx[mf_info->user2idx[i]]]++;
    // }
    // for (int i = 0; i < mf_info->max_item; i++){
    //     item_group_cnt[mf_info->item_group_idx[mf_info->item2idx[i]]]++;
    // }

    // unsigned int total = 0;
    // filep << "The num of users per user groups" << endl;
    // for (int i = 0; i < mf_info->user_group_num; i++){
    //     total+=user_group_cnt[i];
    //     filep << user_group_cnt[i] << "\t";
    // }
    // filep << total << endl;
    // total = 0;
    // filep << "The num of items per item groups" << endl;
    // for (int i = 0; i < mf_info->item_group_num; i++){
    //     total+=item_group_cnt[i];
    //     filep << item_group_cnt[i] << "\t";
    // }
    // filep << total << endl;

    // // cout << "\nCheck grid rating cnt" << endl;
    // // total = 0;
    // // for (int i = 0; i < mf_info->user_group_num * mf_info->item_group_num; i++){
    // //     total+= grid_rating_cnt[i];
    // //     cout << grid_rating_cnt[i] << " ";
    // // }
    // // cout << "*" << total << endl;
    // // cout << "\n";
    
    
    // total = 0;
    // filep << "The num of ratings per user groups" << endl;
    // for (int i = 0; i < mf_info->user_group_num; i++){
    //     total+=user_group_rating_cnt[i];
    //     filep << user_group_rating_cnt[i] << "\t";
    // }
    // filep << total << endl;
    // total = 0;
    // filep << "The num of ratings per item groups" << endl;
    // for (int i = 0; i < mf_info->item_group_num; i++){
    //     total+=item_group_rating_cnt[i];
    //     filep << item_group_rating_cnt[i] << "\t";
    // }
    // filep << total << endl;

    //! ============================== print file =========================================
    unsigned int* user_group_cnt = new unsigned int[mf_info->user_group_num];
    for (int i = 0; i <mf_info->user_group_num; i++) user_group_cnt[i] = 0;
    unsigned int* item_group_cnt = new unsigned int[mf_info->item_group_num];
    for (int i = 0; i <mf_info->item_group_num; i++) item_group_cnt[i] = 0;
    unsigned int* user_group_rating_cnt = new unsigned int[mf_info->user_group_num];
    for (int i = 0; i <mf_info->user_group_num; i++) user_group_rating_cnt[i] = 0;
    unsigned int* item_group_rating_cnt = new unsigned int[mf_info->item_group_num];
    for (int i = 0; i <mf_info->item_group_num; i++) item_group_rating_cnt[i] = 0;
    // unsigned int* grid_rating_cnt = new unsigned int[mf_info->user_group_num * mf_info->item_group_num];
    // for (int i = 0; i <mf_info->user_group_num * mf_info->item_group_num; i++) grid_rating_cnt[i] = 0;


    for (int i = 0; i < mf_info->n; i++) {
        unsigned int user_group = mf_info->user_group_idx[mf_info->R[i].u];
        unsigned int item_group = mf_info->item_group_idx[mf_info->R[i].i];
        // grid_rating_cnt[user_group * mf_info->item_group_num + item_group]++;
        user_group_rating_cnt[user_group]++;
        item_group_rating_cnt[item_group]++;
    }

    for (int i = 0; i < mf_info->max_user; i++){
        user_group_cnt[mf_info->user_group_idx[mf_info->user2idx[i]]]++;
    }
    for (int i = 0; i < mf_info->max_item; i++){
        item_group_cnt[mf_info->item_group_idx[mf_info->item2idx[i]]]++;
    }

    filep << "Num of user groups : " << mf_info->user_group_num << endl;
    filep << "Num of item groups : " << mf_info->item_group_num << endl;

    unsigned int total = 0;
    filep << "The num of users per user groups" << endl;
    for (int i = 0; i < mf_info->user_group_num; i++){
        total+=user_group_cnt[i];
        filep << user_group_cnt[i] << "\t";
    }
    filep << total << endl;
    total = 0;
    filep << "The num of items per item groups" << endl;
    for (int i = 0; i < mf_info->item_group_num; i++){
        total+=item_group_cnt[i];
        filep << item_group_cnt[i] << "\t";
    }
    filep << total << endl;    
    
    total = 0;
    filep << "The num of ratings per user groups" << endl;
    for (int i = 0; i < mf_info->user_group_num; i++){
        total+=user_group_rating_cnt[i];
        filep << user_group_rating_cnt[i] << "\t";
    }
    filep << total << endl;
    total = 0;
    filep << "The num of ratings per item groups" << endl;
    for (int i = 0; i < mf_info->item_group_num; i++){
        total+=item_group_rating_cnt[i];
        filep << item_group_rating_cnt[i] << "\t";
    }
    filep << total << endl;
    filep.close();



    //! Print on shell 
    // for (int i = 0; i < mf_info->n; i++) {
    //     unsigned int user_group = mf_info->user_group_idx[mf_info->R[i].u];
    //     unsigned int item_group = mf_info->item_group_idx[mf_info->R[i].i];
    //     // grid_rating_cnt[user_group * mf_info->item_group_num + item_group]++;
    //     user_group_rating_cnt[user_group]++;
    //     item_group_rating_cnt[item_group]++;
    // }

    // for (int i = 0; i < mf_info->max_user; i++){
    //     user_group_cnt[mf_info->user_group_idx[mf_info->user2idx[i]]]++;
    // }
    // for (int i = 0; i < mf_info->max_item; i++){
    //     item_group_cnt[mf_info->item_group_idx[mf_info->item2idx[i]]]++;
    // }

    // unsigned int total = 0;
    // cout << "The num of users per user groups" << endl;
    // for (int i = 0; i < mf_info->user_group_num; i++){
    //     total+=user_group_cnt[i];
    //     cout << user_group_cnt[i] << "\t";
    // }
    // cout << total << endl;
    // total = 0;
    // cout << "The num of items per item groups" << endl;
    // for (int i = 0; i < mf_info->item_group_num; i++){
    //     total+=item_group_cnt[i];
    //     cout << item_group_cnt[i] << "\t";
    // }
    // cout << total << endl;    
    
    // total = 0;
    // cout << "The num of ratings per user groups" << endl;
    // for (int i = 0; i < mf_info->user_group_num; i++){
    //     total+=user_group_rating_cnt[i];
    //     cout << user_group_rating_cnt[i] << "\t";
    // }
    // cout << total << endl;
    // total = 0;
    // cout << "The num of ratings per item groups" << endl;
    // for (int i = 0; i < mf_info->item_group_num; i++){
    //     total+=item_group_rating_cnt[i];
    //     cout << item_group_rating_cnt[i] << "\t";
    // }
    // cout << total << endl;
}
