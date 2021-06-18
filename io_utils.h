#ifndef IO_UTILS_H
#include <iostream>
#include "common_struct.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
#include <set>
#include <boost/filesystem.hpp>
#define IO_UTILS_H
using namespace std;

vector<string>& split(const string &s, char delim, vector<string> &elems){
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> split(const string &s, char delim){
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void read_training_dataset(Mf_info *mf_info, string infile){
    ifstream file;
    string line;
    vector<Node> COO;
    unsigned int user_idx, item_idx, user, item, cnt;
    float rating; 

    const char* data = infile.c_str();
    file.open(data);

    if (strstr(data, "netflix") != NULL || strstr(data, "25M") != NULL){
        user_idx = 1;
        item_idx = 0;
    }
    else {
        user_idx = 0;
        item_idx = 1;
    }
    
    if (strstr(data, "Yahoo") != NULL){
        while(getline(file, line)){
            vector<string> tokens = split(line, '|');
            user = stoi(tokens[0]);
            cnt = stoi(tokens[1]);
            
            pair<map<unsigned int, unsigned int>::iterator, bool> ret_user;
            ret_user = mf_info->user_map.insert(pair<unsigned int, unsigned int> (user, mf_info->max_user));
            unsigned int user_seq_idx = ret_user.first->second;
            
            if(ret_user.second){
                mf_info->user_map2orig[mf_info->max_user] = user;
                mf_info->max_user++;
            }
            for (int i = 0; i < cnt; i++){
                getline(file, line);
                vector<string> tokens = split(line, '\t');
                item = stoi(tokens[0]);
                rating = atof(tokens[1].c_str());
                pair<map<unsigned int, unsigned int>::iterator, bool> ret_item;
                ret_item = mf_info->item_map.insert(pair<unsigned int, unsigned int> (item, mf_info->max_item));
                unsigned int item_seq_idx = ret_item.first->second;
                if(ret_item.second){
                    mf_info->item_map2orig[mf_info->max_item] = item;
                    mf_info->max_item++;
                }
                COO.push_back({(rating/25.f)+1.0f, user_seq_idx, item_seq_idx});
                mf_info->n++;
            }
        }
    }
    else {
        while(getline(file, line)){
            vector<string> tokens = split(line, '\t');
            if(tokens.size() == 3){	
                user = stoi(tokens[user_idx]);
                item = stoi(tokens[item_idx]);

                string s = tokens[2];
                s.erase(s.find_last_not_of(" \n\r\t")+1);
                rating = atof(s.c_str());
                pair<map<unsigned int, unsigned int>::iterator, bool> ret_user;
                pair<map<unsigned int, unsigned int>::iterator, bool> ret_item;

                ret_user = mf_info->user_map.insert(pair<unsigned int, unsigned int> (user, mf_info->max_user));
                ret_item = mf_info->item_map.insert(pair<unsigned int, unsigned int> (item, mf_info->max_item));

                unsigned int user_seq_idx = ret_user.first->second;
                unsigned int item_seq_idx = ret_item.first->second;

                if(ret_user.second){
                    mf_info->user_map2orig[mf_info->max_user] = user;
                    mf_info->max_user++;
                }
                if(ret_item.second){
                    mf_info->item_map2orig[mf_info->max_item] = item;
                    mf_info->max_item++;
                }

                COO.push_back({rating, user_seq_idx, item_seq_idx});
                mf_info->n++;
            }        
        }
    }

    file.close();
    mf_info->R = new Node[mf_info->n];
    copy(COO.begin(), COO.end(), mf_info->R);
    COO.clear();

    // return mf_info;
}

void read_test_dataset(Mf_info *mf_info, string infile){
    vector<unsigned int> remove_user;
    vector<unsigned int> remove_item;
    
    const char* data = infile.c_str();
    string line;
    unsigned int user_idx, item_idx, user, item, cnt, missing = 0;
    float rating; 

    ifstream file;
    file.open(data);

    mf_info->test_R.resize(mf_info->max_user);
    
    if (strstr(data, "netflix") != NULL || strstr(data, "25M") != NULL){
        user_idx = 1;
        item_idx = 0;
    }
    else {
        user_idx = 0;
        item_idx = 1;
    }

    if (strstr(data, "Yahoo") != NULL){
        while(getline(file, line)){
            vector<string> tokens = split(line, '|');
            user = stoi(tokens[0]);
            cnt = stoi(tokens[1]);
            map<unsigned int, unsigned int>::iterator it_user;
            it_user = mf_info->user_map.find(user);
            if (it_user == mf_info->user_map.end()){
                missing += cnt;
                continue;
            }
            for (int i = 0; i < cnt; i++){
                getline(file, line);
                vector<string> tokens = split(line, '\t');
                item = stoi(tokens[0]);
                rating = atof(tokens[1].c_str());
                map<unsigned int, unsigned int>::iterator it_item;
                it_item = mf_info->item_map.find(item);
                if (it_item == mf_info->item_map.end()){
                    missing += 1;
                    continue;
                }
                mf_info->test_R[it_user->second][it_item->second] = (rating/25.f)+1.0f;
                mf_info->test_n++;
            }
        }
    }else{
        while(getline(file, line)){
            vector<string> tokens = split(line, '\t');
            if(tokens.size() == 3){	
                
                user = stoi(tokens[user_idx]);
                item = stoi(tokens[item_idx]);

                string s = tokens[2];
                s.erase(s.find_last_not_of(" \n\r\t")+1);
                rating = atof(s.c_str());

                map<unsigned int, unsigned int>::iterator it_user;
                map<unsigned int, unsigned int>::iterator it_item;

                it_user = mf_info->user_map.find(user);
                it_item = mf_info->item_map.find(item);

                if ((it_user != mf_info->user_map.end()) && (it_item != mf_info->item_map.end())){
                    mf_info->test_R[it_user->second][it_item->second] = rating;
                    mf_info->test_n++;
                }else if (it_user == mf_info->user_map.end()){
                    missing++;
                    remove_user.push_back(user);
                }else{
                    missing++;
                    remove_item.push_back(item);
                }
            }
        }
    }

    file.close();  
    cout << "Missing the number of ratings : " << missing << endl;
    
    boost::filesystem::path p(infile);
    string dir = p.parent_path().string();
    string outpath_file = dir + string("/remove_user.txt");
    
    const char* pt_u = outpath_file.c_str();
    ofstream filep_u;
    filep_u.open(pt_u, std::ofstream::out);

    if (filep_u.fail()){
        cout << "fail to write file named " << pt_u << endl;
        return;
    }

    for (int i = 0; i < remove_user.size(); i++){
        filep_u << remove_user[i] << endl;
    }

    filep_u.close();
    outpath_file = dir + string("/remove_item.txt");
    
    const char* pt_i = outpath_file.c_str();
    ofstream filep_i;
    filep_i.open(pt_i, std::ofstream::out);

    if (filep_i.fail()){
        cout << "fail to write file named " << pt_i << endl;
        return;
    }

    for (int i = 0; i < remove_item.size(); i++){
        filep_i << remove_item[i] << endl;
    }
    filep_i.close();

}

void save_trained_model(Mf_info* mf_info, SGD* sgd_info, string outfile){
    cout << "Save trained model..." << endl;
    string outpath = outfile + string(".txt");
    const char* pt = outpath.c_str();
    ofstream filep;
    filep.open(pt);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        cerr << "Error: " << strerror(errno);
        return;
    }
    
    unsigned int max_user = mf_info->user_map.rbegin()->first;
    unsigned int max_item = mf_info->item_map.rbegin()->first;
    
    filep << max_user+1 << " " << max_item+1 << " " << mf_info->params.k << endl;
    int prev = 0;
    for (auto itr = mf_info->user_map.begin(); itr != mf_info->user_map.end(); itr++){
        int i = itr->first;
        int i2 = itr->second;

        for (int c = prev; c <= i-1; c++){
            for (int j = 0; j < mf_info->params.k; j++){
                filep << 0 << " ";
            }
        }
        for (int j = 0; j < mf_info->params.k; j++){
            filep << sgd_info->p[i2 * mf_info->params.k + j] << " ";
        }    
        prev = i+1;
    }
    filep << endl;
    prev = 0;
    for (auto itr = mf_info->item_map.begin(); itr != mf_info->item_map.end(); itr++){
        int i = itr->first;
        int i2 = itr->second;
        for (int c = prev; c <= i-1; c++){
            for (int j = 0; j < mf_info->params.k; j++){
                filep << 0 << " ";
            }
        }
        for (int j = 0; j < mf_info->params.k; j++){
            filep << sgd_info->q[i2 * mf_info->params.k + j] << " ";
        }
        prev = i+1;
    }

    // for (int i = 0; i < mf_info->max_user; i++){
    //     for (int j = 0; j < mf_info->params.k; j++){
    //         filep << sgd_info->p[i * mf_info->params.k + j] << " ";
    //     }
    // }
    // filep << endl;
    // for (int i = 0; i < mf_info->max_item; i++){
    //     for (int j = 0; j < mf_info->params.k; j++){
    //         filep << sgd_info->q[i * mf_info->params.k + j] << " ";
    //     }
    // }

    filep.close();
}

void save_trained_model_reconst(Mf_info* mf_info, SGD* sgd_info, string outfile){
    cout << "Save trained reconst model..." << endl;
    string outpath = outfile + string("_reconst.txt");
    const char* pt = outpath.c_str();
    ofstream filep;
    filep.open(pt);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        cerr << "Error: " << strerror(errno);
        return;
    }
    
    filep << mf_info->max_user << " " << mf_info->max_item << " " << mf_info->params.k << endl;
    for (int i = 0; i < mf_info->max_user; i++){
        for (int j = 0; j < mf_info->params.k; j++){
            filep << sgd_info->p[i * mf_info->params.k + j] << " ";
        }
    }
    filep << endl;
    for (int i = 0; i < mf_info->max_item; i++){
        for (int j = 0; j < mf_info->params.k; j++){
            filep << sgd_info->q[i * mf_info->params.k + j] << " ";
        }
    }

    filep.close();
}

void read_trained_model(Mf_info* mf_info, SGD* sgd_info, string infile){
    const char* data = infile.c_str();
    ifstream filep;
    string line;
    vector<string> tokens;

    filep.open(data);
    getline(filep, line);
    tokens = split(line, ' ');
    
    unsigned int user_num = stoi(tokens[0]);
    unsigned int item_num = stoi(tokens[1]);
    string s = tokens[2];
    s.erase(s.find_last_not_of(" \n\r\t")+1);
    unsigned int k = atoi(s.c_str());
    sgd_info->p = new float[user_num * k];
    sgd_info->q = new float[item_num * k];

    float inp;
    for (int i = 0; i < user_num; i++){
        for (int j = 0; j < k; j++){
            filep >> inp;
            sgd_info->p[i * k + j] = inp;
        }
    }

    for (int i = 0; i < item_num; i++){
        for (int j = 0; j < k; j++){
            filep >> inp;
            sgd_info->q[i * k + j] = inp;
        }
    }

    mf_info->params.k = k;
    mf_info->max_user = user_num;
    mf_info->max_item = item_num;
    
    filep.close();
}

void save_reconst_testset(Mf_info *mf_info, string testfile){
    cout << "Save reconst mat.." << endl;
    boost::filesystem::path p(testfile);
    string dir = p.parent_path().string();
    string file_name = p.filename().string();
    string outpath_file = dir + string("/reconst_") + file_name;

    const char* pt = outpath_file.c_str();
    ofstream filep;
    filep.open(pt, std::ofstream::out);

    if (filep.fail()){
        cout << "fail to write file named " << pt << endl;
        return;
    }
    
    bool user_first = true;
    
    if (strstr(pt, "netflix") != NULL || strstr(pt, "25M") != NULL){
        user_first = false;
    }

    for (int j = 0; j < mf_info->test_n; j++){
        unsigned int u = mf_info->test_COO[j].u;
        unsigned int i = mf_info->test_COO[j].i;
        float r = mf_info->test_COO[j].r;
        if (user_first) filep << u << "\t" << i << "\t" << r << endl;
        else filep << i << "\t" << u << "\t" << r << endl;
    }

    filep.close();
}

vector<Node> read_testset_pretrained_model(Mf_info *mf_info, string testfile){
    const char* data = testfile.c_str();
    ifstream filep;
    string line;
    filep.open(data);
    vector<string> tokens;
    vector<Node> test_set;
    unsigned int u, i, cnt;
    float r;
    unsigned int user_idx, item_idx;
    if (strstr(data, "netflix") != NULL || strstr(data, "25M") != NULL){
        user_idx = 1;
        item_idx = 0;
    }
    else {
        user_idx = 0;
        item_idx = 1;
    }

    if (strstr(data, "Yahoo") != NULL && strstr(data, "reconst") == NULL){
        while(getline(filep, line)){
            tokens = split(line, '|');
            u = stoi(tokens[0]);
            cnt = stoi(tokens[1]);
            for (int j = 0; j < cnt; j++){
                getline(filep, line);
                tokens = split(line, '\t');
                i = stoi(tokens[0]);
                r = (atof(tokens[1].c_str())/25.f)+1.0f;
                test_set.push_back({r, u , i});
            }
        }
    }else{
        while(getline(filep, line)){
            tokens = split(line, '\t');
            if(tokens.size() == 3){	
                u = stoi(tokens[user_idx]);
                i = stoi(tokens[item_idx]);
                string s = tokens[2];
                s.erase(s.find_last_not_of(" \n\r\t")+1);
                r = stof(s);
                test_set.push_back({r, u , i});
            }
        }
    }


    filep.close();
    return test_set;
}

void remove_elements(Mf_info* mf_info, vector<Node> test_set, unsigned int version ,string test_set_path){
    set<unsigned int> remove_user;
    set<unsigned int> remove_item;

    if (version != 1){
        boost::filesystem::path p(test_set_path);
        string dir = p.parent_path().string();
        string inpath_file = dir + string("/remove_user.txt");
        
        const char* pt_u = inpath_file.c_str();
        ifstream filep_u;
        filep_u.open(pt_u);

        if (filep_u.fail()){
            cout << "fail to write file named " << pt_u << endl;
            return;
        }

        inpath_file = dir + string("/remove_item.txt");
        const char* pt_i = inpath_file.c_str();
        ifstream filep_i;
        filep_i.open(pt_i);

        if (filep_i.fail()){
            cout << "fail to write file named " << pt_i << endl;
            return;
        }

        string line; 
        while (getline(filep_u, line)) {
            line.erase(line.find_last_not_of(" \n\r\t")+1);
            remove_user.insert(stoi(line));
        }

        while (getline(filep_i, line)) {
            line.erase(line.find_last_not_of(" \n\r\t")+1);
            remove_item.insert(stoi(line));
        }

        filep_u.close();
        filep_i.close();
    }

    unsigned int nnz = test_set.size();
    mf_info->test_COO = new Node[nnz];

    for (unsigned int j = 0; j < nnz; j++){
        unsigned int u = test_set[j].u;
        unsigned int i = test_set[j].i;

        if (remove_user.find(u) != remove_user.end() || remove_item.find(i) != remove_item.end()) continue; 
        mf_info->test_COO[mf_info->test_n].u = u;
        mf_info->test_COO[mf_info->test_n].i = i;
        mf_info->test_COO[mf_info->test_n].r = test_set[j].r;

        mf_info->test_n++;
    }
}
#endif