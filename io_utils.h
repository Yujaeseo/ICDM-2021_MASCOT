#ifndef IO_UTILS_H
#include <iostream>
#include "common_struct.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <map>
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

Mf_info read_training_dataset(string infile){
    Mf_info mf_info; 
    ifstream file;
    string line;
    vector<Node> COO;
    unsigned int user_idx, item_idx, user, item, cnt;
    float rating; 

    const char* data = infile.c_str();
    file.open(data);

    if (strstr(data, "netflix") != NULL || strstr(data, "amazon") != NULL || strstr(data, "25M")){
        user_idx = 1;
        item_idx = 0;
    }
    else {
        user_idx = 0;
        item_idx = 1;
    }
    
    COO.reserve(100000000);
    if (strstr(data, "Yahoo") != NULL){
        while(getline(file, line)){
            vector<string> tokens = split(line, '|');
            user = stoi(tokens[0]);
            cnt = stoi(tokens[1]);
            
            pair<map<unsigned int, unsigned int>::iterator, bool> ret_user;
            ret_user = mf_info.user_map.insert(pair<unsigned int, unsigned int> (user, mf_info.max_user));
            unsigned int user_seq_idx = ret_user.first->second;
            
            if(ret_user.second){
                mf_info.user_map2orig[mf_info.max_user] = user;
                mf_info.max_user++;
            }
            // cout << "User : " << user_seq_idx << endl;
            for (int i = 0; i < cnt; i++){
                getline(file, line);
                vector<string> tokens = split(line, '\t');
                item = stoi(tokens[0]);
                rating = atof(tokens[1].c_str());
                pair<map<unsigned int, unsigned int>::iterator, bool> ret_item;
                ret_item = mf_info.item_map.insert(pair<unsigned int, unsigned int> (item, mf_info.max_item));
                unsigned int item_seq_idx = ret_item.first->second;
                if(ret_item.second){
                    mf_info.item_map2orig[mf_info.max_item] = item;
                    mf_info.max_item++;
                }
                // cout << "Item : " << item_seq_idx << endl;
                // cout << "Rating : " << (rating/25.f)+1.0f << endl;

                COO.push_back({(rating/25.f)+1.0f, user_seq_idx, item_seq_idx});
                mf_info.n++;
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

                ret_user = mf_info.user_map.insert(pair<unsigned int, unsigned int> (user, mf_info.max_user));
                ret_item = mf_info.item_map.insert(pair<unsigned int, unsigned int> (item, mf_info.max_item));

                unsigned int user_seq_idx = ret_user.first->second;
                unsigned int item_seq_idx = ret_item.first->second;

                if(ret_user.second){
                    mf_info.user_map2orig[mf_info.max_user] = user;
                    mf_info.max_user++;
                }
                if(ret_item.second){
                    mf_info.item_map2orig[mf_info.max_item] = item;
                    mf_info.max_item++;
                }

                COO.push_back({rating, user_seq_idx, item_seq_idx});
                mf_info.n++;
            }        
        }
    }

    file.close();
    mf_info.R = new Node[mf_info.n];
    copy(COO.begin(), COO.end(), mf_info.R);
    COO.clear();

    return mf_info;
}

void read_test_dataset(Mf_info *mf_info, string infile){
    const char* data = infile.c_str();
    string line;
    unsigned int user_idx, item_idx, user, item, cnt, missing = 0;
    float rating; 

    ifstream file;
    file.open(data);

    mf_info->test_R.resize(mf_info->max_user);
    
    if (strstr(data, "netflix") != NULL || strstr(data, "amazon") != NULL ||strstr(data, "25M") != NULL){
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
                }else{
                    missing++;
                }
            }
        }
    }

    file.close();  
    cout << "Missing the number of ratings : " << missing << endl;
}

#endif