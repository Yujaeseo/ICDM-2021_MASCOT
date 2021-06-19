# MASCOT
## Introduction
In recent years, quantization methods have successfully accelerated the training of large DNN models by reducing the level of precision in computing operations (e.g., forward/backward passes) without sacrificing its accuracy. In this work, therefore, we attempt to apply such a quantization idea to the popular Matrix factorization (MF) methods to deal with  the growing scale of models and datasets in recommender systems. However, to our dismay, we observe that the state-of-the-art quantization methods are not effective in the training of MF models, unlike their successes in the training of DNN models. To this phenomenon, we posit that two 
distinctive features in training MF models could explain the difference: (i) training MF models is much more memory-intensive than training DNN models, and (ii) the quantization errors across users and items in recommendation are not uniform. From these observations, then, we develop a novel quantization framework for MF models, named as MASCOT, employing novel strategies to successfully address two aforementioned unique features in the training of MF models. The comprehensive evaluation using four real-world datasets demonstrates that MASCOT improves the training performance of MF models by about 45%, compared to the training without quantization, while maintaining low model errors, and the strategies and implementation optimizations of MASCOT are quite effective in the training of MF models.

## Building
This project is written in standard C++ and CUDA 10.2. it can be built by running Makefile in the source code directory.

## Run
Run executable file by:  
  ```./quantized_mf -i [train file] -y [test file] -o [output file] [options]```  


Where options are as follows:    
  > -l  : The number of epochs executed during training  
  -k  : The dimensionality of latent space (64 or 128)  
  -b  : Regularization parameter for users and items  
  -a  : Initial learning rate  
  -d  : Decay factor  
  -wg : The number of warps launched during update  
  -b  : The number of threads per block  
  -ug : The number of user groups  
  -ig : The number of item groups  
  -r  : Sampling ratio  
  -it : Error estimate period  
  -e  : Error threshold  
  -rc : Whether to save reconstructed testset matrix  
  -v  : MF version to run (1-MASCOT, 2-AFP, 3-MUPPET, 4-MPT, 5- FP32)
  
It is recommended to tune the number of threads using -wg options to maximize the performance.  
We used an RTX 2070 GPU for our experiments and set the number of warps to 2,048 (k = 128), 2,304 (k = 64)  
Other parameter settings are described in the paper.  

## Datasets

In our experiments, we used four real-world datasets for training and testing 
- MovieLens 10M  
- MovieLens 25M  
- Netflix
- Yahoo!Music

## Test

We provide pre-trained model and you can test it as follows:  

  ```./test_mf -i [pre-trained model file] -y [test file] -v [mf version]```  

## Experimental results
RQ1. Does MASCOT improve the training performance of MF models more than existing quantization methods?  


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/Performance%20comparison.png" width="470" height="400">




RQ2. Does MASCOT provide the errors of MF models lower than existing quantization methods? 


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/RMSE%20comparison.png" width="470" height="400">  


RQ3. How effective are the strategies and optimizations of MASCOT in improving the MF model training?


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/quantization%20optimization.png" width="850" height="380">  


RQ4. How sensitive are the training performance and model error of MASCOT to its hyperparameters?  


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/hyperparameter%20sensitivity.png" width="850" height="400">  


<img src="https://github.com/Yujaeseo/MASCOT/blob/main/Figure/hyperparameter%20sensitivity2.png" width="850" height="380">  

You can produce those result using following commands :  
  - MASCOT  
    ```./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -ug 100 -ig 100 -e 20 -s 0.05 -it 2 -v 1```  
  - MPT  
    ```./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 4```  
  - MuPPET  
    ```./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -s 0.05 -v 3```  
  - AFP  
    ```./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 2```  
  - FP32  
    ```./quantized_mf -i [train file] -y [test file] -o [output file] -wg 2048 -bl 128 -k 128 -l 50 -a 0.01 -d 0.1 -v 5```  
    

    

