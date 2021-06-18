# MASCOT
## Abstract
In recent years, quantization methods have successfully accelerated the training of large DNN models by reducing the level of precision in computing operations (e.g., forward/backward passes) without sacrificing its accuracy. In this work, therefore, we attempt to apply such a quantization idea to the popular Matrix factorization (MF) methods to deal with  the growing scale of models and datasets in recommender systems. However, to our dismay, we observe that the state-of-the-art quantization methods are not effective in the training of MF models, unlike their successes in the training of DNN models. To this phenomenon, we posit that two 
distinctive features in training MF models could explain the difference: (i) training MF models is much more memory-intensive than training DNN models, and (ii) the quantization errors across users and items in recommendation are not uniform. From these observations, then, we develop a novel quantization framework for MF models, named as MASCOT, employing novel strategies to successfully address two aforementioned unique features in the training of MF models. The comprehensive evaluation using four real-world datasets demonstrates that MASCOT improves the training performance of MF models by about 45%, compared to the training without quantization, while maintaining low model errors, and the strategies and implementation optimizations of MASCOT are quite effective in the training of MF models.

## Building
This project is written in standard C++ and CUDA. it can be built by running Makefile in the source code directory.

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
We used an RTX 2070 GPU for our experiments and set the number of warps to 2,048 (k = 128), 2304 (k = 64)  
Other parameter settings are described in the paper.  








