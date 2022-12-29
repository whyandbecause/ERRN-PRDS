## End-to-End Rain Removal Network Based on Progressive Residual Detail Supplement 
[paper:https://ieeexplore.ieee.org/document/9388918]

### Abstract
Methods of rain removal based on deep learning have rapidly developed, and the image quality after rain removal is continuously improving. However, the results of most methods have some common problems, including a loss of details, a blurring of edges, and the existence of artifacts. To remove rain-related information more thoroughly and retain more edge details, this paper proposes an end-to-end rain removal network based on the progressive residual detail supplement (ERRN-PRDS) approach. The entire network structure is designed in an iterative manner to obtain higher-quality rain removal images from coarse to fine. In the network, a diamond residual block is constructed as the main module of iteration to learn the feature information of the background layer. Meanwhile, to keep more texture details in the background layer, a detail supplement mechanism is designed between the iterative layers to transfer more information to the next iterative operation. Experimental results show that this method can remove the rain information more completely and better retain the image edges compared with previous state-of-the-art methods. In addition, because of the sparsity of the detail injection, our network also achieves high-quality results for image denoising tasks.


## Prerequisites
- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- MATLAB for computing [evaluation metrics](statistic/)


## Datasets

To train the models, please download training datasets: https://github.com/nnUyi/DerainZoo


## Getting Started

### 1) Testing

We have provided our pre-trained models in `./logs/`. 
diedai10 is provided for heavy rain, diedai10_L is provided for light rain.

Please change your own parameters in code, like  --logdir --data_path  --save_path ......

Run python scripts to test the models:
```bash
python test.py   # test models on synthetic dataset
```
you can directly compute all the [evaluation metrics](statistic/) in this paper.  

### 2) Evaluation metrics

We also provide the MATLAB scripts to compute the average PSNR and SSIM values reported in the paper.
 

```Matlab
 cd ./statistic
 run statistic_Rain100H.m
 run statistic_Rain100L.m
 run statistic_Rain12.m
```
###
Average PSNR/SSIM values on three datasets:

Dataset    | RESCAN    |PReNet     |Ours     
-----------|-----------|-----------|-----------|
Rain100H   |28.82/0.863|29.46/0.899|31.67/0.927|
Rain100L   |39.22/0.983|37.48/0.979|41.02/0.989|
Rain12     |34.79/0.958|36.15/0.969|36.99/0.971|

### 3) Training

Run python scripts to train the models:
```bash
python train.py 
```
Please change your own parameters in code.
You can use `tensorboard --logdir ./logs/your_model_path` to check the training procedures. 

### Model Configuration

The following tables provide the configurations of options. 

#### Training Mode Configurations

Option                 |Default        | Description
-----------------------|---------------|------------
batchSize              | 12            | Training batch size
recurrent_iter         | 10             | Number of recursive stages
epochs                 | 100           | Number of training epochs
milestone              | [30,50,80]    | When to decay learning rate
lr                     | 1e-3          | Initial learning rate
save_freq              | 5            | save intermediate model
use_GPU                | True          | use GPU or not
gpu_id                 | 0             | GPU id
data_path              | N/A           | path to training images
save_path              | N/A           | path to save models and status           

#### Testing Mode Configurations

Option                 |Default           | Description
-----------------------|------------------|------------
use_GPU                | True             | use GPU or not
gpu_id                 | 0                | GPU id
recurrent_iter         | 10                | Number of recursive stages
logdir                 | N/A              | path to trained model
data_path              | N/A              | path to testing images
save_path              | N/A              | path to save results

## References
[1] Ren, Dongwei and Zuo, Wangmeng and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu. Progressive Image Deraining Networks: A Better and Simpler Baseline. In IEEE CVPR 2019.
## Acknowledgement
Part of our code is borrowed from PReNet(https://github.com/csdwren/PReNet), Thanks for the sharing of codes by Dongwei Ren.

# Citation

```
@ARTICLE{9388918,
  author={Yang, Yong and Guan, Juwei and Huang, Shuying and Wan, Weiguo and Xu, Yating and Liu, Jiaxiang},
  journal={IEEE Transactions on Multimedia}, 
  title={End-to-End Rain Removal Network Based on Progressive Residual Detail Supplement}, 
  year={2022},
  volume={24},
  number={},
  pages={1622-1636},
  doi={10.1109/TMM.2021.3068833}}
 ```
