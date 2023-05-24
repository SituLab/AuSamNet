# AuSamNet
## 如何使用？
1、直接运行eval.py，得到测试集数据对应的网络输出结果。
2、直接运行eval_exp.py，得到实验数据对应的网络输出结果。
## 各文件说明：
### Results文件夹：
ExpResults：为实验数据，其中RawExpData.mat为原始桶信号数据，IntensityMat.mat为实验预处理数据，具体使用方法可参考https://github.com/zibangzhang/Real-time-Fourier-single-pixel-imaging 。
M2：为论文中”FSI-DL”方案所对应的各个采样率下的网络结果。
M3：为论文AuSamNet方案所对应的各个采样率下的网络结果。
### Data文件夹：
为CelebA数据集的一小部分，若需要全部数据可参考http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 。
### Debayer文件夹：
参考了https://github.com/cheind/pytorch-debayer，感谢您的贡献。
### create_data_lists.py：
用于创建.json文件。
### dataset.py：
对数据集的预处理文件。
### models.py：
模型结构。
### train.py：
训练模型。
### eval.py：
计算测试集数据的模型输出结果。
### eval_exp.py：
计算实验数据的模型输出结果、
### utils.py：
为模型中使用的一些子函数。
## 注意事项：
In this folder, M1 corresponds to the FSI scheme; M2 corresponds to the FSI-DL scheme; M3 corresponds to our proposed AuSamNet scheme.

Actual sampling ratio ---------  Naming of M2 and M3 --------- Naming of other methods such as DCAN
7.5% ------------------------------------ 614 ------------------------------------------- 1228
15% -------------------------------------1228--------------------------------------------2457
22.5% ----------------------------------- 1843-------------------------------------------3686
30%------------------------------------- 2457--------------------------------------------4915 
37.5%------------------------------------ 3072-------------------------------------------6144


The training was conducted in a computer with an Intel Xeon Gold 6226R (64) @ 3.900GHz, 
125 GB RAM, and an NVIDIA A100-PCIE-40GB GPU.


