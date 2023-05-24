# AuSamNet
## 如何使用？ How to use? 
1、直接运行eval.py，得到测试集数据对应的网络输出结果。
1. Run eval.py directly to get the network output corresponding to the test dataset.
2、直接运行eval_exp.py，得到实验数据对应的网络输出结果。
2. Run eval_exp.py directly to get the network output corresponding to the experimental data.
## 各文件说明： Description of each document:
### Results：
1、ExpResults：为实验数据，其中RawExpData.mat为原始桶信号数据，IntensityMat.mat为实验预处理数据，具体使用方法可参考https://github.com/zibangzhang/Real-time-Fourier-single-pixel-imaging 。
1. ExpResults：Is the experimental data, where RawExpData.mat is the raw bucket signal data, IntensityMat.mat is the experimental preprocessed data, The specific method of use https://github.com/zibangzhang/Real-time-Fourier-single-pixel-imaging for reference.
2、M2：为论文中”FSI-DL”方案所对应的各个采样率下的网络结果。
2. M2: is the network results under each sampling rate corresponding to the "FSI-DL" scheme in the paper.
3、M3：为论文AuSamNet方案所对应的各个采样率下的网络结果。
3. M3: shows the network results under each sampling rate corresponding to the AuSamNet scheme of the paper.
### Data：
为CelebA数据集的一小部分，若需要全部数据可参考http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html 。
For CelebA a small part of the data set, if need full data may refer to http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
### Debayer：
参考了https://github.com/cheind/pytorch-debayer，感谢您的贡献。
Refer to the https://github.com/cheind/pytorch-debayer, thank you for your contribution.
### create_data_lists.py：
用于创建.json文件。
Used to create.json files.
### dataset.py：
对数据集的预处理文件。
A pre-processed file for the dataset.
### models.py：
模型结构。
model structure.
### train.py：
训练模型。
training model.
### eval.py：
计算测试集数据的模型输出结果。
Compute the model output on the test set data.
### eval_exp.py：
计算实验数据的模型输出结果。
Calculate the model output of the experimental data.
### utils.py：
模型中使用的一些子函数。
Some of the subfunctions used in the model.
## 注意事项： Notes: 
在上述文件中，M1对应论文中的"FSI"方法，M2对应论文中的“FSI-DL”方法，M3对应论文中的“AuSamNet”方法。
In the above files, M1 corresponds to the FSI scheme; M2 corresponds to the FSI-DL scheme; M3 corresponds to our proposed AuSamNet scheme.



