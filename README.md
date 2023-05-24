# AuSamNet
## How to use? 
1. Run eval.py directly to get the network output corresponding to the test dataset.
2. Run eval_exp.py directly to get the network output corresponding to the experimental data.
## Description of each document:
### Results：
1. ExpResults：Is the experimental data, where RawExpData.mat is the raw bucket signal data, IntensityMat.mat is the experimental preprocessed data, The specific method of use https://github.com/zibangzhang/Real-time-Fourier-single-pixel-imaging for reference.
2. M2: is the network results under each sampling rate corresponding to the "FSI-DL" scheme in the paper.
3. M3: shows the network results under each sampling rate corresponding to the AuSamNet scheme of the paper.
### Data：
For CelebA a small part of the data set, if need full data may refer to http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html.
### Debayer：
Refer to the https://github.com/cheind/pytorch-debayer, thank you for your contribution.
### create_data_lists.py：
Used to create.json files.
### dataset.py：
A pre-processed file for the dataset.
### models.py：
model structure.
### train.py：
training model.
### eval.py：
Compute the model output on the test set data.
### eval_exp.py：
Calculate the model output of the experimental data.
### utils.py：
Some of the subfunctions used in the model.
## Notes: 
In the above files, M1 corresponds to the FSI scheme; M2 corresponds to the FSI-DL scheme; M3 corresponds to our proposed AuSamNet scheme.
Some files are named in the following format: 
Actual sampling ratio ---------  Naming of M2 and M3 --------- Naming of other methods such as DCAN
7.5% ------------------------------------ 614 ------------------------------------------- 1228
15% -------------------------------------1228--------------------------------------------2457
22.5% ----------------------------------- 1843-------------------------------------------3686
30%------------------------------------- 2457--------------------------------------------4915 
37.5%------------------------------------ 3072-------------------------------------------6144



