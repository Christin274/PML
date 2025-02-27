# Outlier Interpretation

### Six Outlier Interpretation Methods

**This repository contains seven outlier interpretation methods: ATON [1], SiNNE[3], SHAP[4], LIME[5], Anchor [6] and PARs[7].**

[1] Beyond Outlier Detection: Outlier Interpretation by Attention-Guided Triplet Deviation Network. In WWW. 2021.

[2] A new effective and efficient measure for outlying aspect mining. arXiv preprint arXiv:2004.13550. 2020.

[3] A unified approach to interpreting model predictions. In NeuraIPS. 2017

[4] "Why should I trust you?" Explaining the predictions of any classifier. In SIGKDD. 2016.

[5] Anchors: High Precision Model-Agnostic Explanations. In AAAI. 2018.

[6] PARs: Predicate-based Association Rules for Efficient and Accurate Anomaly Explanation. In CIKM. 2024



### Structure
`data_od_evaluation`: Ground-truth outlier interpretation annotations of real-world datasets  
`data`: real-world datasets in csv format, the last column is label indicating each line is an outlier or a inlier  
`model_xx`: folders of ATON and its contenders, the competitors are introduced in Section 5.1.2  
`config.py`: configuration and default hyper-parameters  
`main.py` main script to run the experiments



### How to use?
##### 1. For PML and competitor ATON, COIN, SHAP, and LIME
1. modify variant `algorithm_name` in `main.py` (support algorithm: `pml`, `aton`, `shap`, `lime`  in lowercase)
2. use `python main.py --path data/ --runs  [times]`
3. the results can be found in `record/[algorithm_name]/` folder  

##### 2. For PML',
1. modify variant `algorithm_name` in `main.py` to `aton` or `coin` or `pml` 
2. use `python main.py --path data/ --w2s_ratio auto --runs [times]` to run PML'   

##### 3. For competitor SiNNE, Anchors
1. modify variant `algorithm_name` in `main2.py` to `sinne` or `anchor`  
please run `python main2.py --path data/ --runs [times]` 

##### 4. For competitor PARs
coming soon


### args of main.py
- `--path [str]`        - the path of data folder or an individual data file (in csv format)  

- `--gpu  [True/False]` - use GPU or not

- `--runs [int]`         - how many times to run a method on each dataset (we run 10 times and report average performance in our submission)

- `--w2s_ratio [auto/real_len/pn]`  - how to transfer feature weight to feature subspace 'real-len', 'auto', or 'pn' 
denote the same length with the ground-truth, auto generating subspace by the proposed threshold or positive-negative.
(in our paper, we use 'auto' in ATON'. As for methods which output, we directly use 'real-len'.)

- `--eval [True/False]` - evaluate or not, use False for scalability test  
  ... (other hypter-parameters of different methods. You may want to use -h to check the corresponding hypter-parameters after modifing the `algorithm_name`)  

  

### Requirements
main packages of this project  
```
torch==1.3.0
numpy==1.15.0
pandas==0.25.2
scikit-learn==0.23.1
pyod==0.8.2
tqdm==4.48.2
prettytable==0.7.2
shap==0.35.0
lime==0.2.0.1
alibi==0.5.5
```



