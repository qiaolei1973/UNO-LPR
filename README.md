# UNO! UNified Offline Training Paradigm for Learning Path Recommendation
This repository is built for the paper  UNified Offline Training Paradigm for Learning Path Recommendation.

## Overview

This approach introduces an offline training paradigm in RL-based LPR to provide dense process rewards by a personalized advantage based on a reward model, which can estimate the students' internal knowledge levels on the learning targets.
Additionally, we propose UniLPR model, a personalized recommendation system that unifies students' knowledge levels and the knowledge structures of questions, capturing the implicit relationships and optimizing their representations through both a supervised method and group relative policy optimization.
Finally, we design multiple learning tasks that encompass historical reviewing, recent learning, and long-term exploratory learning to simulate the comprehensive and diverse learning needs of students. 


## Benchmark Datasets and Preprocessing

Our experiments utilize publicly available datasets. [Assist09](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010?authuser=0) is sourced from the ASSISTments platform from 2009 to 2010. [Junyi](https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198) is collected from Junyi Academy in 2012.

Recent Learning: Represent questions to be learned recently, which are more related to local search and strongly correlated with knowledge structures. Following [CSEAL](https://github.com/bigdata-ustc/EduSim) and [GEHRL](https://gitee.com/mindspore/models/tree/master/research/recommend/GEHRL), we retain the first 60% of the student's learning sequence, mask the middle 20% of the learning sequence, and select learning targets from the deduplicated 20% of the sequence. Due to a large number of repeated answers, there are only 1–5 target nodes, so we set the corresponding number of steps to 5, 10, and 20.

Historical Review:  Represent the review of questions previously learned. We randomly select ten from them, which is more in line with backtracking search. The average number of unique questions answered by students is about 10, so we set the corresponding number of steps to 10, 20, and 30.

Exploratory Learning: Represent the global random selection from questions never met before to expand the knowledge boundary, which is more in line with global search. Similarly, we set the step length to 10, 20, and 30.

### Pretrain Environment Model and Reward Model

First, we split the data into training (80%), validation (10%), and test (10%) sets. We pre-trained 2 Knowledge Tracing models, with each model trained for 100 epochs. We employ Average Precision (AP) and the area under the receiver operating characteristic curve (AUC) to evaluate the performance of all methods in predicting binary student performance. The mean results are shown in Table 1:

| Dataset  | ACC    | AUC    |
|----------|--------|--------|
| Junyi    | 0.8217 | 0.8475 |
| Assist09 | 0.7328 | 0.7703 |

*Table 1: Mean performance of DKT on different datasets.*

## Baseline Models

AC: Use a DKT to model students' learning states and vanilla actor-critic as recommender.

PPO: Use a DKT to model students' learning states and vanilla PPO as recommender.

CSEAL: Use a DKT and a cognitive navigation algorithm to narrow the search space, and use vanilla AC to learn and update. 

GEHRL-ST: A hierarchical reinforcement learning architecture that employs sub-tree pruning algorithms. The high-level policy uses DKT to predict learning states, while the low-level policy performs sub-tree pruning to obtain fine-grained path rewards through sub-goals. The high-level policy is optimized using vanilla AC, and the low-level policy is optimized using PPO.

GEHRL-EB: Similarly, use the node vector algorithm for pre-trained graph embeddings and narrow the search space based on distance.

SRC: Employ a concept-aware encoder to jointly optimize knowledge levels and knowledge states. It employs an LSTM-based decoder with deduplication and greedy strategies to output recommendation sequences, and optimizes parameters using policy gradient based on the final reward of the path.

GRU4Rec: A GRU-based sequential recommendation model. 

SASRec: Use a transformer-based sequential recommendation model with unidirectional causal self-attention.

All settings are remained in /config

## Evaluation Tasks
We score 3 runs with the average $E_p$ during training to monitor convergence speed, and validate convergence by $E_p$ of the last 10\% of training episodes in Appendix. We assess and report real-world effectiveness by the average $E_p$ in the testing stage in the following Table /assets/result.jpg .
| Pattern                  | Dataset      | Steps | Stage | AC         | PPO        | CSEAL      | GEHRL-ST   | GEHRL-EB   | SRC         | GRU4Rec     | SASRec     | **UNO**        |
| ------------------------ | ------------ | ----- | ----- | ---------- | ---------- | ---------- | ---------- | ---------- | ----------- | ----------- | ---------- | -------------- |
| **Recent Learning**      | **Junyi**    | 5     | trian | 25.29±0.58 | 25.73±0.30 | 21.35±0.16 | 21.77±0.26 | 23.95±0.49 | 11.58±0.54  | 11.07±4.93  | 27.26±3.96 | **29.12±0.21** |
|                          |              |       | final | 26.54±0.58 | 25.25±0.30 | 20.92±0.16 | 22.25±0.26 | 23.68±0.49 | 8.23±0.54   | 11.86±4.93  | 23.04±3.96 | **31.98±0.21** |
|                          |              |       | eval  | 28.79±0.98 | 29.08±0.47 | 25.58±0.16 | 24.48±0.86 | 25.88±0.98 | 16.04±3.31  | 16.88±2.77  | 26.98±5.15 | **34.78±1.64** |
|                          |              | 10    | trian | 33.38±0.13 | 33.59±0.33 | 26.96±1.09 | 29.30±0.10 | 32.30±0.47 | 11.71±1.62  | 10.38±4.30  | 28.69±2.85 | **39.97±2.61** |
|                          |              |       | final | 32.30±0.13 | 32.70±0.33 | 25.70±1.09 | 26.98±0.10 | 30.61±0.47 | 7.82±1.62   | 9.29±4.30   | 28.54±4.45 | **44.16±2.61** |
|                          |              |       | eval  | 35.84±0.85 | 37.64±1.01 | 30.19±0.61 | 33.03±1.29 | 34.89±1.00 | 17.00±2.64  | 11.67±2.95  | 25.98±3.91 | **46.01±4.08** |
|                          |              | 20    | trian | 42.69±0.13 | 42.94±0.63 | 31.41±0.08 | 37.73±0.47 | 42.08±0.65 | 10.99±1.12  | 19.41±10.15 | 38.65±1.93 | **49.35±1.02** |
|                          |              |       | final | 41.23±0.13 | 41.94±0.63 | 30.18±0.08 | 37.34±0.47 | 40.64±0.65 | 6.44±1.12   | 18.68±10.15 | 34.69±1.93 | **50.98±1.02** |
|                          |              |       | eval  | 45.71±0.81 | 44.88±1.42 | 34.21±1.03 | 39.94±2.38 | 44.16±0.43 | 15.67±0.80  | 20.18±6.77  | 38.20±3.65 | **54.05±1.44** |
|                          | **Assist09** | 5     | trian | 27.64±1.26 | 29.81±0.53 | 26.07±1.39 | 25.02±0.15 | 26.19±0.27 | 36.98±4.30  | 21.12±8.79  | 30.73±0.96 | **38.31±0.37** |
|                          |              |       | final | 25.02±1.26 | 28.57±0.53 | 22.99±1.39 | 26.34±0.15 | 24.31±0.27 | 37.14±4.30  | 14.66±8.79  | 24.50±0.96 | **40.13±0.37** |
|                          |              |       | eval  | 29.65±1.18 | 30.50±2.25 | 25.49±1.74 | 25.52±1.53 | 26.95±0.79 | 38.90±4.12  | 15.61±11.49 | 23.63±9.69 | **41.50±0.21** |
|                          |              | 10    | trian | 30.18±0.98 | 29.12±0.24 | 28.29±0.65 | 29.99±0.71 | 30.86±0.65 | 37.32±3.81  | 27.83±9.13  | 36.60±3.14 | **39.57±0.55** |
|                          |              |       | final | 26.86±0.98 | 29.01±0.24 | 25.97±0.65 | 27.34±0.71 | 30.41±0.65 | 38.06±3.81  | 24.00±9.13  | 37.12±3.14 | **40.40±0.55** |
|                          |              |       | eval  | 32.24±1.40 | 29.81±1.25 | 28.04±2.40 | 30.36±2.39 | 33.99±0.39 | 39.66±3.04  | 16.77±17.60 | 38.74±2.14 | **42.01±0.20** |
|                          |              | 20    | trian | 30.46±0.21 | 31.32±0.61 | 29.75±0.70 | 31.51±0.57 | 31.72±0.30 | 38.78±1.78  | 16.95±7.76  | 26.73±6.16 | **40.48±0.55** |
|                          |              |       | final | 29.46±0.21 | 28.07±0.61 | 24.72±0.70 | 30.19±0.57 | 30.70±0.30 | 40.08±1.78  | 19.85±7.76  | 25.54±6.16 | **40.73±0.55** |
|                          |              |       | eval  | 30.51±2.48 | 30.94±1.71 | 29.50±1.90 | 33.91±2.02 | 32.18±1.64 | 41.82±0.00  | 20.48±13.28 | 30.71±5.85 | **42.04±0.41** |
| **Historical Revise**    | **Junyi**    | 10    | trian | 10.11±0.59 | 9.29±1.07  | 5.72±0.40  | 7.73±0.71  | 9.10±0.71  | -0.51±1.96  | -2.02±4.91  | 9.29±1.99  | **22.78±2.75** |
|                          |              |       | final | 10.69±0.59 | 10.18±1.07 | 7.39±0.40  | 9.23±0.71  | 10.33±0.71 | -1.13±1.96  | 4.64±4.91   | 7.37±1.99  | **23.35±2.75** |
|                          |              |       | eval  | 11.03±1.10 | 10.46±3.16 | 4.17±1.09  | 6.98±0.59  | 7.29±0.93  | -2.58±3.20  | 1.21±2.75   | 10.65±2.88 | **29.19±5.67** |
|                          |              | 20    | trian | 14.76±0.47 | 16.52±0.80 | 7.55±0.54  | 11.19±0.15 | 14.19±0.32 | -1.76±1.04  | 1.50±4.62   | 10.69±3.13 | **27.62±1.92** |
|                          |              |       | final | 10.01±0.47 | 17.55±0.80 | 2.99±0.54  | 8.31±0.15  | 10.14±0.32 | -10.72±1.04 | 2.36±4.62   | 3.77±3.13  | **26.46±1.92** |
|                          |              |       | eval  | 15.55±0.91 | 17.30±3.95 | 10.20±0.80 | 12.37±1.44 | 16.18±0.66 | -7.57±13.35 | 11.77±2.56  | 8.18±4.90  | **34.32±2.48** |
|                          |              | 30    | trian | 19.89±0.41 | 17.55±0.48 | 9.74±1.34  | 16.34±0.66 | 19.06±0.47 | -4.16±2.57  | -4.55±4.59  | 12.20±3.18 | **30.03±1.01** |
|                          |              |       | final | 20.88±0.41 | 18.31±0.48 | 10.55±1.34 | 18.57±0.66 | 18.99±0.47 | -1.57±2.57  | -5.11±4.59  | 7.19±3.18  | **28.00±1.01** |
|                          |              |       | eval  | 19.71±0.94 | 19.37±1.98 | 10.93±2.65 | 16.37±2.52 | 18.84±1.29 | -5.20±7.73  | -4.61±8.29  | 14.63±3.33 | **34.74±3.88** |
|                          | **Assist09** | 10    | trian | 7.39±3.31  | 3.28±1.54  | 1.86±3.79  | 5.59±1.04  | 7.49±0.63  | 28.50±2.86  | -3.87±16.21 | 12.74±6.72 | **32.46±0.81** |
|                          |              |       | final | -1.32±3.31 | -0.60±1.54 | -7.91±3.79 | 2.11±1.04  | 8.45±0.63  | 27.04±2.86  | 6.57±16.21  | 7.98±6.72  | **31.74±0.81** |
|                          |              |       | eval  | 8.51±6.42  | 8.77±1.14  | -1.48±6.96 | 8.27±2.17  | 11.82±2.96 | 30.37±1.94  | 5.10±23.80  | 8.24±8.92  | **35.32±0.59** |
|                          |              | 20    | trian | 9.63±1.72  | 9.46±4.17  | 4.59±2.57  | 5.84±0.69  | 9.60±0.46  | 29.19±2.28  | 10.22±17.96 | 5.94±12.04 | **32.41±0.44** |
|                          |              |       | final | 4.62±1.72  | 11.61±4.17 | 2.41±2.57  | 3.69±0.69  | 11.04±0.46 | 31.46±2.28  | 5.88±17.96  | 4.58±12.04 | **34.14±0.44** |
|                          |              |       | eval  | 10.47±1.52 | 17.96±8.12 | 1.08±2.26  | 11.19±1.35 | 12.12±2.11 | 34.85±0.00  | 8.23±34.68  | 8.56±14.15 | **35.08±0.02** |
|                          |              | 30    | trian | 10.89±2.98 | 10.82±2.43 | -3.13±1.32 | 6.94±0.62  | 10.47±1.05 | 24.08±10.09 | 9.15±16.59  | 4.35±13.88 | **32.84±0.43** |
|                          |              |       | final | 7.73±2.98  | 13.49±2.43 | -3.09±1.32 | 3.57±0.62  | 10.95±1.05 | 23.07±10.09 | 5.53±16.59  | 1.44±13.88 | **31.65±0.43** |
|                          |              |       | eval  | 9.41±3.24  | 17.22±1.18 | -2.25±1.99 | 10.27±2.75 | 11.95±1.84 | 23.22±16.45 | 9.71±25.62  | 1.05±22.69 | **35.15±0.22** |
| **Exploratory Learning** | **Junyi**    | 10    | trian | 0.44±0.32  | 3.96±0.70  | -1.56±0.12 | 0.48±0.19  | 0.63±0.04  | 4.89±0.32   | 3.14±0.90   | 1.50±0.32  | **14.21±1.88** |
|                          |              |       | final | 0.59±0.32  | 7.57±0.70  | -1.19±0.12 | 0.22±0.19  | 1.31±0.04  | 3.31±0.32   | 4.64±0.90   | 1.40±0.32  | **15.06±1.88** |
|                          |              |       | eval  | 0.73±0.62  | 8.96±0.63  | -1.59±0.71 | 0.59±0.71  | -0.08±0.07 | 5.67±1.34   | 3.12±3.68   | 1.49±1.21  | **14.50±2.27** |
|                          |              | 20    | trian | -0.00±0.15 | 8.46±0.92  | -1.68±0.37 | 0.58±0.34  | 0.50±0.11  | 6.90±3.90   | 0.99±1.91   | 1.36±1.14  | **15.09±1.10** |
|                          |              |       | final | 0.21±0.15  | 8.46±0.92  | -0.82±0.37 | -0.19±0.34 | -0.54±0.11 | 5.77±3.92   | 2.89±1.91   | 1.49±1.14  | **16.68±1.10** |
|                          |              |       | eval  | 0.36±0.44  | 7.36±2.68  | -0.80±0.85 | 0.75±0.07  | 0.88±0.23  | 7.85±3.98   | 1.41±1.31   | 0.83±1.62  | **16.17±1.05** |
|                          |              | 30    | trian | 0.71±0.32  | 6.10±0.62  | -1.96±0.20 | 0.28±0.15  | 0.43±0.37  | -8.66±0.68  | 0.53±0.90   | -0.42±0.18 | **16.79±0.60** |
|                          |              |       | final | 1.29±0.32  | 8.07±0.62  | -1.24±0.20 | 0.46±0.15  | 0.47±0.37  | -8.55±0.68  | 4.51±0.90   | 0.73±0.18  | **18.10±0.63** |
|                          |              |       | eval  | 0.20±0.50  | 8.03±2.33  | -1.92±0.41 | 0.60±0.86  | 0.67±0.36  | -6.82±5.34  | 4.28±3.57   | 0.22±0.82  | **16.40±1.63** |
|                          | **Assist09** | 10    | trian | -3.23±1.08 | -2.68±0.68 | -3.10±0.45 | -5.12±0.17 | -4.12±0.29 | 7.37±1.17   | -7.80±9.47  | -1.90±1.71 | **8.92±0.02**  |
|                          |              |       | final | -4.42±1.08 | -2.82±0.68 | -3.82±0.45 | -6.48±0.17 | -5.28±0.29 | 7.62±1.17   | -5.20±9.47  | -0.65±1.71 | **9.49±0.02**  |
|                          |              |       | eval  | -2.10±0.70 | -2.65±1.54 | -3.69±0.63 | -4.91±1.75 | -4.28±1.09 | 7.31±1.08   | -5.31±13.37 | -2.88±1.69 | **7.95±0.38**  |
|                          |              | 20    | trian | -2.72±0.42 | -2.10±0.35 | -3.54±1.02 | -4.97±0.13 | -3.26±0.17 | 7.37±2.24   | -1.68±4.78  | -1.81±2.34 | **8.95±0.11**  |
|                          |              |       | final | -2.30±0.42 | -3.04±0.35 | -4.11±1.02 | -5.40±0.13 | -4.85±0.17 | 6.97±2.24   | -1.43±4.78  | -2.81±2.34 | **9.47±0.11**  |
|                          |              |       | eval  | -1.27±1.19 | -1.89±1.17 | -3.31±0.86 | -4.25±0.49 | -2.66±1.65 | 7.78±0.62   | -0.73±4.17  | -4.58±5.92 | **8.66±0.00**  |
|                          |              | 30    | trian | -3.75±0.70 | -2.07±0.59 | 3.05±3.15  | -5.14±0.27 | -3.74±0.36 | 8.40±0.63   | -4.29±2.82  | -2.63±1.07 | **8.55±0.06**  |
|                          |              |       | final | -3.48±0.70 | 0.90±0.59  | -7.80±3.15 | -4.86±0.27 | -2.32±0.36 | 7.49±0.63   | -0.75±2.82  | -2.29±1.07 | **9.47±0.06**  |
|                          |              |       | eval  | -1.98±0.92 | -1.46±1.01 | -1.48±4.84 | -4.18±0.81 | -2.82±1.08 | 8.22±0.00   | 1.90±5.33   | -2.68±0.36 | **8.64±0.02**  |

## Incorporate New Datasets or New Models

* For new datasets: Users can put the new datasets in ```data/raw_data``` folder, and then run ```preprocess_xxx.py``` to get the processed datasets.

* For new models: Users can put the model implementation in  ```Agent``` folder, and according settings in ```config``` folder, and create the model in ```trainLPR.py``` to run the model.


## Environments
```{bash}
conda install -r requirements.txt
```

#### Model Training
* We can run ```data/preprocess_junyi.py``` for pre-processing the datasets. For example, to preprocess the Junyi dataset, we can run the following commands:
```{bash}
cd data
python preprocess_junyi.py
```

* We can run ```LifelongKT/trainKT.py``` for pre-processing the DKT. For example, to preprocess the *DKT* for *Junyi* dataset, we can run the following commands:
```{bash}
cd LifelongKT
python trainKT.py --config-name DKT --config-path configs/junyi
```

* Example of training *UNO* on *Junyi* dataset:
```{bash}
python trainLPR.py --config-name UNO --config-path config/junyi
```

## Citation
Detailed cititation information is listed in paper.

