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


## Incorporate New Datasets or New Models

* For new datasets: Users can put the new datasets in data/raw_data folder, and then run preprocess_xxx.py to get the processed datasets.

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

