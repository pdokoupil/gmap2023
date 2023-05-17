# About

This repository contains supprelemtary material for a paper *The Effect of Similarity Metric and Group Size on Outlier Selection & Satisfaction in Group Recommender Systems* that was accepted for GMAP@UMAP2023 workshop.

The underlying recommendation algorithms and other parts of the project were cloned from our [second repository](https://github.com/lpeska/GRS_eval_props)

New experiments are in a separate notebooks, see Project structure below.

## Abstract
Group recommender systems (GRS) are a specific case of recommender systems (RS), where recommendations are constructed to
a group of users rather than an individual. GRS has diverse application areas including trip planning, recommending movies to
watch together, or music in shared environments. However, due to the lack of large datasets with group decision-making feedback
information, or even the group definitions, GRS approaches are often evaluated offline w.r.t. individual user feedback and artificially
generated groups. These synthetic groups are usually constructed w.r.t. pre-defined group size and inter-user similarity metric.
While numerous variants of synthetic group generation procedures were utilized so far, its impact on the evaluation results was not
sufficiently discussed. In this paper, we address this research gap by investigating the impact of various synthetic group generation
procedures, namely the usage of different user similarity metrics and the effect of group sizes. We consider them in the context of
“outlier vs. majority” groups, where a group of similar users is extended with one or more diverse ones. Experimental results indicate a
strong impact of the selected similarity metric on both the typical characteristics of selected outliers as well as the performance of
individual GRS algorithms. Moreover, we show that certain algorithms better adapt to larger groups than others.

## Project Structure
- [ml1m_preprocessing.ipynb](./ml1m_preprocessing.ipynb): Jupyter notebook for preprocessing the ML dataset
- [training.ipynb](./training.ipynb): Jupyter notebook used to generate the groups, their recommendations and other long-running computations
- [group_composition_evaluation.ipynb](./groups_composition_evaluation.ipynb): Jupyter notebook generating results related to group composition (e.g. similarities in group etc.)
- [grs_evaluation.ipynb](./grs_evaluation.ipynb): Jupyter notebook used to generate remaining, GRS algorithm-related results
- [results](./results): folder with the results from the paper
- All other files are taken from our [second repository](https://github.com/lpeska/GRS_eval_props) with minor modifications
- [ml-1m](./ml-1m) folder where ML-1M related data should be placed, see our [second repository](https://github.com/lpeska/GRS_eval_props) for details