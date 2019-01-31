# TUD-Experiment

Time Unit Discretization model discretizes time-series data by time unit.
Let us call the resulting groups of time-series data 'time unit groups'.
For example, if you choose 'weekday'as a time unit, there would be 7 time unit groups.

Time Unit Clustering (TUC) model and Time Unit Hierarchical (TUH) model are part of TUD model. 
See below for further explanation of each model.

## TUC Model
TUC model is a two step process.
First, it clusters(k-means clustering) 'time unit groups' into 'k' clusters based on the summary statistics of each group. Mean or standard deviation can be used as summary statistics.

Then, bayesian model fitting is applied for each clusters.

Cross Validation is used to choose the best number of clusters 'k'.

For Code, see Tutorial>TUC_Experiment (code update & optimization needed)

For detailed explanation, read the following files in Documents folder:
['TUC Model Brief.pdf' is highly recommended.]
1. 'TUC Model Brief.pdf' explains the backgroud, contribution, flow diagram, example of TUC model application.
2. 'Paper_Time Unit Clustering Model Using Multi-level Regression_View from Time Series Analysis of Pallet Movement Amount.docx' is the original paper for TUC model.
3. 'Patent_SKP18192의 본출원.docx' is the draft for the patent (application expected in December).


## TUH Model
TUH model is a one step process which is based on hierachical Bayesian model. 
Partial pooling concept with hyperparameter is applied. 
