# Kaggle's CommonLit Readability Competition
Linear Regression models to predict the readability score of a given text. The dataset is that from the CommonLit Readability Competition on Kaggle. This competition took place in July, and while I could not submit by the deadline, I still wish to share my work here and showcase my use of the data science methodology.

As someone interensted in NLP, this was a passion project where I got to use text features in building a model, and I really enjoyed the process.

## Code and Resources Used

**Python Version**: 3.6

**Packages**: pandas, numpy, matplotlib, seaborn, nltk, string, re, scipy, sklearn, readability

## Data

As we're predicting the reading ease of texts from literature, the data consists of excerpts from several time periods and a wide range of reading ease scores. The test set includes a slightly larger proportion of modern texts, which is the type of texts we seek to generalize to, than the training set.

Link: https://www.kaggle.com/c/commonlitreadabilityprize/data

Columns (from Kaggle Data Description):
- id - unique ID for excerpt
- url_legal - URL of source - this is blank in the test set.
- license - license of source material - this is blank in the test set.
- excerpt - text to predict reading ease of
- target - reading ease
- standard_error - measure of spread of scores among multiple raters for each excerpt. Not included for test data.

## Initial Analysis

Before any data processing. I checked to see the data **shape (# of rows and columns)**, data **types**, presence of **nulls**, and descriptive statistices of the target variable. There were no nulls and the target variable has a median/50% percentile of -0.912.

Initially, the excerpts have an average character length of about 972 with a standard deviation of 117.24. This metric and the variance shows that the length of the excerpts in oour dataset vary greatly.

![alt text](https://github.com/MarcelinoV/HAI-Infections/blob/master/images/pre_proc_excerpt_stats.jpg "Stats of excerpts Pre_Processing")

A word frequency distribution of the excerpts before NLP processing is also shown to give a better sense of the construction of the excerpts in terms of common words:

![alt text](https://github.com/MarcelinoV/HAI-Infections/blob/master/images/pre_proc_word_freq.jpg "Word Frequency Distribution of excerpts Pre_Processing")

## Data Cleaning



## Feature Engineering

## Model Building

##
