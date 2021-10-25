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

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/pre_proc_excerpt_stats.JPG "Stats of excerpts Pre_Processing")

A word frequency distribution of the excerpts before NLP processing is also shown to give a better sense of the construction of the excerpts in terms of the 20 most common words:

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/pre_proc_word_freq.JPG "Word Frequency Distribution of excerpts Pre-Processing")

## Data Cleaning

Through NLTK, I performed the following NLP steps to prepare the data for model fitting:

- convert to lowercase
- word tokenization
- punctuation removal
- stopword removal
- stemming and lemmatization

After processing, the most common words are words with context rather than stopwords as in the previous word frequency distribution chart:

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/post_proc_word_freq.JPG "Word Frequency Distribution of excerpts Post-Processing")

## Feature Engineering

My approach to feature engineering was to generate all the information I could from the excerpts, such as number of nouns, verbs, and other grammar terms, and after some research, using the readability package to generate scores for each excerpt using different readability scoring systems and useful sentence information. I accomplished this by using the NLTK POS tagger, or part-of-speech tagger, and the readability package for scorings and sentence info such as syllables, word types, long words, complex words, and similiar details.

Below is a screenshot of the final dataframe after these features were generated from the pre- and post-processed excerpts:

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/snap_of_feat_df.JPG "Screenshot of feature-engineered dataframe")

From these features, I visualized a Spearman correlation heatmap to get a sense of the features that correlate with the target variable the most, as well as where multicollinearity occurs among our features.

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/feat_heatmap.JPG "Spearman Correlation Heatmap of generated features")

For the sake of simplicity, I decided to use features that had a spearman correlation of at least |35| (absolute value) with the target variable, regardless of multicollinearity. Resulting in these final variables for model building:

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/feat_dict.JPG "Dictionary of features above |35| threshold")

The final feature I wished to use was the text itself, in the form of a TF-IDF, or Term Frequency-Inverse Document Frequecy, vector. Using sklearn's Tfidfvectorizer, I created a vectorizer object to fit and transform the processed excerpt data accordingly, and used scipy's horizontal stacking function generate my final training and testing datasets. 

## Model Building

My plan was to use Ridge Regression to model the data, since Ridge Regression is especially good for data that has multicollinearity. After, I would experiment with different models to see how they fared.

My best Ridge Regression model, after optimizing via RepeatedKFold cross validation, was a model with a Root Mean Squared Error of 0.709 and a R2 of 0.535. Other models I tried include Lasso Regression, Random Forest Regressor, and K Neigbors Regressor, which did not achieve results as good as the initial Ridge Regression.

After some more research, I decided to try my hand at a Stacking Model. For those who don't know, Stacking is an ensemble technique that uses predictions from base learners as features for a meta learner. The meta learner model is used for final predictions on the test data.

I defined a stacking model with Bayesian Ridge Regression, Support Vector Machine, and Random Forest Regressor models as base learners and a Ridge Regression model as a meta learner, the whole stacking ensemble having a cross validation of k=5.

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/stacking.JPG "Summary of Stacking Model")

With a cross validation of 5, the average scores of each set of predictions was calculated, with the average root mean squared error resulting in 0.704 and the average R2 (validation) resulting in 0.541, a slightly better result than our baseline Ridge Regression model.

![alt text](https://github.com/MarcelinoV/kaggle_commonlit/blob/main/images/stacking_scores.JPG "Average Scores of Stacking Model")

## Conclusion and Recommendations
