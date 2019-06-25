# CIKM AnalytiCup at 2017
This project is the 3rd place solution for [Lazada Product Title Quality Challenge](https://competitions.codalab.org/competitions/16652) associated with CIKM AnalytiCup in 2017. The short description article is at this [link](http://www.cikmconference.org/CIKM2017/download/analytiCup/session3/CIKMAnalytiCup2017_LazadaProductTitleQuality_T3.pdf)

## 1. Task Desription
This chanllenge is to help Lazada to score product tile quality automatically. Here, the titile qulity is composed of **Clarity** and **Conciseness** metrics, more precise definitions of them are listed below:
### Clarity
* 1 means within 5 seconds you can understand the title and what the product is. It easy to read. You can quickly figure out what its key attributes are (color, size, model, etc.).
* 0 means you are not sure what the product is after 5 seconds. it is difficult to read. You will have to read the title more than once.
### Conciseness
* 1 means it is short enough to contain all the necessary information.
* 0 means it is too long with many unnecessary words. Or it is too short and it is unsure what the product is.

## 2. Data Exploration
* The dataset contains title, cat1, cat2, cat3, short descripton, price and country category information. Ecept for price, all of them are in text.Moreover, **Clarity** and **Conciseness** are labelled according to the viewpoint of human being, thus it could be regarded as a text semantic analysis problem.
* We found that a title that is not **Clarity** is certain to be not **Conciseness**.This insight is crucial for our model.

## 3. Methods
### 3.1 Requirements
* NLTK, Spacy, BeautifulSoup, pyenchant, sklearn, xgboost, lightgbm, keras.
### 3.2 Manual Features + Model Stacking
#### Feature Engineering
* Entity detection and spellchecking through spacy and pyenchant respectively.
* Fundamental statistic features, including both char-based and word-based. The number of comma, tag, digit, upper-cased, non-alpha and stop words are calculated.
* Inspired by the metric definitions, we explore the duplicated words, which is the *magic* feature eventually.
* Longest common string with length larger than 2 from pairs of words.
* We find the similar words via wordnet in NLTK.
* The description is parsed from HTML format first, then performs the similar procedure as the tile does.

All the details could be found in *feature_generation.py*.
#### Metric-based Two-layer Stacking
##### Stage1
* To sense the property between the metrics, we perform metric stacking as our 1st stacking stage. For two metrics, cross validation is carried out by xgb, adaboost, rf and lgb models in different seeds, which generates new features correspondingly. 
* At this stage, the features have both categorical and numerical types, so the models are all tree-based.

This is presented in *model_stage1.py*.
##### Stage2
* During this stage, we sample numeric features in order to try more model types. Our 2nd-stage stacking is further based on svm, knn, naive bayes and so on.

Except for xgb and lgb models, parameters are not carefully tuned for other models. Please look at *model_stage2.py*.
### 3.3 Ensemble-LSTM
#### Embeddings
Pretrained *glove* embedding is took advantage of by LSTM further. In addition, we train embedding vectors from scratch for the categories of product.
#### Two-layer LSTM
Word embeddings are fed to a rnn of 2-layer's LSTM to extract features from the whole sequence.
#### Ensemble MLP
First, LSTM features, category embedding features, manual features and metric stacking features are concatenated together, then follows a MLP to ensemble them.

Exact code is in *model_rnn.py* and *model_rnn_clarity.py*.<br>
Finally, *model_stage3.py* is weighted over th best xgb, lgb and lstm models.





