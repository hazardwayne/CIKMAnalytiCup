### CIKM AnalytiCup 2017 -- Lazada Product Title Quality Challenge
#### the explanation that how to run our source code
#### team name:AVC2
#### participants name: Hong Diao WEN

##### 1.for checking submission files:
locate at CIKM2017_Analyticup folder   
the 10th submission of us is final_results under results folder   
##### 2.for checking  models with generated features:
(1) install python packages.

pandas    
numpy   
scipy(v19.0+)   
sklearn   
keras   

(2) run model_stage2.py under model folder,this would generate following 4 files in data/ensemble folder   

clarity_test_xgb.predict    
clarity_test_lgb.predict    
conciseness_test_xgb.predict    
conciseness_test_xgb.predict    

(3) run model_rnn.py, mlp.py and mlp_clarity.py under model folder respectively, this would generate following files in data/ensemble folder    

conciseness_test_lstm.predict      
conciseness_test_mlp_3211.predict   
clarity_test_mlp_3211.predict   

(4) run model_stage3.py under model folder, this would generate following files in data/ensemble

clarity_test.predict    
conciseness_test.predict    

alternate for (1)->(4) steps:sh ./checking_models.bat   
(5) open data/ensemble,select and select clarity_test.predict and conciseness_test.predict to a blank folder,then zip and summit them to test engine.   

Note: the random seed have been changed so the conciseness results may shift around the submission online.    
##### 3.for checking all:
(1) install python packages.    

pandas    
numpy   
scipy(v19.0+)   
spacy   
enchant   
beatifulsoup(bs4)   
sklearn   
xgboost   
lightgbm    
keras   
nltk    

(2) download models   

nltk:stopwords,wordnet    
spacy:en_core_web_sm    
glove word embedding:glove.840B.300d.txt（not neccesary when embedding.npy file is in file_tmp）  

(3) feature generation
open feature_generation.py, set re_generate = True,then run it, this would
generate  3 files in data/features    

df_feature.csv    
sparse_clarity.npz    
sparse_conciseness.npz    

(4) model_stage1    
open model_stage1.py, set re_generate = True, then run it, this would
generate  1 files in data/features   

df_feature_stage1.csv   

(5) open model_stage2.py,set re_generate = True then run it.the following step is same with 2.for checking  models with generated features:

clarity_prob.csv    
clarity_prob_lgb.csv    
clarity_ada.csv   
clarity_knn_b.csv   
clarity_rf.csv    
clarity_mlp.csv   
...   
conciseness_prob.csv    
conciseness_prob_lgb.csv    
conciseness_ada.csv   
conciseness_knn_b.csv   
conciseness_rf.csv    
conciseness_mlp.csv   
...

(6) run model_stage2.py under model folder,this would generate following 4 files in data/ensemble folder   

clarity_test_xgb.predict    
clarity_test_lgb.predict    
conciseness_test_xgb.predict    
conciseness_test_xgb.predict    

(7) open model_rnn.py,set re_generate = True,then run model_rnn.py, mlp.py and mlp_clarity.py under model folder respectively, this would generate following files in data/ensemble folder    

conciseness_test_lstm.predict      
conciseness_test_mlp_3211.predict   
clarity_test_mlp_3211.predict   

(8) run model_stage3.py under model folder, this would generate following files in data/ensemble

clarity_test.predict    
conciseness_test.predict    

(9) open data/ensemble,select and select clarity_test.predict and conciseness_test.predict to a blank folder,then zip and summit them to test engine.   

alternate for above steps:       
（１）turn all re_generate = True in:   
feature_generation.py   
model_stage1.py   
model_stage2.py  
model_rnn.py  

(2）sh ./checking_all.bat

