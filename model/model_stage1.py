import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

re_generate = False

print("loading features")

feature_dir = '../data/features/'
datadir = '../data/'

ensemble_dir = os.path.join(datadir, 'ensemble')
if not os.path.exists(ensemble_dir):
    os.makedirs(ensemble_dir)

df = pd.read_csv(os.path.join(feature_dir, 'df_feature.csv'))
sparse_tfidf_title_clarity = sparse.load_npz(os.path.join(feature_dir, 'sparse_clarity.npz'))
sparse_tfidf_title_conciseness = sparse.load_npz(os.path.join(feature_dir, 'sparse_conciseness.npz'))

tr_clarity = pd.read_csv(datadir + 'training/clarity_train.labels', header=None)
tr_conciseness = pd.read_csv(datadir + 'training/conciseness_train.labels', header=None)

if re_generate:
    features_clarity = [
                        'country_id'
                        , 'my'
                        , 'ph'
                        , 'sg'
                        , 'title_wordnum'
                        , 'wordnum_q1'
                        , 'wordnum_q2'
                        , 'wordnum_q3'
                        , 'wordnum_q4'
                        , 'wordnum_q5'
                        , 'title_word_duplicate_num'
                        , 'title_word_duplicate_ratio'
                        , 'title_charnum'
                        , 'title_wordcharlargenum'
                        , 'title_nonalphanum'
                        , 'title_uppernum'
                        , 'title_nonengnum'
                        , 'title_wordlemmassnum'
                        , 'title_wordsynsetdepthsum'
                        , 'title_cat3_diff'
                        , 'description_nonalphanum'
                        , 'title_descrition_inter_num'
                        , 'title_descrition_inter_ratio1'
                        , 'title_descrition_inter_ratio2'
                        , 'category_lvl_1_id'
                        , 'category_lvl_2_id'
                        , 'category_lvl_3_id'
                        , 'category_lvl_2_ratio1'
                        , 'category_lvl_3_ratio2'
                        , 'category_lvl_1_ratio3'
                        , 'lda_equal_conciseness'
                        , 'lda_true_prob_conciseness'
                        , 'international'
                        , 'local'
                        , 'title_type_check_num'
                        , 'title_C_upperratio'
                        , 'title_C_upperwordratio'
                        , 'title_eng_word_simi_path_num'
                        ]

    tr_df = df[df['select'] == 1]
    valid_df = df[df['select'] == 0]

    tr_len = tr_df.shape[0]
    val_len = valid_df.shape[0]

    tr_df_clarity = sparse.hstack([tr_df[features_clarity], sparse_tfidf_title_clarity[:tr_len]]).tocsr()
    val_df_clarity = sparse.hstack([valid_df[features_clarity], sparse_tfidf_title_clarity[tr_len:]]).tocsr()

    tr_clarity = np.asarray(tr_clarity).reshape(tr_df.shape[0])

    dtrain_clarity = xgb.DMatrix(data=tr_df_clarity, label=tr_clarity)
    dvalid_clarity = xgb.DMatrix(data=val_df_clarity)

    param_clarity = {'booster': 'gbtree'
                     , 'max_depth': 6
                     , 'eta': 0.05
                     , 'gamma': 1.0
                     , 'silent': 1
                     , 'objective': 'binary:logistic'
                     , 'eval_metric': 'rmse'
                     , 'colsample_bytree': 0.7
                     , 'col_sample_bylevel': 0.5
                     , 'subsample': 0.7
                     }

    num_round = 2000
    # random.seed(321)
    # bst_clarity_cv = xgb.cv(param_clarity
    #                         , dtrain_clarity
    #                         , num_round
    #                         , nfold=5
    #                         , metrics=['rmse']
    #                         , verbose_eval=True
    #                         , early_stopping_rounds=20
    #                         )

    param_clarity_2 = {'booster': 'gbtree'
                       , 'max_depth': 8
                       , 'eta': 0.1
                       , 'gamma': 1.0
                       , 'silent': 1
                       , 'objective': 'binary:logistic'
                       , 'eval_metric': 'rmse'
                       , 'colsample_bytree': 0.9
                       , 'col_sample_bylevel': 0.9
                       , 'subsample': 0.9
                       }

    # bst_clarity_cv_2 = xgb.cv(param_clarity_2
    #                           , dtrain_clarity
    #                           , num_round
    #                           , nfold=5
    #                           , metrics=['rmse']
    #                           , verbose_eval=True
    #                           , early_stopping_rounds=20
    #                           )

    dtrain_clarity_lgb = lgb.Dataset(tr_df_clarity, tr_clarity)
    dvalid_clarity_lgb = lgb.Dataset(val_df_clarity)

    param_clarity_lgb = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': {'mse'},
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'max_depth': 9,
        'num_leaves': 95,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4
    }

    # gbm_clarity_cv = lgb.cv(param_clarity_lgb,
    #                         dtrain_clarity_lgb,
    #                         num_boost_round=num_round,
    #                         verbose_eval=5,
    #                         early_stopping_rounds=30
    #                         )

    rf = RandomForestClassifier(n_estimators=200, n_jobs=4)

    param_clarity_dart = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': {'mse'},
        'lambda_l1': 1,
        'lambda_l2': 1,
        'max_depth': 10,
        'num_leaves': 127,
        'learning_rate': 0.05,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 7,
        'verbose': 0,
        'num_threads': 4
    }

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),
                             algorithm="SAMME.R",
                             learning_rate=0.07,
                             n_estimators=80,
                             )

    clarity_tr_prob = np.zeros(tr_len)
    clarity_val_prob = np.zeros(val_len)
    # # 2nd xgb
    clarity_tr_prob_2 = np.zeros(tr_len)
    clarity_val_prob_2 = np.zeros(val_len)

    clarity_tr_prob_lgb = np.zeros(tr_len)
    clarity_val_prob_lgb = np.zeros(val_len)
    clarity_tr_prob_rf = np.zeros(tr_len)
    clarity_val_prob_rf = np.zeros(val_len)
    clarity_tr_prob_dart = np.zeros(tr_len)
    clarity_val_prob_dart = np.zeros(val_len)
    clarity_tr_prob_ada = np.zeros(tr_len)
    clarity_val_prob_ada = np.zeros(val_len)

    print("clarity ensemble")

    print ("ensemble xgb")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=111)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_clarity = xgb.DMatrix(X_test, label=y_test)

        bst_clarity_train = xgb.train(param_clarity
                                      , dtrain_clarity
                                      , 125
                                      )

        clarity_tr_prob[test_index] = bst_clarity_train.predict(dtrain_other_clarity)
        clarity_val_prob += bst_clarity_train.predict(dvalid_clarity)/5.0

    print ("ensemble xgb2")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=112)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_clarity = xgb.DMatrix(X_test, label=y_test)

        bst_clarity_train_2 = xgb.train(param_clarity_2
                                        , dtrain_clarity
                                        , 65
                                        )

        clarity_tr_prob_2[test_index] = bst_clarity_train_2.predict(dtrain_other_clarity)
        clarity_val_prob_2 += bst_clarity_train_2.predict(dvalid_clarity) / 5.0

    print ("ensemble lgb")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=113)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity_lgb = lgb.Dataset(X_train, label=y_train)

        gbm_clarity_train = lgb.train(param_clarity_lgb
                                      , dtrain_clarity_lgb
                                      , num_boost_round=145
                                      )

        clarity_tr_prob_lgb[test_index] = gbm_clarity_train.predict(X_test)
        clarity_val_prob_lgb += gbm_clarity_train.predict(val_df_clarity)/5.0

    print ("ensemble rf")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=114)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        rf.fit(X_train, y_train)

        clarity_tr_prob_rf[test_index] = rf.predict_proba(X_test)[:, 1]
        clarity_val_prob_rf += rf.predict_proba(val_df_clarity)[:, 1]/5.0

    print ("ensemble dart")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=115)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity_lgb = lgb.Dataset(X_train, label=y_train)

        dart_clarity_train = lgb.train(param_clarity_dart
                                       , dtrain_clarity_lgb
                                       , num_boost_round=150
                                       )

        clarity_tr_prob_dart[test_index] = dart_clarity_train.predict(X_test)
        clarity_val_prob_dart += dart_clarity_train.predict(val_df_clarity)/5.0

    print ("ensemble ada")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=116)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        ada.fit(X_train, y_train)

        clarity_tr_prob_ada[test_index] = ada.predict_proba(X_test)[:, 1]
        clarity_val_prob_ada += ada.predict_proba(val_df_clarity)[:, 1]/5.0

    df['clarity_prob'] = np.hstack([clarity_tr_prob, clarity_val_prob])
    df['clarity_prob_2'] = np.hstack([clarity_tr_prob_2, clarity_val_prob_2])
    df['clarity_prob_lgb'] = np.hstack([clarity_tr_prob_lgb, clarity_val_prob_lgb])
    df['clarity_prob_rf'] = np.hstack([clarity_tr_prob_rf, clarity_val_prob_rf])
    df['clarity_prob_dart'] = np.hstack([clarity_tr_prob_dart, clarity_val_prob_dart])
    df['clarity_prob_ada'] = np.hstack([clarity_tr_prob_ada, clarity_val_prob_ada])

    df['clarity_prob'].to_csv(os.path.join(ensemble_dir, 'clarity_prob.csv'), index=None)
    df['clarity_prob_2'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_2.csv'), index=None)
    df['clarity_prob_lgb'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_lgb.csv'), index=None)
    df['clarity_prob_rf'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_rf.csv'), index=None)
    df['clarity_prob_dart'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_dart.csv'), index=None)
    df['clarity_prob_ada'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_ada.csv'), index=None)

    # # corse-to-fine feature
    df = pd.concat([df, pd.get_dummies(pd.qcut(df['clarity_prob'], 10,
                    labels=["clarity_prob_q1", "clarity_prob_q2", "clarity_prob_q3", 'clarity_prob_q4', 'clarity_prob_q5',
                    "clarity_prob_q6", "clarity_prob_q7", "clarity_prob_q8", 'clarity_prob_q9', 'clarity_prob_q10']))], axis=1)
    df = pd.concat([df, pd.get_dummies(pd.qcut(df['clarity_prob_2'], 5,
                                               labels=["clarity_prob_2_q1", "clarity_prob_2_q2", "clarity_prob_2_q3",
                                                       'clarity_prob_2_q4', 'clarity_prob_2_q5']))], axis=1)

    features_conciseness = ['country_id'
                            , 'my'
                            , 'ph'
                            , 'sg'
                            , 'sku_id'
                            , 'title_wordnum'
                            , 'title_cat3_num'
                            , 'title_cat3_include_num'
                            , 'title_cat2_diff'
                            , 'title_cat3_diff'
                            , 'wordnum_q1'
                            , 'wordnum_q2'
                            , 'wordnum_q3'
                            , 'wordnum_q4'
                            , 'wordnum_q5'
                            , 'title_ent_cat_list_num_max'
                            , 'title_eng_word_simi_num'
                            , 'title_eng_word_simi_path_num'
                            , 'title_word_duplicate_num'
                            , 'title_word_duplicate_nums'
                            , 'title_word_duplicate_cat_num'
                            , 'title_word_duplicate_ratio'
                            , 'title_charnum'
                            , 'title_wordcharlargenum'
                            , 'title_avgwordlen'
                            , 'title_nonalphanum'
                            , 'title_uppernum'
                            , 'title_stopsnum'
                            , 'title_nonengnum'
                            , 'title_meaningword_ratio'
                            , 'title_wordlemmassnum'
                            , 'title_wordsynsetdepthsum'
                            , 'description_content_word_num'
                            , 'title_descrition_inter_num'
                            , 'category_lvl_1_id'
                            , 'category_lvl_2_id'
                            , 'category_lvl_3_id'
                            , 'category_lvl_1_frequency'
                            , 'category_lvl_2_frequency'
                            , 'category_lvl_3_frequency'
                            , 'category_lvl_2_ratio1'
                            , 'category_lvl_3_ratio2'
                            , 'category_lvl_1_ratio3'
                            , 'title_cat2_duplicate_ratio_avg_contrast'
                            , 'title_cat_wordnum_mean_contrast'
                            , 'title_cat2_wordnum_mean_contrast'
                            , 'title_cat3_wordnum_mean_contrast'
                            , 'title_cat_wordlemmasssum_mean_contrast'
                            , 'title_cat2_wordlemmasssum_mean_contrast'
                            , 'title_cat3_wordlemmasssum_mean_contrast'
                            , 'price'
                            , 'price_country_mean_contrast'
                            , 'price_cat_mean_contrast'
                            , 'price_cat2_mean_contrast'
                            , 'price_cat3_mean_contrast'
                            , 'product_type_id'
                            , 'NA'
                            , 'international'
                            , 'local'
                            , 'title_C_upperratio'
                            , 'title_C_upperwordratio'
                            , 'title_D_ratio'
                            , 'title_D_wordratio'
                            , 'title_word_lcs_num'
                            , 'title_word_lcs_cat_num'
                            , 'title_digitnum'
                            , 'title_word_duplicate_num2'
                            , 'tittle_upper_word'
                            , 'tittle_small_upper_word'
                            , 'title_word_duplicate_num_cleaned'
                            , 'title_word_duplicate_cat_num_cleaned'
                            , 'description_C_upperratio'
                            , 'description_C_upperwordratio'
                            , 'description_nonalphanum'
                            , 'description_li_num'
                            ]

    tr_df_conciseness = sparse.hstack([tr_df[features_conciseness], sparse_tfidf_title_conciseness[:tr_df.shape[0]]]).tocsr()
    val_df_conciseness = sparse.hstack([valid_df[features_conciseness], sparse_tfidf_title_conciseness[tr_df.shape[0]:]]).tocsr()

    tr_conciseness = np.asarray(tr_conciseness).reshape(tr_df.shape[0])

    dtrain_conciseness = xgb.DMatrix(data=tr_df_conciseness, label=tr_conciseness)
    dvalid_conciseness = xgb.DMatrix(data=val_df_conciseness)

    param_conciseness = {'booster': 'gbtree'
                         , 'max_depth': 6
                         , 'eta': 0.05
                         , 'gamma': 1.0
                         , 'silent': 1
                         , 'objective': 'binary:logistic'
                         , 'eval_metric': 'rmse'
                         , 'colsample_bytree': 0.7
                         , 'col_sample_bylevel': 0.5
                         , 'subsample': 0.7
                         }

    # random.seed(322)
    # bst_conciseness_cv = xgb.cv(param_conciseness
    #                             , dtrain_conciseness
    #                             , num_round
    #                             , nfold=5
    #                             , metrics=['rmse']
    #                             , verbose_eval=20
    #                             , early_stopping_rounds=50
    #                             )

    param_conciseness_2 = {'booster': 'gbtree'
                           , 'max_depth': 9
                           , 'eta': 0.04
                           , 'gamma': 1.0
                           , 'silent': 1
                           , 'objective': 'binary:logistic'
                           , 'eval_metric': 'rmse'
                           , 'colsample_bytree': 0.8
                           , 'col_sample_bylevel': 0.6
                           , 'subsample': 0.9
                           }

    # bst_conciseness_cv_2 = xgb.cv(param_conciseness_2
    #                               , dtrain_conciseness
    #                               , num_round
    #                               , nfold=5
    #                               , metrics=['rmse']
    #                               , verbose_eval=20
    #                               , early_stopping_rounds=50
    #                              )

    dtrain_conciseness_lgb = lgb.Dataset(data=tr_df_conciseness, label=tr_conciseness)
    dvalid_conciseness_lgb = lgb.Dataset(data=val_df_conciseness)

    param_conciseness_lgb = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression_l2',
        'metric': {'mse'},
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'max_depth': 8,
        'num_leaves': 95,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4
    }

    # gbm_conciseness_cv = lgb.cv(param_conciseness_lgb,
    #                             dtrain_conciseness_lgb,
    #                             num_boost_round=num_round,
    #                             verbose_eval=5,
    #                             early_stopping_rounds=30
    #                            )

    print("conciseness ensemble")

    rf = RandomForestClassifier(n_estimators=300, n_jobs=4)

    param_conciseness_dart = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': {'mse'},
        'lambda_l1': 1,
        'lambda_l2': 1,
        'max_depth': 9,
        'num_leaves': 95,
        'learning_rate': 0.05,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.65,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4
    }

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),
                             algorithm="SAMME.R",
                             learning_rate=0.07,
                             n_estimators=100)

    conciseness_tr_prob = np.zeros(tr_len)
    conciseness_val_prob = np.zeros(val_len)
    conciseness_tr_prob_2 = np.zeros(tr_len)
    conciseness_val_prob_2 = np.zeros(val_len)
    conciseness_tr_prob_lgb = np.zeros(tr_len)
    conciseness_val_prob_lgb = np.zeros(val_len)
    conciseness_tr_prob_rf = np.zeros(tr_len)
    conciseness_val_prob_rf = np.zeros(val_len)
    conciseness_tr_prob_dart = np.zeros(tr_len)
    conciseness_val_prob_dart = np.zeros(val_len)
    conciseness_tr_prob_ada = np.zeros(tr_len)
    conciseness_val_prob_ada = np.zeros(val_len)

    print "ensemble xgb"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=121)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_conciseness = xgb.DMatrix(X_test, label=y_test)

        bst_conciseness_train = xgb.train(param_conciseness
                                          , dtrain_conciseness
                                          , 660
                                          )

        conciseness_tr_prob[test_index] = bst_conciseness_train.predict(dtrain_other_conciseness)
        conciseness_val_prob += bst_conciseness_train.predict(dvalid_conciseness) / 5.0

    print "ensemble xgb2"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=122)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_conciseness = xgb.DMatrix(X_test, label=y_test)

        bst_conciseness_train_2 = xgb.train(param_conciseness_2
                                            , dtrain_conciseness
                                            , 410
                                            )

        conciseness_tr_prob_2[test_index] = bst_conciseness_train_2.predict(dtrain_other_conciseness)
        conciseness_val_prob_2 += bst_conciseness_train_2.predict(dvalid_conciseness) / 5.0

    print "ensemble lgb"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness_lgb = lgb.Dataset(X_train, label=y_train)
        dtrain_other_conciseness_lgb = lgb.Dataset(X_test, label=y_test)

        gbm_conciseness_train = lgb.train(param_conciseness_lgb
                                          , dtrain_conciseness_lgb
                                          , num_boost_round=405
                                          )

        conciseness_tr_prob_lgb[test_index] = gbm_conciseness_train.predict(X_test)
        conciseness_val_prob_lgb += gbm_conciseness_train.predict(val_df_conciseness) / 5.0

    print "ensemble rf"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=124)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        rf.fit(X_train, y_train)
        conciseness_tr_prob_rf[test_index] = rf.predict_proba(X_test)[:, 1]
        conciseness_val_prob_rf += rf.predict_proba(val_df_conciseness)[:, 1] / 5.0

    print "ensemble dart"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=125)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness_lgb = lgb.Dataset(X_train, label=y_train)

        dart_conciseness_train = lgb.train(param_conciseness_dart
                                           , dtrain_conciseness_lgb
                                           , num_boost_round=150
                                           )

        conciseness_tr_prob_dart[test_index] = dart_conciseness_train.predict(X_test)
        conciseness_val_prob_dart += dart_conciseness_train.predict(val_df_conciseness) / 5.0

    print "ensemble ada"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=126)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        ada.fit(X_train, y_train)

        conciseness_tr_prob_ada[test_index] = ada.predict_proba(X_test)[:, 1]
        conciseness_val_prob_ada += ada.predict_proba(val_df_conciseness)[:, 1] / 5.0

    df['conciseness_prob'] = np.hstack([conciseness_tr_prob, conciseness_val_prob])
    df['conciseness_prob_2'] = np.hstack([conciseness_tr_prob_2, conciseness_val_prob_2])
    df['conciseness_prob_lgb'] = np.hstack([conciseness_tr_prob_lgb, conciseness_val_prob_lgb])
    df['conciseness_prob_rf'] = np.hstack([conciseness_tr_prob_rf, conciseness_val_prob_rf])
    df['conciseness_prob_dart'] = np.hstack([conciseness_tr_prob_dart, conciseness_val_prob_dart])
    df['conciseness_prob_ada'] = np.hstack([conciseness_tr_prob_ada, conciseness_val_prob_ada])

    df['conciseness_prob'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob.csv'), index=None)
    df['conciseness_prob_2'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_2.csv'), index=None)
    df['conciseness_prob_lgb'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_lgb.csv'), index=None)
    df['conciseness_prob_rf'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_rf.csv'), index=None)
    df['conciseness_prob_dart'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_dart.csv'), index=None)
    df['conciseness_prob_ada'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_ada.csv'), index=None)

    df = pd.concat([df, pd.get_dummies(pd.qcut(df['conciseness_prob'], 10,
                    labels=["conciseness_prob_q1", "conciseness_prob_q2", "conciseness_prob_q3", 'conciseness_prob_q4', 'conciseness_prob_q5',
                    "conciseness_prob_q6", "conciseness_prob_q7", "conciseness_prob_q8", 'conciseness_prob_q9', 'conciseness_prob_q10']))], axis=1)

    df = pd.concat([df, pd.get_dummies(pd.qcut(df['conciseness_prob_2'], 5,
                                               labels=["conciseness_prob_2_q1", "conciseness_prob_2_q2",
                                                       "conciseness_prob_2_q3", 'conciseness_prob_2_q4',
                                                       'conciseness_prob_2_q5']))], axis=1)

    print("clarity with conciseness feature")
    features_clarity = [
                        'country_id'
                        , 'my'
                        , 'ph'
                        , 'sg'
                        , 'title_wordnum'
                        , 'wordnum_q1'
                        , 'wordnum_q2'
                        , 'wordnum_q3'
                        , 'wordnum_q4'
                        , 'wordnum_q5'
                        , 'title_word_duplicate_num'
                        , 'title_word_duplicate_ratio'
                        , 'title_charnum'
                        , 'title_wordcharlargenum'
                        , 'title_nonalphanum'
                        , 'title_uppernum'
                        , 'title_nonengnum'
                        , 'title_wordlemmassnum'
                        , 'title_wordsynsetdepthsum'
                        , 'title_cat3_diff'
                        , 'description_nonalphanum'
                        , 'title_descrition_inter_num'
                        , 'title_descrition_inter_ratio1'
                        , 'title_descrition_inter_ratio2'
                        , 'category_lvl_1_id'
                        , 'category_lvl_2_id'
                        , 'category_lvl_3_id'
                        , 'category_lvl_2_ratio1'
                        , 'category_lvl_3_ratio2'
                        , 'category_lvl_1_ratio3'
                        , 'lda_equal_conciseness'
                        , 'lda_true_prob_conciseness'
                        , 'NA'
                        , 'international'
                        , 'local'
                        , 'conciseness_prob'
                        , 'conciseness_prob_2'
                        , 'conciseness_prob_lgb'
                        , 'conciseness_prob_rf'
                        , 'conciseness_prob_dart'
                        , 'conciseness_prob_ada'
                        , 'title_type_check_num'
                        , 'title_C_upperratio'
                        , 'title_C_upperwordratio'
                        , 'description_li_num'
                        , "conciseness_prob_q1"
                        , "conciseness_prob_q2"
                        , "conciseness_prob_q3"
                        , 'conciseness_prob_q4'
                        , 'conciseness_prob_q5'
                        , "conciseness_prob_q6"
                        , "conciseness_prob_q7"
                        , "conciseness_prob_q8"
                        , 'conciseness_prob_q9'
                        , 'conciseness_prob_q10'
                        , "conciseness_prob_2_q1"
                        , "conciseness_prob_2_q2"
                        , "conciseness_prob_2_q3"
                        , 'conciseness_prob_2_q4'
                        , 'conciseness_prob_2_q5'
                        ]

    tr_df = df[df['select'] == 1]
    valid_df = df[df['select'] == 0]

    tr_df_clarity = sparse.hstack([tr_df[features_clarity], sparse_tfidf_title_clarity[:tr_df.shape[0]]]).tocsr()
    val_df_clarity = sparse.hstack([valid_df[features_clarity], sparse_tfidf_title_clarity[tr_df.shape[0]:]]).tocsr()

    dtrain_clarity = xgb.DMatrix(data=tr_df_clarity, label=tr_clarity)
    dvalid_clarity = xgb.DMatrix(data=val_df_clarity)

    param_clarity = {'booster': 'gbtree'
                     , 'max_depth': 6
                     , 'eta': 0.03
                     , 'gamma': 1
                     , 'silent': 1
                     , 'objective': 'binary:logistic'
                     , 'eval_metric': 'rmse'
                     , 'colsample_bytree': 0.7
                     , 'col_sample_bylevel': 0.6
                     , 'subsample': 0.7
                     }

    random.seed(323)
    # bst_clarity_cv = xgb.cv(param_clarity
    #                         , dtrain_clarity
    #                         , num_round
    #                         , nfold=5
    #                         , metrics=['rmse']
    #                         , verbose_eval=True
    #                         , early_stopping_rounds=30
    #                         )

    param_clarity_2 = {'booster': 'gbtree'
                       , 'max_depth': 8
                       , 'eta': 0.05
                       , 'gamma': 1
                       , 'silent': 1
                       , 'objective': 'binary:logistic'
                       , 'eval_metric': 'rmse'
                       , 'colsample_bytree': 0.8
                       , 'col_sample_bylevel': 0.75
                       , 'subsample': 0.8
                       }

    # bst_clarity_cv_2 = xgb.cv(param_clarity_2
    #                           , dtrain_clarity
    #                           , num_round
    #                           , nfold=5
    #                           , metrics=['rmse']
    #                           , verbose_eval=True
    #                           , early_stopping_rounds=30
    #                           )

    dtrain_clarity_lgb = lgb.Dataset(data=tr_df_clarity, label=tr_clarity)
    dvalid_clarity_lgb = lgb.Dataset(data=val_df_clarity)

    param_clarity_lgb = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'mse'},
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'max_depth': 9,
        'num_leaves': 95,
        'learning_rate': 0.03,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 6,
        'verbose': 0,
        'num_threads': 4
    }

    # gbm_clarity_cv = lgb.cv(param_clarity_lgb,
    #                         dtrain_clarity_lgb,
    #                         num_boost_round=num_round,
    #                         verbose_eval=5,
    #                         early_stopping_rounds=30
    #                         )

    print('conciseness with clarity feature')

    features_conciseness = ['country_id'
                            , 'my'
                            , 'ph'
                            , 'sg'
                            , 'sku_id'
                            , 'title_wordnum'
                            , 'title_cat3_num'
                            , 'title_cat3_include_num'
                            , 'title_cat2_diff'
                            , 'title_cat3_diff'
                            , 'wordnum_q1'
                            , 'wordnum_q2'
                            , 'wordnum_q3'
                            , 'wordnum_q4'
                            , 'wordnum_q5'
                            , 'title_ent_cat_list_num_max'
                            , 'title_eng_word_simi_num'
                            , 'title_eng_word_simi_path_num'
                            , 'title_word_duplicate_num'
                            , 'title_word_duplicate_nums'
                            , 'title_word_duplicate_cat_num'
                            , 'title_word_duplicate_ratio'
                            , 'title_charnum'
                            , 'title_wordcharlargenum'
                            , 'title_avgwordlen'
                            , 'title_nonalphanum'
                            , 'title_uppernum'
                            , 'title_stopsnum'
                            , 'title_nonengnum'
                            , 'title_meaningword_ratio'
                            , 'title_wordlemmassnum'
                            , 'title_wordsynsetdepthsum'
                            , 'description_content_word_num'
                            , 'title_descrition_inter_num'
                            , 'category_lvl_1_id'
                            , 'category_lvl_2_id'
                            , 'category_lvl_3_id'
                            , 'category_lvl_1_frequency'
                            , 'category_lvl_2_frequency'
                            , 'category_lvl_3_frequency'
                            , 'category_lvl_2_ratio1'
                            , 'category_lvl_3_ratio2'
                            , 'category_lvl_1_ratio3'
                            , 'title_cat2_duplicate_ratio_avg_contrast'
                            , 'title_cat_wordnum_mean_contrast'
                            , 'title_cat2_wordnum_mean_contrast'
                            , 'title_cat3_wordnum_mean_contrast'
                            , 'title_cat_wordlemmasssum_mean_contrast'
                            , 'title_cat2_wordlemmasssum_mean_contrast'
                            , 'title_cat3_wordlemmasssum_mean_contrast'
                            , 'price'
                            , 'price_country_mean_contrast'
                            , 'price_cat_mean_contrast'
                            , 'price_cat2_mean_contrast'
                            , 'price_cat3_mean_contrast'
                            , 'product_type_id'
                            , 'NA'
                            , 'international'
                            , 'local'
                            , 'title_C_upperratio'
                            , 'title_C_upperwordratio'
                            , 'title_D_ratio'
                            , 'title_D_wordratio'
                            , 'title_word_lcs_num'
                            , 'title_word_lcs_cat_num'
                            , 'title_digitnum'
                            , 'title_word_duplicate_num2'
                            , 'tittle_upper_word'
                            , 'tittle_small_upper_word'
                            , 'title_word_duplicate_num_cleaned'
                            , 'title_word_duplicate_cat_num_cleaned'
                            , 'description_C_upperratio'
                            , 'description_C_upperwordratio'
                            , 'description_nonalphanum'
                            , 'description_li_num'
                            , 'clarity_prob'
                            , 'clarity_prob_2'
                            , 'clarity_prob_lgb'
                            , 'clarity_prob_rf'
                            , 'clarity_prob_dart'
                            , 'clarity_prob_ada'
                            , "clarity_prob_q1"
                            , "clarity_prob_q2"
                            , "clarity_prob_q3"
                            , 'clarity_prob_q4'
                            , 'clarity_prob_q5'
                            , "clarity_prob_q6"
                            , "clarity_prob_q7"
                            , "clarity_prob_q8"
                            , 'clarity_prob_q9'
                            , 'clarity_prob_q10'
                            , "clarity_prob_2_q1"
                            , "clarity_prob_2_q2"
                            , "clarity_prob_2_q3"
                            , 'clarity_prob_2_q4'
                            , 'clarity_prob_2_q5'
                            ]

    tr_df = df[df['select'] == 1]
    valid_df = df[df['select'] == 0]
    tr_df_conciseness = sparse.hstack([tr_df[features_conciseness], sparse_tfidf_title_conciseness[:tr_df.shape[0]]]).tocsr()
    val_df_conciseness = sparse.hstack([valid_df[features_conciseness], sparse_tfidf_title_conciseness[tr_df.shape[0]:]]).tocsr()

    dtrain_conciseness = xgb.DMatrix(data=tr_df_conciseness, label=tr_conciseness)
    dvalid_conciseness = xgb.DMatrix(data=val_df_conciseness)

    param_conciseness = {'booster': 'gbtree'
                         , 'max_depth': 6
                         , 'eta': 0.02
                         , 'gamma': 1.0
                         , 'silent': 1
                         , 'objective': 'binary:logistic'
                         , 'eval_metric': 'rmse'
                         , 'colsample_bytree': 0.7
                         , 'col_sample_bylevel': 0.6
                         , 'subsample': 0.7
                         }

    random.seed(333)

    # bst_conciseness_cv = xgb.cv(param_conciseness
    #                             , dtrain_conciseness
    #                             , num_round
    #                             , nfold=5
    #                             , metrics=['rmse']
    #                             , verbose_eval=20
    #                             , early_stopping_rounds=30
    #                             )

    param_conciseness_2 = {'booster': 'gbtree'
                           , 'max_depth': 8
                           , 'eta': 0.03
                           , 'gamma': 1.0
                           , 'silent': 1
                           , 'objective': 'binary:logistic'
                           , 'eval_metric': 'rmse'
                           , 'colsample_bytree': 0.8
                           , 'col_sample_bylevel': 0.7
                           , 'subsample': 0.75
                          }

    random.seed(333)

    # bst_conciseness_cv_2 = xgb.cv(param_conciseness_2
    #                               , dtrain_conciseness
    #                               , num_round
    #                               , nfold=5
    #                               , metrics=['rmse']
    #                               , verbose_eval=20
    #                               , early_stopping_rounds=30
    #                               )

    dtrain_conciseness_lgb = lgb.Dataset(data=tr_df_conciseness, label=tr_conciseness)
    dvalid_conciseness_lgb = lgb.Dataset(data=val_df_conciseness)

    param_conciseness_lgb = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'mse'},
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'max_depth': 9,
        'num_leaves': 95,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.6,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4
    }

    # gbm_conciseness_cv = lgb.cv(param_conciseness_lgb,
    #                             dtrain_conciseness_lgb,
    #                             num_boost_round=num_round,
    #                             verbose_eval=5,
    #                             early_stopping_rounds=30
    #                            )

    print('clarity ensemble with conciseness feature')

    rf = RandomForestClassifier(n_estimators=200, n_jobs=4)

    param_clarity_dart = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': 'rmse',
        'num_leaves': 79,
        'max_depth': 9,
        'learning_rate': 0.1,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 6,
        'verbose': 0
    }

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=8),
                             algorithm="SAMME.R",
                             learning_rate=0.07,
                             n_estimators=80)

    clarity_tr_prob = np.zeros(tr_len)
    clarity_val_prob = np.zeros(val_len)
    clarity_tr_prob_2 = np.zeros(tr_len)
    clarity_val_prob_2 = np.zeros(val_len)
    clarity_tr_prob_lgb = np.zeros(tr_len)
    clarity_val_prob_lgb = np.zeros(val_len)
    clarity_tr_prob_rf = np.zeros(tr_len)
    clarity_val_prob_rf = np.zeros(val_len)
    clarity_tr_prob_dart = np.zeros(tr_len)
    clarity_val_prob_dart = np.zeros(val_len)
    clarity_tr_prob_ada = np.zeros(tr_len)
    clarity_val_prob_ada = np.zeros(val_len)

    print ("ensemble xgb")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=131)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_clarity = xgb.DMatrix(X_test, label=y_test)

        bst_clarity_train = xgb.train(param_clarity
                                      , dtrain_clarity
                                      , 237
                                      )

        clarity_tr_prob[test_index] = bst_clarity_train.predict(dtrain_other_clarity)
        clarity_val_prob += bst_clarity_train.predict(dvalid_clarity)/5.0

    print ("ensemble xgb2")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=132)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_clarity = xgb.DMatrix(X_test, label=y_test)

        bst_clarity_train_2 = xgb.train(param_clarity_2
                                        , dtrain_clarity
                                        , 105
                                        )

        clarity_tr_prob_2[test_index] = bst_clarity_train_2.predict(dtrain_other_clarity)
        clarity_val_prob_2 += bst_clarity_train_2.predict(dvalid_clarity) / 5.0

    print ("ensemble lgb")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=133)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity_lgb = lgb.Dataset(X_train, label=y_train)

        gbm_clarity_train = lgb.train(param_clarity_lgb
                                      , dtrain_clarity_lgb
                                      , num_boost_round=200
                                      )

        clarity_tr_prob_lgb[test_index] = gbm_clarity_train.predict(X_test)
        clarity_val_prob_lgb += gbm_clarity_train.predict(val_df_clarity)/5.0

    print ("ensemble rf")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=134)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        rf.fit(X_train, y_train)

        clarity_tr_prob_rf[test_index] = rf.predict_proba(X_test)[:, 1]
        clarity_val_prob_rf += rf.predict_proba(val_df_clarity)[:, 1]/5.0

    print ("ensemble dart")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=135)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity_lgb = lgb.Dataset(X_train, label=y_train)

        dart_clarity_train = lgb.train(param_clarity_dart
                                       , dtrain_clarity_lgb
                                       , num_boost_round=150
                                       )

        clarity_tr_prob_dart[test_index] = dart_clarity_train.predict(X_test)
        clarity_val_prob_dart += dart_clarity_train.predict(val_df_clarity)/5.0

    print ("ensemble ada")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=136)
    for train_index, test_index in skf.split(tr_df_clarity, tr_clarity):
        X_train, X_test = tr_df_clarity[train_index], tr_df_clarity[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        ada.fit(X_train, y_train)

        clarity_tr_prob_ada[test_index] = ada.predict_proba(X_test)[:, 1]
        clarity_val_prob_ada += ada.predict_proba(val_df_clarity)[:, 1]/5.0

    # # call them as boosted feature
    df['clarity_prob_b'] = np.hstack([clarity_tr_prob, clarity_val_prob])
    df['clarity_prob_b_2'] = np.hstack([clarity_tr_prob_2, clarity_val_prob_2])
    df['clarity_prob_lgb_b'] = np.hstack([clarity_tr_prob_lgb, clarity_val_prob_lgb])
    df['clarity_prob_rf_b'] = np.hstack([clarity_tr_prob_rf, clarity_val_prob_rf])
    df['clarity_prob_dart_b'] = np.hstack([clarity_tr_prob_dart, clarity_val_prob_dart])
    df['clarity_prob_ada_b'] = np.hstack([clarity_tr_prob_ada, clarity_val_prob_ada])

    df['clarity_prob_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_b.csv'), index=None)
    df['clarity_prob_b_2'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_b_2.csv'), index=None)
    df['clarity_prob_lgb_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_lgb_b.csv'), index=None)
    df['clarity_prob_rf_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_rf_b.csv'), index=None)
    df['clarity_prob_dart_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_dart_b.csv'), index=None)
    df['clarity_prob_ada_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_ada_b.csv'), index=None)

    print("conciseness ensemble with clarity feature")

    rf = RandomForestClassifier(n_estimators=300, n_jobs=4)

    param_conciseness_dart = {
        'task': 'train',
        'boosting_type': 'dart',
        'objective': 'binary',
        'metric': 'rmse',
        'num_leaves': 127,
        'max_depth': 9,
        'learning_rate': 0.1,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.6,
        'bagging_freq': 6,
        'verbose': 0
    }

    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=9),
                             algorithm="SAMME.R",
                             learning_rate=0.06,
                             n_estimators=100)

    conciseness_tr_prob = np.zeros(tr_len)
    conciseness_val_prob = np.zeros(val_len)
    conciseness_tr_prob_2 = np.zeros(tr_len)
    conciseness_val_prob_2 = np.zeros(val_len)
    conciseness_tr_prob_lgb = np.zeros(tr_len)
    conciseness_val_prob_lgb = np.zeros(val_len)
    conciseness_tr_prob_rf = np.zeros(tr_len)
    conciseness_val_prob_rf = np.zeros(val_len)
    conciseness_tr_prob_dart = np.zeros(tr_len)
    conciseness_val_prob_dart = np.zeros(val_len)
    conciseness_tr_prob_ada = np.zeros(tr_len)
    conciseness_val_prob_ada = np.zeros(val_len)

    print "ensemble xgb"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=141)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_conciseness = xgb.DMatrix(X_test, label=y_test)

        bst_conciseness_train = xgb.train(param_conciseness
                                          , dtrain_conciseness
                                          , 1200
                                          )

        conciseness_tr_prob[test_index] = bst_conciseness_train.predict(dtrain_other_conciseness)
        conciseness_val_prob += bst_conciseness_train.predict(dvalid_conciseness) / 5.0

    print "ensemble xgb2"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=142)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_conciseness = xgb.DMatrix(X_test, label=y_test)

        bst_conciseness_train_2 = xgb.train(param_conciseness_2
                                            , dtrain_conciseness
                                            , 580
                                            )

        conciseness_tr_prob_2[test_index] = bst_conciseness_train_2.predict(dtrain_other_conciseness)
        conciseness_val_prob_2 += bst_conciseness_train_2.predict(dvalid_conciseness) / 5.0

    print "ensemble lgb"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=143)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness_lgb = lgb.Dataset(X_train, label=y_train)
        dtrain_other_conciseness_lgb = lgb.Dataset(X_test, label=y_test)

        gbm_conciseness_train = lgb.train(param_conciseness_lgb
                                          , dtrain_conciseness_lgb
                                          , num_boost_round=320
                                          )

        conciseness_tr_prob_lgb[test_index] = gbm_conciseness_train.predict(X_test)
        conciseness_val_prob_lgb += gbm_conciseness_train.predict(val_df_conciseness) / 5.0

    print "ensemble rf"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=144)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        rf.fit(X_train, y_train)
        conciseness_tr_prob_rf[test_index] = rf.predict_proba(X_test)[:, 1]
        conciseness_val_prob_rf += rf.predict_proba(val_df_conciseness)[:, 1] / 5.0

    print "ensemble dart"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=145)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness_lgb = lgb.Dataset(X_train, label=y_train)

        dart_conciseness_train = lgb.train(param_conciseness_dart
                                           , dtrain_conciseness_lgb
                                           , num_boost_round=150
                                           )

        conciseness_tr_prob_dart[test_index] = dart_conciseness_train.predict(X_test)
        conciseness_val_prob_dart += dart_conciseness_train.predict(val_df_conciseness) / 5.0

    print "ensemble ada"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=146)
    for train_index, test_index in skf.split(tr_df_conciseness, tr_conciseness):
        X_train, X_test = tr_df_conciseness[train_index], tr_df_conciseness[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        ada.fit(X_train, y_train)

        conciseness_tr_prob_ada[test_index] = ada.predict_proba(X_test)[:, 1]
        conciseness_val_prob_ada += ada.predict_proba(val_df_conciseness)[:, 1] / 5.0

    df['conciseness_prob_b'] = np.hstack([conciseness_tr_prob, conciseness_val_prob])
    df['conciseness_prob_b_2'] = np.hstack([conciseness_tr_prob_2, conciseness_val_prob_2])
    df['conciseness_prob_lgb_b'] = np.hstack([conciseness_tr_prob_lgb, conciseness_val_prob_lgb])
    df['conciseness_prob_rf_b'] = np.hstack([conciseness_tr_prob_rf, conciseness_val_prob_rf])
    df['conciseness_prob_dart_b'] = np.hstack([conciseness_tr_prob_dart, conciseness_val_prob_dart])
    df['conciseness_prob_ada_b'] = np.hstack([conciseness_tr_prob_ada, conciseness_val_prob_ada])

    df['conciseness_prob_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_b.csv'), index=None)
    df['conciseness_prob_b_2'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_b_2.csv'), index=None)
    df['conciseness_prob_lgb_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_lgb_b.csv'), index=None)
    df['conciseness_prob_rf_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_rf_b.csv'), index=None)
    df['conciseness_prob_dart_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_dart_b.csv'), index=None)
    df['conciseness_prob_ada_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_ada_b.csv'), index=None)

    print('ensemble stage1 finished, save file to disk')

    df.to_csv(os.path.join(feature_dir, 'df_feature_stage1.csv'), encoding='utf-8')