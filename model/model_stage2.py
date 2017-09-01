import os
import random
from scipy import sparse
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import neighbors
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

re_generate = False

feature_dir = '../data/features/'
datadir = '../data/'

ensemble_dir = os.path.join(datadir, 'ensemble')
if not os.path.exists(ensemble_dir):
    os.makedirs(ensemble_dir)

print ('loading features')
df = pd.read_csv(os.path.join(feature_dir, 'df_feature_stage1.csv'))
sparse_tfidf_title_clarity = sparse.load_npz(os.path.join(feature_dir, 'sparse_clarity.npz'))
sparse_tfidf_title_conciseness = sparse.load_npz(os.path.join(feature_dir, 'sparse_conciseness.npz'))

tr_clarity = pd.read_csv(datadir + 'training/clarity_train.labels', header=None)
tr_conciseness = pd.read_csv(datadir + 'training/conciseness_train.labels', header=None)

tr_len = tr_clarity.shape[0]

tr_clarity = np.asarray(tr_clarity).reshape(tr_len)
tr_conciseness = np.asarray(tr_conciseness).reshape(tr_len)

# # at this stage, all the features are numeric so we can use almost all the models
if re_generate:
    features_clarity = ['my'
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

    features_conciseness = [
                            'country_id'
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
                            , 'title_type_check_num'
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

    tr_df = df[df['select'] == 1]
    valid_df = df[df['select'] == 0]

    tr_len = tr_df.shape[0]
    val_len = valid_df.shape[0]

    tr_clarity = np.asarray(tr_clarity).reshape(tr_len)
    tr_conciseness = np.asarray(tr_conciseness).reshape(tr_len)

    tr_df_clarity = sparse.hstack([tr_df[features_clarity], sparse_tfidf_title_clarity[:tr_len]]).tocsr()
    val_df_clarity = sparse.hstack([valid_df[features_clarity], sparse_tfidf_title_clarity[tr_len:]]).tocsr()
    df_clarity = sparse.hstack([df[features_clarity], sparse_tfidf_title_clarity]).tocsr()

    # # using all conciseness features to
    tr_df_clarity_c = tr_df[features_conciseness].values
    val_df_clarity_c = valid_df[features_conciseness].values

    dtrain_clarity = xgb.DMatrix(data=tr_df_clarity, label=tr_clarity)
    dvalid_clarity = xgb.DMatrix(data=val_df_clarity)

    dtrain_clarity_c = xgb.DMatrix(data=tr_df_clarity_c, label=tr_clarity)
    dvalid_clarity_c = xgb.DMatrix(data=val_df_clarity_c)

    num_round = 2000

    param_clarity_c = {'booster': 'gbtree'
                       , 'max_depth': 5
                       , 'eta': 0.03
                       , 'gamma': 1
                       , 'silent': 1
                       , 'objective': 'reg:logistic'
                       , 'eval_metric': 'rmse'
                       , 'colsample_bytree': 0.75
                       , 'col_sample_bylevel': 0.6
                       , 'subsample': 0.7
                      }

    # random.seed(325)
    # bst_clarity_cv_c = xgb.cv(param_clarity_c
    #                           , dtrain_clarity_c
    #                           , num_round
    #                           , nfold=5
    #                           , metrics=['rmse']
    #                           , verbose_eval=True
    #                           , early_stopping_rounds=30
    #                           )

    print("scaling data")
    scaler = preprocessing.StandardScaler(with_mean=False)
    df_clarity_sparse = scaler.fit_transform(df_clarity)
    tr_df_clarity_sparse = df_clarity_sparse[:tr_len]
    val_df_clarity_sparse = df_clarity_sparse[tr_len:]

    clarity_tr_prob_xgb_c = np.zeros(tr_len)
    clarity_val_prob_xgb_c = np.zeros(val_len)
    clarity_tr_prob_knn = np.zeros(tr_len)
    clarity_val_prob_knn = np.zeros(val_len)
    clarity_tr_prob_lsvm = np.zeros(tr_len)
    clarity_val_prob_lsvm = np.zeros(val_len)
    clarity_tr_prob_nb = np.zeros(tr_len)
    clarity_val_prob_nb = np.zeros(val_len)
    clarity_tr_prob_mlp = np.zeros(tr_len)
    clarity_val_prob_mlp = np.zeros(val_len)
    clarity_tr_prob_rnr = np.zeros(tr_len)
    clarity_val_prob_rnr = np.zeros(val_len)
    clarity_tr_prob_lasso = np.zeros(tr_len)
    clarity_val_prob_lasso = np.zeros(val_len)

    knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance')
    lsvm = linear_model.SGDClassifier(loss='log', n_jobs=4)
    nb = GaussianNB()
    mlp = MLPClassifier(alpha=0.001, learning_rate='invscaling')
    rnr = neighbors.RadiusNeighborsRegressor()
    lasso = linear_model.Lasso(alpha=0.1)

    print ("ensemble model with numeric feature input for clarity")

    print ("ensemble xgb c")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=151)
    for train_index, test_index in skf.split(tr_df_clarity_c, tr_clarity):
        X_train, X_test = tr_df_clarity_c[train_index], tr_df_clarity_c[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        dtrain_clarity = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_clarity = xgb.DMatrix(X_test, label=y_test)

        bst_clarity_train = xgb.train(param_clarity_c
                                      , dtrain_clarity
                                      , 250
                                      )

        clarity_tr_prob_xgb_c[test_index] = bst_clarity_train.predict(dtrain_other_clarity)
        clarity_val_prob_xgb_c += bst_clarity_train.predict(dvalid_clarity_c) / 5.0

    print ("ensemble knn")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=152)
    for train_index, test_index in skf.split(tr_df_clarity_sparse, tr_clarity):
        X_train, X_test = tr_df_clarity_sparse[train_index], tr_df_clarity_sparse[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        knn.fit(X_train, y_train)

        clarity_tr_prob_knn[test_index] = knn.predict_proba(X_test)[:, 1]
        clarity_val_prob_knn += knn.predict_proba(val_df_clarity_sparse)[:, 1] / 5.0

    print ("ensemble lsvm")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=153)
    for train_index, test_index in skf.split(tr_df_clarity_sparse, tr_clarity):
        X_train, X_test = tr_df_clarity_sparse[train_index], tr_df_clarity_sparse[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        lsvm.fit(X_train, y_train)

        clarity_tr_prob_lsvm[test_index] = lsvm.predict_proba(X_test)[:, 1]
        clarity_val_prob_lsvm += lsvm.predict_proba(val_df_clarity_sparse)[:, 1] / 5.0

    print ("ensemble nb")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=154)
    for train_index, test_index in skf.split(tr_df_clarity_sparse, tr_clarity):
        X_train, X_test = tr_df_clarity_sparse.toarray()[train_index], tr_df_clarity_sparse.toarray()[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        nb.fit(X_train, y_train)

        clarity_tr_prob_nb[test_index] = nb.predict_proba(X_test)[:, 1]
        clarity_val_prob_nb += nb.predict_proba(val_df_clarity_sparse.toarray())[:, 1] / 5.0

    print ("ensemble mlp")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=155)
    for train_index, test_index in skf.split(tr_df_clarity_sparse, tr_clarity):
        X_train, X_test = tr_df_clarity_sparse[train_index], tr_df_clarity_sparse[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        mlp.fit(X_train, y_train)

        clarity_tr_prob_mlp[test_index] = mlp.predict_proba(X_test)[:, 1]
        clarity_val_prob_mlp += mlp.predict_proba(val_df_clarity_sparse)[:, 1] / 5.0

    print ("ensemble rnr")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=156)
    for train_index, test_index in skf.split(tr_df_clarity_sparse, tr_clarity):
        X_train, X_test = tr_df_clarity_sparse[train_index], tr_df_clarity_sparse[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        rnr.fit(X_train, y_train)

        clarity_tr_prob_rnr[test_index] = rnr.predict(X_test)
        clarity_val_prob_rnr += rnr.predict(val_df_clarity_sparse) / 5.0

    print ("ensemble lasso")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=158)
    for train_index, test_index in skf.split(tr_df_clarity_sparse, tr_clarity):
        X_train, X_test = tr_df_clarity_sparse[train_index], tr_df_clarity_sparse[test_index]
        y_train, y_test = tr_clarity[train_index], tr_clarity[test_index]

        lasso.fit(X_train, y_train)

        clarity_tr_prob_lasso[test_index] = lasso.predict(X_test)
        clarity_val_prob_lasso += lasso.predict(val_df_clarity_sparse) / 5.0

    df['clarity_prob_xgb_c_b'] = np.hstack([clarity_tr_prob_xgb_c, clarity_val_prob_xgb_c])
    df['clarity_prob_knn_b'] = np.hstack([clarity_tr_prob_knn, clarity_val_prob_knn])
    df['clarity_prob_lsvm_b'] = np.hstack([clarity_tr_prob_lsvm, clarity_val_prob_lsvm])
    df['clarity_prob_nb_b'] = np.hstack([clarity_tr_prob_nb, clarity_val_prob_nb])
    df['clarity_prob_mlp_b'] = np.hstack([clarity_tr_prob_mlp, clarity_val_prob_mlp])
    df['clarity_prob_rnr_b'] = np.hstack([clarity_tr_prob_rnr, clarity_val_prob_rnr])
    df['clarity_prob_lasso_b'] = np.hstack([clarity_tr_prob_lasso, clarity_val_prob_lasso])

    df['clarity_prob_xgb_c_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_xgb_c_b.csv'), index=None)
    df['clarity_prob_knn_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_knn_b.csv'), index=None)
    df['clarity_prob_lsvm_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_lsvm_b.csv'), index=None)
    df['clarity_prob_nb_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_nb_b.csv'), index=None)
    df['clarity_prob_mlp_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_mlp_b.csv'), index=None)
    df['clarity_prob_rnr_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_rnr_b.csv'), index=None)
    df['clarity_prob_lasso_b'].to_csv(os.path.join(ensemble_dir, 'clarity_prob_lasso_b.csv'), index=None)

    features_conciseness = [
                            'country_id'
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
                            , 'title_type_check_num'
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
    df_conciseness = sparse.hstack([df[features_conciseness], sparse_tfidf_title_conciseness]).tocsr()

    tr_df_conciseness_c = tr_df[features_conciseness].values
    val_df_conciseness_c = valid_df[features_conciseness].values

    dtrain_conciseness_c = xgb.DMatrix(data=tr_df_conciseness_c, label=tr_conciseness)
    dvalid_conciseness_c = xgb.DMatrix(data=val_df_conciseness_c)

    param_conciseness_c = {'booster': 'gbtree'
                           , 'max_depth': 4
                           , 'eta': 0.05
                           , 'gamma': 1.0
                           , 'silent': 1
                           , 'objective': 'reg:logistic'
                           , 'eval_metric': 'rmse'
                           , 'colsample_bytree': 0.7
                           , 'col_sample_bylevel': 0.6
                           , 'subsample': 0.7
                          }

    random.seed(333)

    bst_conciseness_cv_c = xgb.cv(param_conciseness_c
                                  , dtrain_conciseness_c
                                  , num_round
                                  , nfold=5
                                  , metrics=['rmse']
                                  , verbose_eval=20
                                  , early_stopping_rounds=30
                                  )

    print("scaling data")

    df_conciseness_sparse = scaler.fit_transform(df_conciseness)
    tr_df_conciseness_sparse = df_conciseness_sparse[:tr_len]
    val_df_conciseness_sparse = df_conciseness_sparse[tr_len:]

    conciseness_tr_prob_xgb_c = np.zeros(tr_len)
    conciseness_val_prob_xgb_c = np.zeros(val_len)
    conciseness_tr_prob_knn = np.zeros(tr_len)
    conciseness_val_prob_knn = np.zeros(val_len)
    conciseness_tr_prob_lsvm = np.zeros(tr_len)
    conciseness_val_prob_lsvm = np.zeros(val_len)
    conciseness_tr_prob_nb = np.zeros(tr_len)
    conciseness_val_prob_nb = np.zeros(val_len)
    conciseness_tr_prob_mlp = np.zeros(tr_len)
    conciseness_val_prob_mlp = np.zeros(val_len)
    conciseness_tr_prob_rnr = np.zeros(tr_len)
    conciseness_val_prob_rnr = np.zeros(val_len)
    conciseness_tr_prob_lasso = np.zeros(tr_len)
    conciseness_val_prob_lasso = np.zeros(val_len)

    knn = neighbors.KNeighborsClassifier(n_neighbors=7, weights='distance')
    lsvm = linear_model.SGDClassifier(loss='log', n_jobs=4)
    nb = GaussianNB()
    mlp = MLPClassifier(alpha=0.001)
    rnr = neighbors.RadiusNeighborsRegressor()
    lasso = linear_model.Lasso(alpha=0.1)

    print ("ensemble model with numeric feature input for conciseness")

    print "ensemble xgb c"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=171)
    for train_index, test_index in skf.split(tr_df_conciseness_c, tr_conciseness):
        X_train, X_test = tr_df_conciseness_c[train_index], tr_df_conciseness_c[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        dtrain_conciseness = xgb.DMatrix(X_train, label=y_train)
        dtrain_other_conciseness = xgb.DMatrix(X_test, label=y_test)

        bst_conciseness_train_c = xgb.train(param_conciseness_c
                                            , dtrain_conciseness
                                            , 480
                                            )

        conciseness_tr_prob_xgb_c[test_index] = bst_conciseness_train_c.predict(dtrain_other_conciseness)
        conciseness_val_prob_xgb_c += bst_conciseness_train_c.predict(dvalid_conciseness_c) / 5.0

    print ("ensemble knn")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=172)
    for train_index, test_index in skf.split(tr_df_conciseness_sparse, tr_conciseness):
        X_train, X_test = tr_df_conciseness_sparse[train_index], tr_df_conciseness_sparse[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        knn.fit(X_train, y_train)

        conciseness_tr_prob_knn[test_index] = knn.predict_proba(X_test)[:, 1]
        conciseness_val_prob_knn += knn.predict_proba(val_df_conciseness_sparse)[:, 1] / 5.0

    print ("ensemble lsvm")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=173)
    for train_index, test_index in skf.split(tr_df_conciseness_sparse, tr_conciseness):
        X_train, X_test = tr_df_conciseness_sparse[train_index], tr_df_conciseness_sparse[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        lsvm.fit(X_train, y_train)

        conciseness_tr_prob_lsvm[test_index] = lsvm.predict_proba(X_test)[:, 1]
        conciseness_val_prob_lsvm += lsvm.predict_proba(val_df_conciseness_sparse)[:, 1] / 5.0

    print ("ensemble nb")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=174)
    for train_index, test_index in skf.split(tr_df_conciseness_sparse, tr_conciseness):
        X_train, X_test = tr_df_conciseness_sparse.toarray()[train_index], tr_df_conciseness_sparse.toarray()[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        nb.fit(X_train, y_train)

        conciseness_tr_prob_nb[test_index] = nb.predict_proba(X_test)[:, 1]
        conciseness_val_prob_nb += nb.predict_proba(val_df_conciseness_sparse.toarray())[:, 1] / 5.0

    print ("ensemble mlp")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=175)
    for train_index, test_index in skf.split(tr_df_conciseness_sparse, tr_conciseness):
        X_train, X_test = tr_df_conciseness_sparse[train_index], tr_df_conciseness_sparse[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        mlp.fit(X_train, y_train)

        conciseness_tr_prob_mlp[test_index] = mlp.predict_proba(X_test)[:, 1]
        conciseness_val_prob_mlp += mlp.predict_proba(val_df_conciseness_sparse)[:, 1] / 5.0

    print ("ensemble rnr")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=176)
    for train_index, test_index in skf.split(tr_df_conciseness_sparse, tr_conciseness):
        X_train, X_test = tr_df_conciseness_sparse[train_index], tr_df_conciseness_sparse[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        rnr.fit(X_train, y_train)

        conciseness_tr_prob_rnr[test_index] = rnr.predict(X_test)
        conciseness_val_prob_rnr += rnr.predict(val_df_conciseness_sparse) / 5.0

    print ("ensemble lasso")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=178)
    for train_index, test_index in skf.split(tr_df_conciseness_sparse, tr_conciseness):
        X_train, X_test = tr_df_conciseness_sparse[train_index], tr_df_conciseness_sparse[test_index]
        y_train, y_test = tr_conciseness[train_index], tr_conciseness[test_index]

        lasso.fit(X_train, y_train)

        conciseness_tr_prob_lasso[test_index] = lasso.predict(X_test)
        conciseness_val_prob_lasso += lasso.predict(val_df_conciseness_sparse) / 5.0

    df['conciseness_prob_xgb_c_b'] = np.hstack([conciseness_tr_prob_xgb_c, conciseness_val_prob_xgb_c])
    df['conciseness_prob_knn_b'] = np.hstack([conciseness_tr_prob_knn, conciseness_val_prob_knn])
    df['conciseness_prob_lsvm_b'] = np.hstack([conciseness_tr_prob_lsvm, conciseness_val_prob_lsvm])
    df['conciseness_prob_nb_b'] = np.hstack([conciseness_tr_prob_nb, conciseness_val_prob_nb])
    df['conciseness_prob_mlp_b'] = np.hstack([conciseness_tr_prob_mlp, conciseness_val_prob_mlp])
    df['conciseness_prob_rnr_b'] = np.hstack([conciseness_tr_prob_rnr, conciseness_val_prob_rnr])
    df['conciseness_prob_lasso_b'] = np.hstack([conciseness_tr_prob_lasso, conciseness_val_prob_lasso])

    df['conciseness_prob_xgb_c_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_xgb_c_b.csv'), index=None)
    df['conciseness_prob_knn_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_knn_b.csv'), index=None)
    df['conciseness_prob_lsvm_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_lsvm_b.csv'), index=None)
    df['conciseness_prob_nb_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_nb_b.csv'), index=None)
    df['conciseness_prob_mlp_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_mlp_b.csv'), index=None)
    df['conciseness_prob_rnr_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_rnr_b.csv'), index=None)
    df['conciseness_prob_lasso_b'].to_csv(os.path.join(ensemble_dir, 'conciseness_prob_lasso_b.csv'), index=None)

    print('ensemble stage2 finished, save file to disk')

    df.to_csv(os.path.join(feature_dir, 'df_feature_stage2.csv'), encoding='utf-8')

else:
    df = pd.read_csv(os.path.join(feature_dir, 'df_feature_stage2.csv'), encoding='utf-8')


df = pd.concat([df, pd.get_dummies(pd.qcut(df['conciseness_prob_b'], 10,
                    labels=["conciseness_prob_b_q1", "conciseness_prob_b_q2", "conciseness_prob_b_q3", 'conciseness_prob_b_q4', 'conciseness_prob_b_q5',
                    "conciseness_prob_b_q6", "conciseness_prob_b_q7", "conciseness_prob_b_q8", 'conciseness_prob_b_q9', 'conciseness_prob_b_q10']))], axis=1)

# # preform model ensemble

features_clarity_models = [
                           'clarity_prob'
                           , 'clarity_prob_2'
                           , 'clarity_prob_lgb'
                           , 'clarity_prob_rf'
                           , 'clarity_prob_dart'
                           , 'clarity_prob_ada'
                           , 'clarity_prob_b'
                           , 'clarity_prob_b_2'
                           , 'clarity_prob_lgb_b'
                           , 'clarity_prob_rf_b'
                           , 'clarity_prob_dart_b'
                           , 'clarity_prob_ada_b'
                           , 'clarity_prob_knn_b'
                           , 'clarity_prob_lsvm_b'
                           , 'clarity_prob_nb_b'
                           , 'clarity_prob_mlp_b'

                           , 'clarity_prob_xgb_c_b'
                           , 'clarity_prob_rnr_b'
                           , 'clarity_prob_lasso_b'

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
                           # , "clarity_prob_2_q1"
                           # , "clarity_prob_2_q2"
                           # , "clarity_prob_2_q3"
                           # , 'clarity_prob_2_q4'
                           # , 'clarity_prob_2_q5'

                           , 'conciseness_prob'
                           , 'conciseness_prob_2'
                           , 'conciseness_prob_lgb'
                           , 'conciseness_prob_rf'
                           , 'conciseness_prob_dart'
                           , 'conciseness_prob_ada'
                           , 'conciseness_prob_b'
                           , 'conciseness_prob_b_2'
                           , 'conciseness_prob_lgb_b'
                           , 'conciseness_prob_rf_b'
                           , 'conciseness_prob_dart_b'
                           , 'conciseness_prob_ada_b'
                           , 'conciseness_prob_knn_b'
                           , 'conciseness_prob_lsvm_b'
                           , 'conciseness_prob_nb_b'
                           , 'conciseness_prob_mlp_b'

                           , 'conciseness_prob_xgb_c_b'
                           , 'conciseness_prob_rnr_b'
                           , 'conciseness_prob_lasso_b'

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

                           , "conciseness_prob_b_q1"
                           , "conciseness_prob_b_q2"
                           , "conciseness_prob_b_q3"
                           , 'conciseness_prob_b_q4'
                           , 'conciseness_prob_b_q5'
                           , "conciseness_prob_b_q6"
                           , "conciseness_prob_b_q7"
                           , "conciseness_prob_b_q8"
                           , 'conciseness_prob_b_q9'
                           , 'conciseness_prob_b_q10'

                           , "conciseness_prob_2_q1"
                           , "conciseness_prob_2_q2"
                           , "conciseness_prob_2_q3"
                           , 'conciseness_prob_2_q4'
                           , 'conciseness_prob_2_q5'
                         ]

tr_df = df[df['select'] == 1]
valid_df = df[df['select'] == 0]


tr_df_clarity_m = tr_df[features_clarity_models]
val_df_clarity_m = valid_df[features_clarity_models]

dtrain_clarity = xgb.DMatrix(data=tr_df_clarity_m, label=tr_clarity)
dvalid_clarity = xgb.DMatrix(data=val_df_clarity_m)

param_clarity = {'booster': 'gbtree'
                 , 'max_depth': 3
                 , 'eta': 0.02
                 , 'gamma': 1
                 , 'silent': 1
                 , 'objective': 'binary:logistic'
                 , 'eval_metric': 'rmse'
                 , 'colsample_bytree': 0.75
                 , 'col_sample_bylevel': 0.6
                 , 'subsample': 0.7
                 }

num_round = 2000

random.seed(325)
bst_clarity_cv = xgb.cv(param_clarity
                        , dtrain_clarity
                        , num_round
                        , nfold=5
                        , metrics=['rmse']
                        , verbose_eval=True
                        , early_stopping_rounds=30
                        )

random.seed(324)
bst_clarity_train = xgb.train(param_clarity
                              , dtrain_clarity
                              , 337
                              )

dtrain_clarity_lgb = lgb.Dataset(tr_df_clarity_m, tr_clarity)
dvalid_clarity_lgb = lgb.Dataset(val_df_clarity_m)

param_clarity_lgb = {
                    'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'regression_l2',
                    'metric': {'mse'},
                    'lambda_l1': 1.0,
                    'max_depth': 3,
                    'num_leaves': 7,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.7,
                    'bagging_fraction': 0.5,
                    'bagging_freq': 3,
                    'verbose': 0,
                    'num_threads': 4
                    }

gbm_clarity_cv = lgb.cv(param_clarity_lgb,
                        dtrain_clarity_lgb,
                        num_boost_round=num_round,
                        verbose_eval=5,
                        early_stopping_rounds=30
                        )
random.seed(326)
gbm_clarity = lgb.train(param_clarity_lgb
                        , dtrain_clarity_lgb
                        , num_boost_round=600
                        )

features_conciseness_models = [
                                'clarity_prob'
                                , 'clarity_prob_2'
                                , 'clarity_prob_lgb'
                                , 'clarity_prob_rf'
                                , 'clarity_prob_dart'
                                , 'clarity_prob_ada'
                                , 'clarity_prob_b'
                                , 'clarity_prob_b_2'
                                , 'clarity_prob_lgb_b'
                                , 'clarity_prob_rf_b'
                                , 'clarity_prob_dart_b'
                                , 'clarity_prob_ada_b'
                                , 'clarity_prob_knn_b'
                                , 'clarity_prob_lsvm_b'
                                , 'clarity_prob_nb_b'
                                , 'clarity_prob_mlp_b'

                                , 'clarity_prob_xgb_c_b'
                                , 'clarity_prob_rnr_b'
                                , 'clarity_prob_lasso_b'

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

                                , 'conciseness_prob'
                                , 'conciseness_prob_2'
                                , 'conciseness_prob_lgb'
                                , 'conciseness_prob_rf'
                                , 'conciseness_prob_dart'
                                , 'conciseness_prob_ada'
                                , 'conciseness_prob_b'
                                , 'conciseness_prob_b_2'
                                , 'conciseness_prob_lgb_b'
                                , 'conciseness_prob_rf_b'
                                , 'conciseness_prob_dart_b'
                                , 'conciseness_prob_ada_b'
                                , 'conciseness_prob_knn_b'
                                , 'conciseness_prob_lsvm_b'
                                , 'conciseness_prob_nb_b'
                                , 'conciseness_prob_mlp_b'

                                , 'conciseness_prob_xgb_c_b'
                                , 'conciseness_prob_rnr_b'
                                , 'conciseness_prob_lasso_b'

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

                                # , "conciseness_prob_b_q1"
                                # , "conciseness_prob_b_q2"
                                # , "conciseness_prob_b_q3"
                                # , 'conciseness_prob_b_q4'
                                # , 'conciseness_prob_b_q5'
                                # , "conciseness_prob_b_q6"
                                # , "conciseness_prob_b_q7"
                                # , "conciseness_prob_b_q8"
                                # , 'conciseness_prob_b_q9'
                                # , 'conciseness_prob_b_q10'

                                , "conciseness_prob_2_q1"
                                , "conciseness_prob_2_q2"
                                , "conciseness_prob_2_q3"
                                , 'conciseness_prob_2_q4'
                                , 'conciseness_prob_2_q5'
                         ]

tr_df = df[df['select'] == 1]
valid_df = df[df['select'] == 0]

tr_df_conciseness_m = tr_df[features_conciseness_models]
val_df_conciseness_m = valid_df[features_conciseness_models]


dtrain_conciseness = xgb.DMatrix(data=tr_df_conciseness_m, label=tr_conciseness)
dvalid_conciseness = xgb.DMatrix(data=val_df_conciseness_m)

param_conciseness = {'booster': 'gbtree'
                     , 'max_depth': 4
                     , 'eta': 0.03
                     , 'gamma': 1
                     , 'silent': 1
                     , 'objective': 'binary:logistic'
                     , 'eval_metric': 'rmse'
                     , 'colsample_bytree': 0.7
                     , 'col_sample_bylevel': 0.6
                     , 'subsample': 0.75
                     }

random.seed(327)
bst_conciseness_cv = xgb.cv(param_conciseness
                            , dtrain_conciseness
                            , num_round
                            , nfold=5
                            , metrics=['rmse']
                            , verbose_eval=True
                            , early_stopping_rounds=30
                            )

np.random.seed(328)
bst_conciseness_train = xgb.train(param_conciseness
                                  , dtrain_conciseness
                                  , 233
                                  )

dtrain_conciseness_lgb = lgb.Dataset(tr_df_conciseness_m, tr_conciseness)
dvalid_conciseness_lgb = lgb.Dataset(val_df_conciseness_m)

param_conciseness_lgb = {
                        'task': 'train',
                        'boosting_type': 'gbdt',
                        'objective': 'binary',
                        'metric': {'mse'},
                        'lambda_l1': 1.0,
                        'max_depth': 3,
                        'num_leaves': 7,
                        'learning_rate': 0.01,
                        'feature_fraction': 0.7,
                        'bagging_fraction': 0.5,
                        'bagging_freq': 3,
                        'verbose': 0
                        }

gbm_conciseness_cv = lgb.cv(param_conciseness_lgb,
                            dtrain_conciseness_lgb,
                            num_boost_round=num_round,
                            verbose_eval=5,
                            early_stopping_rounds=50
                            )

random.seed(329)
gbm_conciseness = lgb.train(param_conciseness_lgb
                            , dtrain_conciseness_lgb
                            , num_boost_round=2000
                            )

np.savetxt(os.path.join(ensemble_dir, "clarity_test_xgb.predict"),
           bst_clarity_train.predict(dvalid_clarity), fmt='%.6f')
np.savetxt(os.path.join(ensemble_dir, "clarity_test_lgb.predict"),
           gbm_clarity.predict(valid_df[features_clarity_models]), fmt='%.6f')

np.savetxt(os.path.join(ensemble_dir, "conciseness_test_xgb.predict"),
           bst_conciseness_train.predict(dvalid_conciseness), fmt='%.6f')
np.savetxt(os.path.join(ensemble_dir, "conciseness_test_lgb.predict"),
           gbm_conciseness.predict(valid_df[features_conciseness_models]), fmt='%.6f')