import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.models import Model
from keras.layers import Input, Dense, Dropout

np.random.seed(7)

feature_dir = '../data/features/'
datadir = '../data/'

ensemble_dir = os.path.join(datadir, 'ensemble')
if not os.path.exists(ensemble_dir):
    os.makedirs(ensemble_dir)

print ('loading features')
df = pd.read_csv(os.path.join(feature_dir, 'df_feature_stage2.csv'))

tr_clarity = pd.read_csv(datadir + 'training/clarity_train.labels', header=None)
tr_conciseness = pd.read_csv(datadir + 'training/conciseness_train.labels', header=None)

tr_len = tr_clarity.shape[0]

features = ['title_wordnum'
            , 'wordnum_q1'
            , 'wordnum_q2'
            , 'wordnum_q3'
            , 'wordnum_q4'
            , 'wordnum_q5'
            , 'title_stopsnum'
            , 'title_word_duplicate_cat_num'
            , 'title_nonalphanum'
            , 'title_word_duplicate_num'
            , 'title_word_duplicate_nums'
            , 'title_charnum'
            , 'title_avgwordlen'
            , 'title_nonengnum'
            , 'title_wordsynsetdepthsum'
            , 'price'
            , 'title_wordlemmassnum'
            , 'category_lvl_1_frequency'
            , 'category_lvl_2_frequency'
            , 'category_lvl_3_frequency'
            , 'title_wordcharlargenum'
            , 'title_uppernum'
            , 'title_cat3_wordnum_mean_contrast'
            , 'title_C_upperwordratio'
            , 'title_meaningword_ratio'
            , 'title_type_check_num'

            , 'title_word_lcs_num'
            , 'title_word_lcs_cat_num'
            , 'title_digitnum'
            , 'tittle_upper_word'
            , 'title_word_duplicate_num_cleaned'
            , 'title_word_duplicate_cat_num_cleaned'
            , 'description_C_upperratio'
            , 'description_C_upperwordratio'
            , 'description_li_num'

            , 'clarity_prob'
            # , 'clarity_prob_2'
            , 'clarity_prob_lgb'
            , 'clarity_prob_rf'
            , 'clarity_prob_dart'
            , 'clarity_prob_ada'
            , 'clarity_prob_b'
            # , 'clarity_prob_b_2'
            , 'clarity_prob_lgb_b'
            , 'clarity_prob_rf_b'
            , 'clarity_prob_dart_b'
            , 'clarity_prob_ada_b'

            , 'clarity_prob_xgb_c_b'
            , 'clarity_prob_rnr_b'
            , 'clarity_prob_lasso_b'

            , 'conciseness_prob'
            # , 'conciseness_prob_2'
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
            # , 'conciseness_prob_knn_b'
            # , 'conciseness_prob_lsvm_b'
            # , 'conciseness_prob_nb_b'
            # , 'conciseness_prob_mlp_b'

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

            , "conciseness_prob_2_q1"
            , "conciseness_prob_2_q2"
            , "conciseness_prob_2_q3"
            , 'conciseness_prob_2_q4'
            , 'conciseness_prob_2_q5'
            ]

min_max_scaler = preprocessing.StandardScaler()

df = df.fillna(-1)
df_scale = preprocessing.scale(df[features].values)
df_scale = pd.DataFrame(df_scale, columns=features)

df_tr_n = df_scale.iloc[:tr_len]
df_test_n = df_scale.iloc[tr_len:]

X_train = df_tr_n[features].values
X_test = df_test_n[features].values

y_train = tr_conciseness.values

X_train, y_train, X_test = np.asarray(X_train), np.asarray(y_train).reshape(len(y_train)), np.asarray(X_test)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=41)

auxiliary_input_numeric = Input(shape=(X_train.shape[1],), dtype='float32', name='aux_input_numeric')

mlp1 = Dense(64, activation='relu')(auxiliary_input_numeric)
mlp2 = Dense(128, activation='relu')(mlp1)
mlp3 = Dense(192, activation='relu')(mlp2)
mlp4 = Dense(32, activation='relu')(mlp3)

x = Dropout(0.1)(mlp4)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[auxiliary_input_numeric],
              outputs=[main_output])

optim = optimizers.Adam(0.0005)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['mse'])


model.fit([
           X_train_split], [y_train_split],
          validation_data=[[
                            X_test_split],
                           [y_test_split]], epochs=9, verbose=2, batch_size=24)

seed = 3211
np.random.seed(seed)
model.fit([
           X_train], [y_train], epochs=3, verbose=2, batch_size=24)

test_predict = model.predict([
                               X_test], batch_size=24)


np.savetxt(os.path.join(ensemble_dir, 'conciseness_test_mlp_%d.predict' % seed), test_predict, fmt='%.6f')