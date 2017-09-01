import os
import numpy as np
import pandas as pd
import collections
from scipy import sparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import keras
from keras import optimizers
from keras.models import Model
from keras.layers import Input, Reshape, Dense, Dropout, LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

np.random.seed(7)

re_generate = False

feature_dir = '../data/features/'
datadir = '../data/'

ensemble_dir = os.path.join(datadir, 'ensemble')
if not os.path.exists(ensemble_dir):
    os.makedirs(ensemble_dir)

print ('loading features')
df = pd.read_csv(os.path.join(feature_dir, 'df_feature_stage2.csv'))

df_tr = pd.read_csv(datadir + 'training/data_train.csv'
                    , header=None, names=['country', 'sku_id', 'title', 'category_lvl_1',
                                          'category_lvl_2', 'category_lvl_3', 'short_description', 'price',
                                          'product_type'])

df_valid = pd.read_csv(datadir + 'testing/data_test.csv'
                       , header=None, names=['country', 'sku_id', 'title', 'category_lvl_1',
                                             'category_lvl_2', 'category_lvl_3', 'short_description', 'price',
                                             'product_type'])

sparse_tfidf_title_clarity = sparse.load_npz(os.path.join(feature_dir, 'sparse_clarity.npz'))
sparse_tfidf_title_conciseness = sparse.load_npz(os.path.join(feature_dir, 'sparse_conciseness.npz'))

tr_clarity = pd.read_csv(datadir + 'training/clarity_train.labels', header=None)
tr_conciseness = pd.read_csv(datadir + 'training/conciseness_train.labels', header=None)

features = ['my'
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
            , 'title_type_check_num'
            , 'title_C_upperratio'
            , 'title_C_upperwordratio'
            , 'description_li_num'

            , 'clarity_prob'
            , 'clarity_prob_2'
            , 'clarity_prob_lgb'
            , 'clarity_prob_rf'
            , 'clarity_prob_dart'
            , 'clarity_prob_ada'
            , 'clarity_prob_b'
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

            , 'conciseness_prob'
            # , 'conciseness_prob_2'
            # , 'conciseness_prob_lgb'
            # , 'conciseness_prob_rf'
            # , 'conciseness_prob_dart'
            # , 'conciseness_prob_ada'
            # , 'conciseness_prob_b'
            # , 'conciseness_prob_lgb_b'
            # , 'conciseness_prob_rf_b'
            # , 'conciseness_prob_dart_b'
            # , 'conciseness_prob_ada_b'
            # , 'conciseness_prob_knn_b'
            # , 'conciseness_prob_svm_b'
            # , 'conciseness_prob_nb_b'
            # , 'conciseness_prob_mlp_b'
            ]

min_max_scaler = preprocessing.MinMaxScaler()

df = df.fillna(-1)
df_scale = preprocessing.scale(df[features].values)
df_scale = pd.DataFrame(df_scale, columns=features)

df_tr_n = df_scale.iloc[:df_tr.shape[0]]
df_valid_n = df_scale.iloc[df_tr.shape[0]:]

X_train_f = df_tr_n[features].values
X_valid_f = df_valid_n[features].values

print (X_train_f.shape)
print("preparing word embedding")
if re_generate:
    words_list, words_list_tr, words_list_valid = [], [], []

    for i in range(df_tr.shape[0]):
        words_list.append(df_tr.iloc[i]['title'])
        words_list_tr.append(df_tr.iloc[i]['title'])

    for i in range(df_valid.shape[0]):
        words_list.append(df_valid.iloc[i]['title'])
        words_list_valid.append(df_valid.iloc[i]['title'])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words_list)

    tr_sequences = tokenizer.texts_to_sequences(words_list_tr)
    te_sequences = tokenizer.texts_to_sequences(words_list_valid)

    word_index = tokenizer.word_index
    max_review_length = 40

    X_train = pad_sequences(tr_sequences, maxlen=max_review_length)
    y_train = tr_clarity.values.reshape((tr_clarity.shape[0], 1))

    X_valid = pad_sequences(te_sequences, maxlen=max_review_length)

    nb_words = len(word_index)+1

    EMBEDDING_FILE = datadir + 'file_temp/glove.840B.300d.txt'

    embeddings_index = {}
    f = open(EMBEDDING_FILE)
    count = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((nb_words, 300))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    print("preparing category embedding")
    np.save(datadir + 'file_temp/embedding', embedding_matrix)
else:
    words_list, words_list_tr, words_list_valid = [], [], []

    for i in range(df_tr.shape[0]):
        words_list.append(df_tr.iloc[i]['title'])
        words_list_tr.append(df_tr.iloc[i]['title'])

    for i in range(df_valid.shape[0]):
        words_list.append(df_valid.iloc[i]['title'])
        words_list_valid.append(df_valid.iloc[i]['title'])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(words_list)

    tr_sequences = tokenizer.texts_to_sequences(words_list_tr)
    te_sequences = tokenizer.texts_to_sequences(words_list_valid)

    word_index = tokenizer.word_index
    max_review_length = 40
    nb_words = len(word_index)+1

    X_train = pad_sequences(tr_sequences, maxlen=max_review_length)
    y_train = tr_clarity.values.reshape((tr_clarity.shape[0], 1))

    X_valid = pad_sequences(te_sequences, maxlen=max_review_length)

    embedding_matrix = np.load(datadir + 'file_temp/embedding.npy')


def creat_cats(cat_list, df, cat_string):
    for i in range(df.shape[0]):
        cat_list.append(df[cat_string].iloc[i])
    return cat_list

cat_strings = ['sku_id', 'category_lvl_1', 'category_lvl_2', 'category_lvl_3']

cat_to_ix_lens = []
for cat_string in cat_strings:
    cats_list = []
    cats_list = creat_cats(cats_list, df_tr, cat_string)
    cats_list = creat_cats(cats_list, df_valid, cat_string)

    cats_count = collections.Counter(cats_list)

    cat_to_ix = {cat: i for i, cat in enumerate(cats_count.keys())}
    cat_to_ix_lens.append(len(cat_to_ix))
    X_train_cat, X_valid_cat = [], []

    for i in range(df_tr.shape[0]):
        idxs = [cat_to_ix[df_tr[cat_string].iloc[i]]]
        X_train_cat.append(idxs)

    for i in range(df_valid.shape[0]):
        idxs = [cat_to_ix[df_valid[cat_string].iloc[i]]]
        X_valid_cat.append(idxs)

    X_train_cat = np.asarray(X_train_cat)
    X_valid_cat = np.asarray(X_valid_cat)

    max_review_cat_length = 1
    X_train_cat = sequence.pad_sequences(X_train_cat, maxlen=max_review_cat_length)
    X_train = np.hstack([X_train, X_train_cat])
    X_valid_cat = sequence.pad_sequences(X_valid_cat, maxlen=max_review_cat_length)
    X_valid = np.hstack([X_valid, X_valid_cat])

X_train = np.hstack([X_train, X_train_f])
X_valid = np.hstack([X_valid, X_valid_f])

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=41)

main_input = Input(shape=(max_review_length,), dtype='int32', name='main_input')

Embedding_layer = Embedding(nb_words, 300, weights=[embedding_matrix], input_length=max_review_length, trainable=True)
word_embedding = Embedding_layer(main_input)

auxiliary_input_sku = Input(shape=(max_review_cat_length,), dtype='int32', name='aux_input_sku')
sku_embedding = Embedding(output_dim=16, input_dim=cat_to_ix_lens[0], input_length=max_review_cat_length)\
    (auxiliary_input_sku)

auxiliary_input_cat1 = Input(shape=(max_review_cat_length,), dtype='int32', name='aux_input_cat1')
cat1_emdedding = Embedding(output_dim=4, input_dim=cat_to_ix_lens[1], input_length=max_review_cat_length)\
    (auxiliary_input_cat1)

auxiliary_input_cat2 = Input(shape=(max_review_cat_length,), dtype='int32', name='aux_input_cat2')
cat2_emdedding = Embedding(output_dim=4, input_dim=cat_to_ix_lens[2], input_length=max_review_cat_length)\
    (auxiliary_input_cat2)

auxiliary_input_cat3 = Input(shape=(max_review_cat_length,), dtype='int32', name='aux_input_cat3')
cat3_emdedding = Embedding(output_dim=8, input_dim=cat_to_ix_lens[3], input_length=max_review_cat_length)\
    (auxiliary_input_cat3)

auxiliary_input_numeric = Input(shape=(X_train_f.shape[1],), dtype='float32', name='aux_input_numeric')


sku_embedding = Reshape((16, ))(sku_embedding)
cat1_emdedding = Reshape((4, ))(cat1_emdedding)
cat2_emdedding = Reshape((4, ))(cat2_emdedding)
cat3_emdedding = Reshape((8, ))(cat3_emdedding)

cat_emdedding = keras.layers.concatenate([sku_embedding, cat1_emdedding, cat2_emdedding, cat3_emdedding], axis=1)

lstm_out = LSTM(64, recurrent_dropout=0.1, return_sequences=True)(word_embedding)
lstm_out = LSTM(64)(lstm_out)

mlp1 = Dense(64, activation='relu')(auxiliary_input_numeric)
mlp2 = Dense(128, activation='relu')(mlp1)
mlp3 = Dense(192, activation='relu')(mlp2)
mlp3 = Dropout(0.1)(mlp3)

x = keras.layers.concatenate([lstm_out, cat_emdedding, mlp3])

x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.1)(x)

main_output = Dense(1, activation='sigmoid', name='main_output')(x)

model = Model(inputs=[main_input,  auxiliary_input_sku, auxiliary_input_cat1, auxiliary_input_cat2,
                      auxiliary_input_cat3, auxiliary_input_numeric],
              outputs=[main_output])

optim = optimizers.RMSprop(0.0005)
model.compile(optimizer=optim, loss='binary_crossentropy', metrics=['mse'])

bst_model_path = datadir+'file_temp/best_tmp.h5'

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

# hist = model.fit([X_train_split[:, :max_review_length], X_train_split[:, max_review_length],
#                   X_train_split[:, max_review_length+1],X_train_split[:, max_review_length+2],
#                   X_train_split[:, max_review_length+3], X_train_split[:, max_review_length+4:]], [y_train_split],
#                   validation_data=[[X_test_split[:, :max_review_length], X_test_split[:, max_review_length],
#                                     X_test_split[:, max_review_length+1], X_test_split[:, max_review_length+2],
#                                     X_test_split[:, max_review_length+3], X_test_split[:, max_review_length+4:]],
#                                    [y_test_split]], epochs=7, verbose=2, batch_size=24, callbacks=[early_stopping, model_checkpoint])

# model.load_weights(bst_model_path)
# bst_val_score = min(hist.history['val_loss'])

model.fit([X_train[:, :max_review_length], X_train[:, max_review_length], X_train[:, max_review_length+1],
           X_train[:, max_review_length+2], X_train[:, max_review_length+3],
           X_train[:, max_review_length+4:]], [y_train], epochs=3, verbose=2, batch_size=24)

valid_predict = model.predict([X_valid[:, :max_review_length], X_valid[:, max_review_length],
                               X_valid[:, max_review_length + 1], X_valid[:, max_review_length+2], X_valid[:, max_review_length+3],
                               X_valid[:, max_review_length+4:]], batch_size=24)

np.savetxt(os.path.join(ensemble_dir, 'clarity_test_lstm.predict'),
           valid_predict, fmt='%.6f')
