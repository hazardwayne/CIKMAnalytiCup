import os
import re
import pandas as pd
import numpy as np
import scipy.sparse  # v19.0/v19.1
import spacy
import enchant
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

re_generate = False
if re_generate:
    datadir = '../data/'
    engstops = set(stopwords.words("english"))
    stemmer = SnowballStemmer('english')
    nlp = spacy.load("en_default")

    df_tr = pd.read_csv(datadir+'training/data_train.csv'
                        , header=None, names=['country', 'sku_id', 'title', 'category_lvl_1',
                        'category_lvl_2', 'category_lvl_3', 'short_description', 'price', 'product_type'])

    df_tr['select'] = 1
    tr_clarity = pd.read_csv(datadir+'training/clarity_train.labels', header=None)
    tr_conciseness = pd.read_csv(datadir+'training/conciseness_train.labels', header=None)

    df_tr['clarity'] = tr_clarity
    df_tr['conciseness'] = tr_conciseness

    # we named test as valid
    df_valid = pd.read_csv(datadir+'testing/data_test.csv'
                           , header=None, names=['country', 'sku_id', 'title', 'category_lvl_1',
                           'category_lvl_2', 'category_lvl_3', 'short_description', 'price', 'product_type'])

    df_valid['select'] = 0

    df = pd.concat([df_tr, df_valid], axis=0, ignore_index=True)
    df = df.fillna('NA')
    df.loc[df['category_lvl_3'] == 'NA', 'category_lvl_3'] = df.loc[df['category_lvl_3'] == 'NA', 'category_lvl_2']

    df['cleaned_title'] = df['title'].apply(lambda x: ' '.join([stemmer.stem(w) for w in re.sub(r"[\'\"</\[\+>()-]", " ",unicode(x, 'utf-8')).split()]))

    df['title_entity'] = df['title'].apply(lambda x: nlp(unicode(x, 'utf-8')).ents)

    df['title_ent_cat_list'] = df['title_entity'].apply(lambda x: [len(item) for item in x])
    df['title_ent_cat_list_num_mean'] = df['title_ent_cat_list'].apply(lambda x: np.mean(x))
    df['title_ent_cat_list_num_std'] = df['title_ent_cat_list'].apply(lambda x: np.std(x))
    df['title_ent_cat_list_num_max'] = df['title_ent_cat_list'].apply(lambda x: np.max(x) if len(x) > 0 else 0)
    df['title_ent_cat_list_num'] = df['title_ent_cat_list'].apply(lambda x: len(x))

    print("creating statistic features")

    df['country'] = df['country'].astype('string')
    df['country_id'] = pd.factorize(df['country'].astype('category'))[0]
    df['sku_id'] = pd.factorize(df['sku_id'].astype('category'))[0]
    df = pd.concat([df, pd.get_dummies(df['country'])], axis=1)  # one-hot coding for country

    df['title_wordnum'] = df['title'].apply(lambda x: len(x.split(' ')))

    cat1 = ''.join([item+' ' for item in set(df['category_lvl_1'].apply(lambda x: x.lower()).values)])
    cat2 = ''.join([item+' ' for item in set(df['category_lvl_2'].apply(lambda x: x.lower()).values)])
    cat3 = ''.join([item+' ' for item in set(df['category_lvl_3'].apply(lambda x: x.lower()).values)])


    def title_category(row):
        titleword = row['title'].lower().split(' ')
        titlecat1 = row['category_lvl_1'].lower()
        titlecat2 = row['category_lvl_2'].lower()
        titlecat3 = row['category_lvl_3'].lower()
        titlecat1_num = len(set([w for w in titleword if cat1.count(w) >= 1]))
        titlecat2_num = len(set([w for w in titleword if cat2.count(w) >= 1]))
        titlecat3_num = len(set([w for w in titleword if cat3.count(w) >= 1]))
        titlecat1include_num = len(set([w for w in titleword if titlecat1.count(w) >= 1]))
        titlecat2include_num = len(set([w for w in titleword if titlecat2.count(w) >= 1]))
        titlecat3include_num = len(set([w for w in titleword if titlecat3.count(w) >= 1]))

        return '{}:{}:{}:{}:{}:{}'.format(titlecat1_num, titlecat2_num, titlecat3_num,
                                          titlecat1include_num, titlecat2include_num, titlecat3include_num)

    print("creating title category intersection num")

    df['title_cat'] = df.apply(title_category, axis=1, raw=True)
    df['title_cat1_num'] = df['title_cat'].apply(lambda x: float(x.split(':')[0]))
    df['title_cat2_num'] = df['title_cat'].apply(lambda x: float(x.split(':')[1]))
    df['title_cat3_num'] = df['title_cat'].apply(lambda x: float(x.split(':')[2]))
    df['title_cat1_include_num'] = df['title_cat'].apply(lambda x: float(x.split(':')[3]))
    df['title_cat2_include_num'] = df['title_cat'].apply(lambda x: float(x.split(':')[4]))
    df['title_cat3_include_num'] = df['title_cat'].apply(lambda x: float(x.split(':')[5]))

    df['title_cat1_diff'] = df['title_cat1_num'] - df['title_cat1_include_num']
    df['title_cat2_diff'] = df['title_cat2_num'] - df['title_cat2_include_num']
    df['title_cat3_diff'] = df['title_cat3_num'] - df['title_cat3_include_num']

    df = pd.concat([df, pd.get_dummies(pd.qcut(df['title_wordnum'], 5,
                                               labels=["wordnum_q1", "wordnum_q2", "wordnum_q3", 'wordnum_q4', 'wordnum_q5']))], axis=1)

    d = enchant.Dict("en_US")


    def longest_common_substring(s1, s2):
       m = [[0] * (1 + len(s2)) for i in xrange(1 + len(s1))]
       longest, x_longest = 0, 0
       for x in xrange(1, 1 + len(s1)):
           for y in xrange(1, 1 + len(s2)):
               if s1[x - 1] == s2[y - 1]:
                   m[x][y] = m[x - 1][y - 1] + 1
                   if m[x][y] > longest:
                       longest = m[x][y]
                       x_longest = x
               else:
                   m[x][y] = 0
       return s1[x_longest - longest: x_longest]


    df['title_type_check_num'] = df['title'].apply(lambda x: len([w for w in x.split() if not d.check(w)]))

    df['title_word_duplicate_num'] = df['title'].apply(lambda x: len(x.split(' ')) - len(set((x.lower()).split(' '))))
    df['title_word_duplicate_num2'] = df['title'].apply(lambda x: np.sum([x.lower().count(w) for w in (x.translate(None, "()<>{}").lower()).split(' ') if len(w) > 2 and x.lower().count(w) > 1]))

    df['title_word_duplicate_cat_num'] = df['title'].apply(lambda x: len(set([w for w in (x.translate(None, "()<>{}").lower()).split(' ') if len(w) > 1 and x.lower().count(w) > 1])))
    df['title_word_duplicate_nums'] = df['title'].apply(lambda x: np.sum([x.lower().count(w) for w in (x.translate(None, "()<>{}").lower()).split(' ') if len(w) > 1 and x.lower().count(w) > 1]))
    df['title_word_duplicate_nums1'] = df['title'].apply(lambda x: np.sum([x.lower().count(w) for w in (x.lower()).split(' ') if len(w) > 1 and x.lower().count(w) > 1]))

    df['title_word_duplicate_num_cleaned_1'] = df['cleaned_title'].apply(lambda x: len(x.split(' ')) - len(set((x.lower()).split(' '))))
    df['title_word_duplicate_num_cleaned'] = df['cleaned_title'].apply(lambda x: np.sum([x.count(w) for w in x.split(' ') if len(w)> 2 and x.count(w) > 1]))
    df['title_word_duplicate_cat_num_cleaned'] = df['cleaned_title'].apply(lambda x: len(set([w for w in x.split(' ') if len(w) > 1 and x.count(w) > 1])))

    df['title_word_duplicate_nums_std'] = df['title'].apply(lambda x: np.mean([pd.Series([i for i,m in enumerate(x.lower().split(' ')) if m==w]).dropna().sum() for w in x.lower().split(' ') if len(w) > 1 and x.lower().count(w) > 1]))
    df['title_word_duplicate_nums_std'] = df['title_word_duplicate_nums_std'].fillna(-1)

    df['title_word_lcs'] = df['title'].apply(lambda x: [longest_common_substring(x.split(' ')[i], x.split(' ')[j])
                                            for i in range(len(x.split(' '))) for j in range(len(x.split(' '))) if i!=j and len(longest_common_substring(x.split(' ')[i], x.split(' ')[j]))>2])

    df['title_word_lcs_num'] = df['title_word_lcs'].apply(lambda x: len(x)/2.0)
    df['title_word_lcs_cat_num'] = df['title_word_lcs'].apply(lambda x: len(set(x)))

    df['title_comma_num'] = df['title'].apply(lambda x: x.count(',')+x.count(';'))
    df['title_tag_num'] = df['title'].apply(lambda x: x.lower().count('('))

    df['title_word_duplicate_ratio'] = df['title_word_duplicate_num']/df['title_wordnum']

    df['title_charnum'] = df['title'].apply(lambda x: len(str(x).replace(' ', '')))
    df['title_digitnum'] = df['title'].apply(lambda x: len([w for w in list(x) if w.isdigit()]))

    df['title_wordcharlargenum'] = df['title'].apply(lambda x: len([word for word in x.split(' ') if len(word) > 10]))
    df['title_avgwordlen'] = df['title_charnum']/df['title_wordnum']
    df['title_nonalphanum'] = df['title'].apply(lambda x: len([w for w in set(x.split(' ')) if not w.isalpha() or w.isdigit()]))

    df['title_uppernum'] = df['title'].apply(lambda x: len([w for w in set(x.split(' ')) if w.isupper()]))
    df['tittle_upper_word'] = df['title'].apply(lambda x: len([w for w in x.split(' ') if len([c for c in list(w) if c.isupper()]) == len(list(w))]))

    df['tittle_small_upper_word'] = df['title'].apply(lambda x: len([w for w in x.split(' ') if len(list(w))<=4 and len([c for c in list(w) if c.isupper()]) == len(list(w))]))
    df['tittle_small_upper_word1'] = df['title'].apply(lambda x: len([w for w in x.split(' ') if 1<len(list(w))<=2 and len([c for c in list(w) if c.isupper()]) == len(list(w))]))

    df['title_C_upperratio'] = df['title'].apply(lambda x: float(len([w for w in list(x) if w.isupper()]))/len(x))
    df['title_C_upperwordratio'] = df['title'].apply(lambda x: float(len([w for w in list(x) if w.isupper()]))/len(x.split(' ')))

    df['title_D_ratio'] = df['title'].apply(lambda x: float(len([w for w in list(x) if w.isdigit()]))/len(x))
    df['title_D_wordratio'] = df['title'].apply(lambda x: float(len([w for w in list(x) if w.isdigit()]))/len(x.split(' ')))

    transition_char_list = list('\\/\'-_+|~&%$#*@<>?;:[]')

    df['title_special_num'] = df['title'].apply(lambda x: len([w for w in x.split(' ')
                              if len(list(w)) > 0 and len([c for c in list(w) if c.isdigit() or c.isupper() or c in transition_char_list])]))

    df['title_stopsnum'] = df['title'].apply(lambda x: len(set(x.split(' ')).intersection(engstops)))
    df['title_eng_word'] = df['title'].apply(lambda x: (''.join([''.join(w.lower()+' ') for w in set(x.split(' ')) if len(wn.synsets(unicode(w, 'utf-8'))) != 0 and w not in engstops])).strip())


    def title_word_simi(row):
        title_words = list(row['title_eng_word'].split(' '))
        length = len(title_words)

        simis_x, simis_y = [], []
        cnt_x, cnt_y = 0, 0
        if length > 1:
            for i in range(length):
                for j in range(length):
                    if i != j:
                        simix = wn.synset(wn.synsets(unicode(title_words[i], 'utf-8'))[0].name()).wup_similarity(
                            wn.synset(wn.synsets(unicode(title_words[j], 'utf-8'))[0].name()))
                        simiy = wn.synset(wn.synsets(unicode(title_words[i], 'utf-8'))[0].name()).path_similarity(
                            wn.synset(wn.synsets(unicode(title_words[j], 'utf-8'))[0].name()))
                        simis_x.append(simix)
                        simis_y.append(simiy)
                        cnt_x += 1 if simix != None and simix > 0.2 else 0.0
                        cnt_y += 1 if simiy != None and simiy > 0.2 else 0.0
        else:
            pass
        simis_x, simis_y = pd.Series(simis_x).dropna(), pd.Series(simis_y).dropna()
        return "{}:{}:{}:{}:{}:{}:{}".format(length, cnt_x, cnt_y, simis_x.mean(), simis_x.var(), simis_y.mean(), simis_y.var())

    print("creating title english word similarity")

    df['title_eng_word'] = df.apply(title_word_simi, axis=1, raw=True)
    df['title_eng_word_length'] = df['title_eng_word'].apply(lambda x: float(x.split(':')[0]))
    df['title_eng_word_simi_num'] = df['title_eng_word'].apply(lambda x: float(x.split(':')[1]))
    df['title_eng_word_simi_path_num'] = df['title_eng_word'].apply(lambda x: float(x.split(':')[2]))

    df['title_nonengnum'] = df['title'].apply(lambda x: len([w for w in set((x.translate(None, "(){}<>+-\"&")).split(' ')) if len(wn.synsets(unicode(w, 'utf-8'))) == 0]))
    df['title_nonengratio'] = df['title_nonalphanum'] / df['title_wordnum']
    df['title_wordsynsetdepthsum'] = \
    df['title'].apply(lambda x: np.sum([wn.synset(wn.synsets(unicode(w, 'utf-8'))[0].name()).min_depth() for w in set(x.split(' ')) if len(wn.synsets(unicode(w, 'utf-8'))) != 0]))

    df['title_meaningword_ratio'] = 1 - df['title_nonengnum']/df['title_wordnum']
    df['title_wordlemmassnum'] = df['title'].apply(lambda x: np.sum([len(wn.lemmas(unicode(w, 'utf-8'))) for w in set(x.split(' '))
                                                                     if len(wn.synsets(unicode(w, 'utf-8'))) != 0]))

    def my_soup(x):
        soup = BeautifulSoup(x, "html.parser")
        li = soup.find_all(re.compile("^l"))
        return soup.get_text()+";" if len(li) == 0 else "".join(["".join(item.find_all(text=True))+";" for item in li])


    def title_description(row):

        titlenum = len(row['title'].lower().split(' '))
        descriptionnum = len(row['description_content'].lower().split(' '))
        internum = 0
        for w in row['title'].lower().split(' '):
            if w in row['description_content'].lower().split(' '):
                internum += 1
        return '{}:{}:{}'.format(internum, float(internum)/titlenum, float(internum)/descriptionnum)

    print("parsing description in text")

    df['description_content'] = df['short_description'].apply(lambda x: my_soup(x))

    df['description_C_upperratio'] = df['description_content'].apply(lambda x: float(len([w for w in list(x) if w.isupper()]))/len(x))
    df['description_C_upperwordratio'] = df['description_content'].apply(lambda x: float(len([w for w in list(x) if w.isupper()]))/len(x.split(' ')))
    df['description_digitnum'] = df['description_content'].apply(lambda x: len([w for w in list(x) if w.isdigit()]))
    df['title_descrition_digit_diff'] = df['title_digitnum'] - df['description_digitnum']

    df['description_nonalphanum'] = df['description_content'].apply(lambda x: len([w for w in set(x.split(' ')) if not w.isalpha() or w.isdigit()]))
    df['title_descrition'] = df.apply(title_description, axis=1, raw=True)

    df['title_descrition_inter_num'] = df['title_descrition'].apply(lambda x: float(x.split(':')[0]))
    df['title_descrition_inter_ratio1'] = df['title_descrition'].apply(lambda x: float(x.split(':')[1]))
    df['title_descrition_inter_ratio2'] = df['title_descrition'].apply(lambda x: float(x.split(':')[2]))

    df['description_content_word_num'] = df['description_content'].apply(lambda x: len(x.split(' ')))
    df['description_content_duplicate_num'] = df['description_content'].apply(lambda x: len(x.split(' '))-len(set((x.lower()).split(' '))))
    df['description_upper_word'] = df['description_content'].apply(lambda x: len([w for w in x.split(' ') if len([c for c in list(w) if c.isupper()]) == len(list(w))]))

    df['description_li_num'] = df['description_content'].apply(lambda x: len(x.split(';')))
    df['description_special_num'] = df['description_content'].apply(lambda x: len([w for w in x.encode('ascii', 'ignore').split(' ')
                               if len(list(w)) > 0 and len([c for c in list(w) if c.isdigit() or c.isupper() or c in transition_char_list])]))

    frequency_count = lambda x: x.count()

    df['category_lvl_1_id'] = pd.factorize(df['category_lvl_1'].astype('category'))[0]
    df['category_lvl_2_id'] = pd.factorize(df['category_lvl_2'].astype('category'))[0]
    df['category_lvl_3_id'] = pd.factorize(df['category_lvl_3'].astype('category'))[0]
    df['category_lvl_1_frequency'] = df.groupby('category_lvl_1_id')['category_lvl_1_id'].transform(frequency_count)
    df['category_lvl_2_frequency'] = df.groupby('category_lvl_2_id')['category_lvl_2_id'].transform(frequency_count)
    df['category_lvl_3_frequency'] = df.groupby('category_lvl_3_id')['category_lvl_3_id'].transform(frequency_count)
    df['category_lvl_2_ratio1'] = df['category_lvl_2_frequency']/df['category_lvl_1_frequency']
    df['category_lvl_3_ratio2'] = df['category_lvl_3_frequency']/df['category_lvl_2_frequency']
    df['category_lvl_1_ratio3'] = df['category_lvl_3_frequency']/df['category_lvl_1_frequency']

    df.loc[df['country'] == 'my', 'price'] = df.loc[df['country'] == 'my', 'price']/4.33
    df.loc[df['country'] == 'sg', 'price'] = df.loc[df['country'] == 'sg', 'price']/1.40
    df.loc[df['country'] == 'ph', 'price'] = df.loc[df['country'] == 'ph', 'price']/49.88

    avg_contrast = lambda x: x / x.mean() if len(x) > 0 else -1

    df['title_cat2_duplicate_ratio_avg_contrast'] = df.groupby('category_lvl_2_id')['title_word_duplicate_ratio'].transform(avg_contrast)

    df['title_cat_wordlemmasssum_mean_contrast'] = \
        df.groupby(['category_lvl_1_id', 'category_lvl_2_id', 'category_lvl_3_id'])['title_wordlemmassnum'].transform(avg_contrast)
    df['title_cat2_wordlemmasssum_mean_contrast'] = df.groupby('category_lvl_2_id')['title_wordlemmassnum'].transform(avg_contrast)
    df['title_cat3_wordlemmasssum_mean_contrast'] = df.groupby('category_lvl_3_id')['title_wordlemmassnum'].transform(avg_contrast)

    df['title_cat_wordnum_mean_contrast'] = \
        df.groupby(['category_lvl_1_id', 'category_lvl_2_id', 'category_lvl_3_id'])['title_wordnum'].transform(avg_contrast)
    df['title_cat2_wordnum_mean_contrast'] = df.groupby('category_lvl_2_id')['title_wordnum'].transform(avg_contrast)
    df['title_cat3_wordnum_mean_contrast'] = df.groupby('category_lvl_3_id')['title_wordnum'].transform(avg_contrast)

    df['price_country_mean_contrast'] = df.groupby('country_id')['price'].transform(avg_contrast)
    df['price_cat_mean_contrast'] = \
        df.groupby(['category_lvl_1_id', 'category_lvl_2_id', 'category_lvl_3_id'])['price'].transform(avg_contrast)
    df['price_cat2_mean_contrast'] = df.groupby('category_lvl_2_id')['price'].transform(avg_contrast)
    df['price_cat3_mean_contrast'] = df.groupby('category_lvl_3_id')['price'].transform(avg_contrast)

    df['product_type_id'] = pd.factorize(df['product_type'].astype('category'))[0]
    df = pd.concat([df, pd.get_dummies(df['product_type'])], axis=1)  # one-hot coding for product type

    print("creating bag of words features")
    n_features_title = 4000
    n_features_title_clarity = 2800

    tfidf_vectorizer_title_clarity = CountVectorizer(max_df=1.0
                                                     , min_df=1
                                                     , ngram_range=(1, 1)
                                                     , analyzer='word'
                                                     , max_features=n_features_title_clarity)

    tfidf_vectorizer_title_conciseness = CountVectorizer(max_df=1.0
                                                         , min_df=1
                                                         , ngram_range=(1, 12)
                                                         , analyzer='char'
                                                         , max_features=n_features_title)

    sparse_tfidf_title_clarity = tfidf_vectorizer_title_clarity.fit_transform(df['title'])

    sparse_tfidf_title_conciseness = tfidf_vectorizer_title_conciseness.fit_transform(df['title'])

    print("creating LDA features")
    lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=False)
    lda_fit_conciseness = lda.fit(sparse_tfidf_title_clarity.todense()[np.where(df['conciseness'] == 1)[0]].reshape(-1, n_features_title_clarity), df.loc[df['conciseness'] == 1, 'category_lvl_3_id'])

    lda_proba_conciseness = lda_fit_conciseness.predict_proba(sparse_tfidf_title_clarity.todense())
    lda_pred_conciseness = lda_fit_conciseness.predict(sparse_tfidf_title_clarity.todense())

    cat3_condition_conciseness = np.zeros_like(lda_proba_conciseness)
    for i in range(lda_pred_conciseness.shape[0]):
        cat3_condition_conciseness[i, lda_pred_conciseness[i]] = 1

    df['lda_equal_conciseness'] = (pd.Series(lda_pred_conciseness) == df['category_lvl_3_id']).astype(int)
    df['lda_true_prob_conciseness'] = np.extract(cat3_condition_conciseness, lda_proba_conciseness)

    print("feature preparing finished, saving them to disk")

    feature_dir = os.path.join(datadir, 'features')
    if not os.path.exists(feature_dir):
        os.makedirs(feature_dir)

    df.to_csv(os.path.join(feature_dir, 'df_feature.csv'), encoding='utf-8')

    scipy.sparse.save_npz(os.path.join(feature_dir, 'sparse_clarity.npz'), sparse_tfidf_title_clarity)
    scipy.sparse.save_npz(os.path.join(feature_dir, 'sparse_conciseness.npz'), sparse_tfidf_title_conciseness)

