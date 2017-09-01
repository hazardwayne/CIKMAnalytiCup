import numpy as np

ensembledir = '../data/ensemble/'

xgb_clarity = np.loadtxt(ensembledir+'clarity_test_xgb.predict')
lgb_clarity = np.loadtxt(ensembledir+'clarity_test_lgb.predict')
# lstm_clarity = np.loadtxt(ensembledir+'clarity_test_lstm.predict')
mlp_clarity = np.loadtxt(ensembledir+'clarity_test_mlp_3211.predict')

np.savetxt(ensembledir+'clarity_test_1.predict',
           0.6*xgb_clarity+0.2*lgb_clarity+0.2*mlp_clarity, fmt='%.6f')

xgb_conciseness = np.loadtxt(ensembledir+'conciseness_test_xgb.predict')
lgb_conciseness = np.loadtxt(ensembledir+'conciseness_test_lgb.predict')
lstm_conciseness = np.loadtxt(ensembledir+'conciseness_test_lstm.predict')
mlp_conciseness = np.loadtxt(ensembledir+'conciseness_test_mlp_3211.predict')

np.savetxt(ensembledir+'conciseness_test_1.predict',
           0.35*xgb_conciseness+0.25*lgb_conciseness+0.3*lstm_conciseness+0.1*mlp_conciseness, fmt='%.6f')
