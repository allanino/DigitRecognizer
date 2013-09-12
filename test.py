from sklearn.externals import joblib
import numpy as np
from scipy import misc

num = misc.imread('n.png').flatten()

# test = np.load('test_small.npy')
# tmpl = np.load('tmpl_small.npy')

rf = joblib.load('rfClassifier.pkl')



rf_predictions = rf.predict(num)

print rf_predictions[0]