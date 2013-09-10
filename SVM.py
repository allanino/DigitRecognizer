import scipy
import numpy as np
import operator
import csv
from sklearn.svm import SVC

# loading csv data into numpy array
def read_data(f, header=True, test=False, rows=0):
    data = []
    labels = []

    csv_reader = csv.reader(open(f, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index = index + 1
        if rows > 0 & index > rows:
            break
        if header and index == 1:
            continue

        if not test:
            labels.append(int(row[0]))
            row = row[1:]

        data.append(np.array(np.int64(row)))
    return (data, labels)

def predictSVC(train, labels, test):
    print 'start SVC'
    clf = SVC(probability=True)
    clf.fit(train, labels)
    print 'predicting'
    svc_predictions = clf.predict(test)
    svc_probs = clf.predict_proba(test)
    print svc_probs
    svc_bestProbs = svc_probs.max(axis=1)
    print 'svc done!'
    return svc_predictions, svc_bestProbs

class PredScore:
    def __init__(self,prediction,score):
        self.prediction = prediction
        self.score = score
    prediction = -1
    score = 0

if __name__ == '__main__':
    print 'read data!'
    #only use the below for initial creation of npy files
    train, labels = read_data("train.csv", rows=100)
    np.save('train_test.npy', train)
    np.save('labels_test.npy', labels)

    train = np.load('train_test.npy')
    labels = np.load('labels_test.npy')

    print 'done reading train'
    
    #only use the below for the initial creation of npy files.
    test, tmpl = read_data("test.csv", test=True, rows=100)
    np.save('test_test.npy', test)
    np.save('tmpl_test.npy', tmpl)

    test = np.load('test_test.npy')
    tmpl = np.load('tmpl_test.npy')

    print 'done reading test!'

    svcPredictions, svcScore = predictSVC(train, labels, test)
    
    np.savetxt('svcPredictions.csv', svcPredictions, delimiter=',',fmt='%i')
    print 'done!!!'