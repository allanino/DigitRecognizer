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
    clf = SVC()
    clf.fit(train, labels)
    print 'predicting'
    svc_predictions = clf.predict(test)
    # print 'calculation probability'
    # svc_probs = clf.predict_proba(test)
    # debugg
    # np.savetxt('svcProbs.csv', svc_probs, delimiter=',',fmt='%i')
    ####
    # svc_bestProbs = svc_probs.max(axis=1)
    print 'svc done!'
    return svc_predictions

if __name__ == '__main__':
    print 'read data!'
    #only use the below for initial creation of npy files
    # train, labels = read_data("train.csv", rows=1000)
    # np.save('train_small.npy', train)
    # np.save('labels_small.npy', labels)

    train = np.load('train_small.npy')
    labels = np.load('labels_small.npy')

    print 'done reading train'
    
    #only use the below for the initial creation of npy files.
    # test, tmpl = read_data("test.csv", test=True, rows=1000)
    # np.save('test_small.npy', test)
    # np.save('tmpl_small.npy', tmpl)

    test = np.load('test_small.npy')
    tmpl = np.load('tmpl_small.npy')

    print 'done reading test!'

    svcPredictions  = predictSVC(train, labels, test)
    
    np.savetxt('svcPredictions.csv', svcPredictions, delimiter=',',fmt='%i')
    print 'done!!!'