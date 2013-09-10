from sys import argv
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

	# only to generate submission.npy
f="knnPredictions.csv"
csv_reader = csv.reader(open(f, "r"))
data = []
for row in csv_reader:
	data.append(np.array(np.int64(row)))
np.save('predictions.npy', data)

	# to plot the train sample
# image = np.load('train_small.npy')
# prediction = np.load('labels_small.npy')

	# to plot the test sample
image = np.load('test_small.npy')
prediction = np.load('predictions.npy')

img = np.eye(28,dtype=int)

for k in range(int(argv[1]),int(argv[2])):
	for i in range(0,28):
		for j in range(0,28):
			img[i,j] = image[k, i*28 + j]
	print prediction[k]
	plt.imshow(img, cmap = cm.Greys_r)
	plt.show()