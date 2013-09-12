from sys import argv
import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm

f="rfSubmission.csv" # file with predictions
csv_reader = csv.reader(open(f, "r"),delimiter=",")
csv_reader.next() # skip the header
data = []
for row in csv_reader:
	data.append(np.array(np.int64(row[1])))
np.save('predictions.npy', data)

	# to plot the train sample 
# image = np.load('train_small.npy')
# prediction = np.load('labels_small.npy')

	# to plot the test sample
image = np.load('test_small.npy')
prediction = np.load('predictions.npy')

img = np.empty((28, 28),dtype=int)


# Show images and predictions from lines argv[1] to argv[2]
if len(argv) == 3:
	for k in range(int(argv[1]),int(argv[2])):
		for i in range(0,28):
			for j in range(0,28):
				img[i,j] = image[k, i*28 + j]
		plt.title("Classification: %d" % (prediction[k]), fontsize=40)
		fig = plt.imshow(img, cmap = cm.Greys_r)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.show()
else:
	r1 = int(raw_input("begin: "))
	r2 = int(raw_input("end: "))
	for k in range(r1, r2):
		for i in range(0,28):
			for j in range(0,28):
				img[i,j] = image[k, i*28 + j]
		plt.title("Classification: %d" % (prediction[k]), fontsize=40)
		fig = plt.imshow(img, cmap = cm.Greys_r)
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.show()
