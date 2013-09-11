import csv

f="submission.csv" # file with predictions
submission = open("submission.csv", 'r+')

csv_reader = csv.reader(open(f, "r"))
data = "ImageId,Label\n"
i = 1
for row in csv_reader:
	data = "%s%d,%d\n" % (data, i, int(row[0]))
	i += 1

submission.write(data)