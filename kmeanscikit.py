from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import numpy as np

#
# def get_data(filename):
#     dataMat = []
#     fr = open(filename)
#     for line in fr.readlines():
#         curLine = line.strip().split('\t')
#         fltLine = list(map(float, curLine))
#         dataMat.append(fltLine)
#     return dataMat
#

def get_data2():
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    # datMat = mat(datList)
    return datList


# x = get_data("testSet.txt")

# store entire dataset list in a variable
x = get_data2()
# convert to numpy array
x = np.array(x)
estimator = KMeans(6)
estimator.fit(x)

centroids = estimator.cluster_centers_
# y_means = estimator.predict(x)
labels = estimator.labels_

colors = ['k.','r.', 'b.', 'c.', 'm.', 'y.']


print('the boy is very studpid {}'.strip().format('but not my shade'))

for i in range(len(x)):
    print("coordinates: ", x[i], " label: {} ", labels[i])
    plt.plot(x[i][0], x[i][1],colors[labels[i]], markersize=10)

plt.scatter(centroids[:,0], centroids[:,1], marker="*", s=250, linewidths=10)
# img = plt.imread("Portland.png")
# plt.imshow(img)
plt.show()


