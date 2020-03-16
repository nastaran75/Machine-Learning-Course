import numpy as np
import matplotlib.pyplot as plt

class NearestNeighbor():
    def train(self,X,Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self,Xte):
        num_test = Xte.shape[0]
        Y_pred = np.zeros(num_test)

        for i,test_image in enumerate(Xte):
            distances = np.sum(np.abs(self.Xtr-test_image),axis = 1)
            min_index = np.argmin(distances)
            Y_pred[i] = self.Ytr[min_index]

        return Y_pred

    def predictKNN(self, X, k):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype = self.Ytr.dtype)

        # loop over all test rows
        for i in xrange(num_test):
          # find the nearest training image to the i'th test image
          # using the L1 distance (sum of absolute value differences)
          distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
          min_indexes = np.argsort(distances)[:k]
          # print self.ytr[min_indexes[0]]
          min_index = np.argmax(np.bincount(self.Ytr[min_indexes])) # get the index with smallest distance
          # print min_index
          # print self.ytr[min_index]
          Ypred[i] = min_index # predict the label of the nearest example

        return Ypred


#for python3
# def unpickle(file):
#     import pickle
#     with open(file, 'rb') as fo:
#         dict = pickle.load(fo, encoding='bytes')
#     return dict


def load_data(filename):
    datadict = np.load(filename,allow_pickle=True) #for python2 only
    # datadict = unpickle(filename)  #for python3
    X = datadict['data']   # or datadict[b'data'] if you are using python3
    X = X.astype('float')
    Y = datadict['labels'] # datadict[b'labels'] if you are using python3
    Y = np.array(Y)
    return X,Y


train_images = []
train_labels = []

for i in range(1,2):
    X, Y = load_data('cifar-10-batches-py/data_batch_' + str(i))
    train_images.append(X)
    train_labels.append(Y)

Xtr = np.concatenate(train_images)
Ytr = np.concatenate(train_labels)

Xte,Yte = load_data('cifar-10-batches-py/test_batch')
Xte = Xte[:100]   # take a small batch of test images if you want a quick result
Yte = Yte[:100]


model = NearestNeighbor()
model.train(Xtr,Ytr)
Y_predict = model.predict(Xte)
print np.mean(Y_predict==Yte)   # you should get around 32% accuracy

Xval = Xtr[:100, :] # take first 1000 for validation
Yval = Ytr[:100]
Xtr = Xtr[100:, :] # keep last 49,000 for train
Ytr = Ytr[100:]

validation_accuracies = []
for k in [1, 3, 5, 7, 10, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = NearestNeighbor()
  nn.train(Xtr, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predictKNN(Xval, k = k)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))
print validation_accuracies


# using sklearn
from sklearn.neighbors import KNeighborsClassifier
validation_accuracies = []
for k in [1, 3, 5,7, 10, 20, 50, 100]:

  # use a particular value of k and evaluation on validation data
  nn = KNeighborsClassifier(n_neighbors=k,n_jobs=7)
  nn.fit(Xtr, Ytr)
  # here we assume a modified NearestNeighbor class that can take a k as input
  Yval_predict = nn.predict(Xval)
  acc = np.mean(Yval_predict == Yval)
  print 'accuracy: %f' % (acc,)

  # keep track of what works on the validation set
  validation_accuracies.append((k, acc))




#for displaying a single image
# A = np.load('cifar-10-batches-py/data_batch_1',allow_pickle= True)  #for python2 only
# A = unpickle('cifar-10-batches-py/data_batch_1') # for python 2 or 3
# print A.keys()
# images = A['data']
# print images.shape
# img = images[100]
#
# plt.imshow(img.reshape(3,32,32).transpose(1,2,0))
# plt.show()


