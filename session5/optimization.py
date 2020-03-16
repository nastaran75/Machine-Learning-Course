import numpy as np
import matplotlib.pyplot as plt

# Train a Linear Classifier
N= 100
K = 5
X = np.zeros((N*K,2))
y = np.zeros(N*K,dtype='uint8')
points = np.arange(N)
theta = np.linspace(0,360,N)
for i in range(0,K):
    X[np.arange(i*N,i*N+N)] = np.c_[(i+1)*np.sin(theta), (i+1)*np.cos(theta)]
    y[np.arange(i*N,i*N+N)] = i
    
# y = np.random.randint(0,K,size=y.shape)
plt.scatter(X[:, 0], X[:, 1],c=y)
plt.show()



def softmax(W,X,y,b):

    # evaluate class scores, [N x K]
    scores = np.dot(X, W) + b
    scores = scores - np.max(scores)


    # compute the class probabilities
    exp_scores = np.exp(scores)
    assert exp_scores.all()>0

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]


    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_logprobs) / num_examples

    reg_loss = reg * np.sum(W * W)  # remember to multiply by 0.5

    return probs,reg_loss,data_loss


# initialize parameters randomly
W = np.random.randn(X.shape[1],K) * 0.01
b = np.zeros((1, K))
reg = 1e-3
num_examples = X.shape[0]


bestloss = float("inf") # Python assigns the highest possible float value
for num in xrange(1000):
  W = np.random.randn(X.shape[1],K) * 0.01 # generate random parameters
  probs,reg_loss,data_loss = softmax(W,X,y,b) # get the loss over the entire training set
  loss = reg_loss + data_loss
  if loss < bestloss: # keep track of the best solution
    bestloss = loss
    bestW = W
  # print 'in attempt %d the loss was %f, best %f' % (num, loss, bestloss)

scores = X.dot(bestW)+b # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 1)
# and calculate accuracy (fraction of predictions that are correct)
print np.mean(Yte_predict == y)




W = np.random.randn(X.shape[1], K) * 0.01 # generate random starting W
bestloss = float("inf")
for i in xrange(1000):
  step_size = 0.0001
  Wtry = W + np.random.randn(X.shape[1], K) * step_size
  probs,reg_loss,data_loss = softmax(Wtry,X,y,b)
  loss = reg_loss + data_loss
  if loss < bestloss:
    W = Wtry
    bestloss = loss
  # print 'iter %d loss is %f' % (i, bestloss)

scores = X.dot(W)+b # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 1)
# and calculate accuracy (fraction of predictions that are correct)
print np.mean(Yte_predict == y)






