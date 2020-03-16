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


learning_rate = 1e-0

for i in range(100):
	prob,reg_loss,data_loss = softmax(W,X,y,b)

	dscore = prob
	dscore[range(num_examples),y] -= 1
	dscore/=num_examples

	dW = np.dot(X.T,dscore)
	db = np.sum(dscore,axis=0)

	dW += 2*reg*W

	W -= learning_rate*dW
	b -= learning_rate*db

scores = X.dot(W)+b # 10 x 10000, the class scores for all test examples
# find the index with max score in each column (the predicted class)
Yte_predict = np.argmax(scores, axis = 1)
# and calculate accuracy (fraction of predictions that are correct)
print np.mean(Yte_predict == y)




h = 100
W1 = np.random.randn(X.shape[1],h) 
b1 = np.zeros((1,h))
W2 = np.random.randn(h,K)
b2 = np.zeros((1,K))

for i in range(1000):
	layer1 = np.dot(X,W1) + b1
	ReLU = np.maximum(0,layer1)

	prob,reg_loss,data_loss = softmax(W2,ReLU,y,b2)

	dscore = prob
	dscore[range(num_examples),y] -= 1
	dscore/=num_examples

	dW2 = np.dot(ReLU.T,dscore)
	dW2 += 2*reg*W2

	db2 = np.sum(dscore,axis=0)

	dReLU = np.dot(dscore,W2.T)
	dReLU[ReLU==0] = 0

	dW1 = np.dot(X.T,dReLU)
	dW1 += 2*reg*W1

	db1 = np.sum(dReLU,axis=0)


	W2 -= learning_rate*dW2
	W1 -= learning_rate*dW1
	b1 -= learning_rate*db1
	b2 -= learning_rate*db2

layer1 = np.dot(X,W1) + b1
ReLU = np.maximum(0,layer1)
score = np.dot(ReLU,W2)+b2
prediction = np.argmax(score,axis=1)
print np.mean(prediction==y)










