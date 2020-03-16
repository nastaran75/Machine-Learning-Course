import numpy as np
import matplotlib.pyplot as plt

# Train a Linear Classifier
N = 100
K = 5
X = np.zeros((N*K,2))
y = np.zeros(N*K,dtype='uint8')
points = np.arange(N)
theta = np.linspace(0,360,N)
for i in range(0,K):
    X[np.arange(i*N,i*N+N)] = np.c_[(i+1)*np.sin(theta), (i+1)*np.cos(theta)]
    # y[np.arange(i*N,i*N+N)] = i
    
y = np.random.randint(0,K,size=y.shape)
plt.scatter(X[:, 0], X[:, 1],c=y)
plt.show()





def softmax(W,X,y,b):

    # evaluate class scores, [N x K]
    scores = np.dot(X, W) + b
    scores = scores - np.max(scores)


    # compute the class probabilities
    exp_scores = np.exp(scores)
 

    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


    # compute the loss: average cross-entropy loss and regularization
    correct_logprobs = -np.log(probs[range(num_examples), y])

    data_loss = np.sum(correct_logprobs) / num_examples

    reg_loss = reg * np.sum(W * W)  # remember to multiply by 0.5

    return probs,reg_loss,data_loss

def svm(W,X,y,b):
	delta = 1.0
	scores = np.dot(X, W) + b
	# compute the margins for all classes in one vector operation
	margins = np.maximum(0, scores - scores[y] + delta)
	# on y-th position scores[y] - scores[y] canceled and gave delta. We want
	# to ignore the y-th position and only consider margin on max wrong class
	margins[y] = 0
	data_loss = np.sum(margins) / num_examples

	reg_loss = reg * np.sum(W * W)  # remember to multiply by 0.5

	return reg_loss,data_loss


# print softmax(W,X,y,b)
# print svm(W,X,y,b)

