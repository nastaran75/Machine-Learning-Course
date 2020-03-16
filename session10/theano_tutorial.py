
import numpy
import theano
import theano.tensor as T
from theano import function
import numpy as np
import matplotlib.pyplot as plt

# define two symbols (Variables) representing the quantities that you want to add
# In Theano, all symbols must be typed. 
# T.dscalar is the type we assign to 0-dimensional arrays (scalar) of doubles (d). It is a Theano Type.
# The owner field of both x and y point to None because they are not the result of another computation
# in Theano, irow and dmatrix both use numpy.ndarray as the underlying type for doing computations and storing data,
# mabahese teory http://cs231n.github.io/

# type
#     a Type defining the kind of value this Variable can hold in computation.
# owner
#     this is either None or an Apply node of which the Variable is an output.
# index
#     the integer such that owner.outputs[index] is r (ignored if owner is None)
# name
#     a string to use in pretty-printing and debugging. 



    # byte: bscalar, bvector, bmatrix, brow, bcol, btensor3, btensor4, btensor5, btensor6, btensor7
    # 16-bit integers: wscalar, wvector, wmatrix, wrow, wcol, wtensor3, wtensor4, wtensor5, wtensor6, wtensor7
    # 32-bit integers: iscalar, ivector, imatrix, irow, icol, itensor3, itensor4, itensor5, itensor6, itensor7
    # 64-bit integers: lscalar, lvector, lmatrix, lrow, lcol, ltensor3, ltensor4, ltensor5, ltensor6, ltensor7
    # float: fscalar, fvector, fmatrix, frow, fcol, ftensor3, ftensor4, ftensor5, ftensor6, ftensor7
    # double: dscalar, dvector, dmatrix, drow, dcol, dtensor3, dtensor4, dtensor5, dtensor6, dtensor7
    # complex: cscalar, cvector, cmatrix, crow, ccol, ctensor3, ctensor4, ctensor5, ctensor6, ctensor7




x = T.dscalar('x')   
y = T.dscalar()
z = x + y
f = function([x, y], z)

print f(2,3)

theano.printing.pydotprint(f, outfile="f.png", var_with_name_simple=True)  


#using eval
print z.eval({x:2,y:3})


#alternative way for defining functions
def sum(x,y):
	return x+y

z = sum(x,y)
print z.eval({x:2,y:3})

# The owner field of both x and y point to None because they are not the result of another computation
# An Apply node is typically an instance of the Apply class.
# it represents the application of an Op on one or more inputs, where each input is a Variable
print z.owner.op.name
print len(z.owner.inputs)
print z.owner.inputs[0]
print z.owner.inputs[1]


#ivector

x = T.ivector()
y = -x
f = function([x],y)
print f([1,2,3])



#exercise: compute this expression: a ** 2 + b ** 2 + 2 * a * b where a and b are vectors

a = T.vector()
b = T.vector()
out = a**2 + b**2 + 2*a*b
print(out.eval({a:[1,2],b:[3,4]}))

#exercise: implement the logistic function

# You want to compute the function elementwise on matrices of doubles,
 # which means that you want to apply this function to each individual element of the matrix.

mat = T.dmatrix('mat')
s = 1 / (1 + T.exp(-mat))
logistic = theano.function([mat], s)

print(logistic([[0, 1], [-1, -2]]))


#Ex
#Create an expression for the logistic function $s(x) = \frac{1}{1+exp(-x)}$.
 # Plot the function and its derivative, and verify that $\frac{ds}{dx} = s(x)(1-s(x))$.

x = T.vector()
s = 1/(1+T.exp(-x))
ds = T.grad(T.sum(s), x) # Need sum to make s scalar



x0 = np.arange(-3, 3, 0.01).astype('float32')
plt.plot(x0, s.eval({x:x0}))
plt.plot(x0, ds.eval({x:x0}))

np.allclose(ds.eval({x:x0}), s.eval({x:x0}) * (1-s.eval({x:x0})))

#Ex
#Fibonacci sequence
a = theano.shared(1)
b = theano.shared(1)
f = a + b
updates = {a: b, b: f}
next_term = theano.function([], f, updates=updates)

print [next_term() for _ in range(3, 10)]



# Computing More than one Thing at the Same Time
m1, m2 = T.dmatrices('m1', 'm2')   #dmatrices produces as many outputs as names that you provide. It is a shortcut for allocating symbolic variables that we will often use in the tutorials.
diff = m1 - m2
abs_diff = abs(diff)
diff_squared = diff**2
f = theano.function([m1, m2], [diff, abs_diff, diff_squared])
print f([[1,2],[3,4]],[[1,2],[2,3]])


# Setting a Default Value for an Argument
from theano import In
m3, m4 = T.dscalars('m3', 'm4')
z = m3 + m4
f = function([m3, In(m4, value=1)], z)
print f(33)
print f(33, 2)

#think: agar chan ta moteghayer dashtebashim k difault value beheshum dadim va mikham hala dovomio khodemun meghdar bdim chi mishe?
# bayad azun esmi k moghe tarif dadim estefade konim ya un esmo override konim In(w, value=2, name='w_by_name')
x, y, w = T.dscalars('x', 'y', 'w')
z = (x + y) * w
f = function([x, In(y, value=1), In(w, value=2, name='w_by_name')], z)
f(33)
f(33, 2)
f(33, 0, 1)
f(33, w_by_name=1)
f(33, w_by_name=1, y=0)

# Exercise: tensors


A = T.matrix('A')
x = T.vector('x')
b = T.vector('b')
y = T.dot(A, x) + b
# Note that squaring a matrix is element-wise
z = T.sum(A**2)
# theano.function can compute multiple things at a time
# You can also set default parameter values
# We'll cover theano.config.floatX later
b_default = np.array([0, 0], dtype=theano.config.floatX)
linear_mix = theano.function([A, x, theano.In(b, value=b_default)], [y, z])
# Supplying values for A, x, and b
print(linear_mix(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=theano.config.floatX), #A
                 np.array([1, 2, 3], dtype=theano.config.floatX), #x
                 np.array([4, 5], dtype=theano.config.floatX))) #b
# Using the default value for b
print(linear_mix(np.array([[1, 2, 3],
                           [4, 5, 6]], dtype=theano.config.floatX), #A
                 np.array([1, 2, 3], dtype=theano.config.floatX))) #x





#shared variable
shared_var= theano.shared(1)
shared_var.set_value(5)
print shared_var.get_value()
print(shared_var.type())

shared_var = theano.shared(np.array([[1, 2], [3, 4]], dtype=theano.config.floatX))
# The type of the shared variable is deduced from its initialization
print(shared_var.type())



# We can set the value of a shared variable using set_value
shared_var.set_value(np.array([[3, 4], [2, 1]], dtype=theano.config.floatX))
# ..and get it using get_value
print(shared_var.get_value())



shared_squared = shared_var**2
# The first argument of theano.function (inputs) tells Theano what the arguments to the compiled function should be.
# Note that because shared_var is shared, it already has a value, so it doesn't need to be an input to the function.
# Therefore, Theano implicitly considers shared_var an input to a function using shared_squared and so we don't need
# to include it in the inputs argument of theano.function.
function_1 = theano.function([], shared_squared)
print(function_1())


#optimization
import theano
a = theano.tensor.vector("a")      # declare symbolic variable
b = a + a ** 10                    # build symbolic expression
f = theano.function([a], b)        # compile function
print(f([0, 1, 2]))                # prints `array([0,2,1026])`
theano.printing.pydotprint(b, outfile="symbolic_graph_unopt.png", var_with_name_simple=True)  
theano.printing.pydotprint(f, outfile="symbolic_graph_opt.png", var_with_name_simple=True)  




#updates

#ex 1
state = theano.shared(0)
inc = T.iscalar('inc')
accumulator = function([inc], state, updates=[(state, state+inc)])
print accumulator(3)
print accumulator(3)
print accumulator(3)
print accumulator(3)
print accumulator(3)

# you can define more than one function to use the same shared variable. These functions can all update the value.


#ex2
# We can also update the state of a shared var in a function
subtract = T.matrix('subtract')
# updates takes a dict where keys are shared variables and values are the new value the shared variable should take
# Here, updates will set shared_var = shared_var - subtract
function_2 = theano.function([subtract], shared_var, updates={shared_var: shared_var - subtract})
print("shared_var before subtracting [[1, 1], [1, 1]] using function_2:")
print(shared_var.get_value())
# Subtract [[1, 1], [1, 1]] from shared_var
function_2(np.array([[1, 1], [1, 1]], dtype=theano.config.floatX))
print("shared_var after calling function_2:")
print(shared_var.get_value())
# Note that this also changes the output of function_1, because shared_var is shared!
print("New output of function_1() (shared_var**2):")
print(function_1())



#IfElse vs Switch
from theano import tensor as T
from theano.ifelse import ifelse
import theano, time, numpy

a,b = T.scalars('a', 'b')
x,y = T.matrices('x', 'y')

z_switch = T.switch(T.lt(a, b), T.mean(x), T.mean(y))
z_lazy = ifelse(T.lt(a, b), T.mean(x), T.mean(y))

f_switch = theano.function([a, b, x, y], z_switch,
                           mode=theano.Mode(linker='vm'))
f_lazyifelse = theano.function([a, b, x, y], z_lazy,
                               mode=theano.Mode(linker='vm'))

val1 = 0.
val2 = 1.
big_mat1 = numpy.ones((10000, 1000))
big_mat2 = numpy.ones((10000, 1000))

n_times = 10

tic = time.clock()
for i in range(n_times):
    f_switch(val1, val2, big_mat1, big_mat2)
print('time spent evaluating both values %f sec' % (time.clock() - tic))

tic = time.clock()
for i in range(n_times):
    f_lazyifelse(val1, val2, big_mat1, big_mat2)
print('time spent evaluating one value %f sec' % (time.clock() - tic))


# loop

# defining the tensor variables
X = T.matrix("X")
W = T.matrix("W")
b_sym = T.vector("b_sym")

results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym), sequences=X)
compute_elementwise = theano.function(inputs=[X, W, b_sym], outputs=results)

# test values
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

print(compute_elementwise(x, w, b))

# comparison with numpy
print(np.tanh(x.dot(w) + b))


#Broadcasting
r = T.row()
print r.broadcastable
mtr = T.matrix()
print mtr.broadcastable
f_row = theano.function([r, mtr], [r + mtr])
R = np.arange(3).reshape(1, 3)
print R
M = np.arange(9).reshape(3, 3)
print M

print f_row(R, M)
c = T.col()
print c.broadcastable
f_col = theano.function([c, mtr], [c + mtr])
C = np.arange(3).reshape(3, 1)
print C

M = np.arange(9).reshape(3, 3)
print f_col(C, M)



#gradients

foo = T.scalar('foo')
bar = foo**2
function_pow = theano.function([foo],bar)

# Recall that bar = foo**2
# We can compute the gradient of bar with respect to foo like so:
bar_grad = T.grad(bar, foo)
# We expect that bar_grad = 2*foo
bar_grad.eval({foo: 10})


#jacobian
# Recall that y = Ax + b

A = T.matrix('A')
x = T.vector('x')
b = T.vector('b')
y = T.dot(A, x) + b
# We can also compute a Jacobian like so:
y_J = theano.gradient.jacobian(y, x)
linear_mix_J = theano.function([A, x, b], y_J)
# Because it's a linear mix, we expect the output to always be A
print(linear_mix_J(np.array([[9, 8, 7], [4, 5, 6]], dtype=theano.config.floatX), #A
                   np.array([1, 2, 3], dtype=theano.config.floatX), #x
                   np.array([4, 5], dtype=theano.config.floatX))) #b



#MLP

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.

        :parameters:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_output, n_input = W_init.shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output,)
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output,
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               # The name parameter is solely for printing purporses
                               name='W',
                               # Setting borrow=True allows Theano to use user memory for this object.
                               # It can make code slightly faster by avoiding a deep copy on construction.
                               # For more details, see
                               # http://deeplearning.net/software/theano/tutorial/aliasing.html
                               borrow=True)
        # We can force our bias vector b to be a column vector using numpy's reshape method.
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting (described below)
        self.b = theano.shared(value=b_init.reshape(n_output, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               # Theano allows for broadcasting, similar to numpy.
                               # However, you need to explicitly denote which axes can be broadcasted.
                               # By setting broadcastable=(False, True), we are denoting that b
                               # can be broadcast (copied) along its second dimension in order to be
                               # added to another variable.  For more information, see
                               # http://deeplearning.net/software/theano/library/tensor/basic.html
                               broadcastable=(False, True))
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]
        
    def output(self, x):
        '''
        Compute this layer's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # Compute linear mix
        lin_output = T.dot(self.W, x) + self.b
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation is None else self.activation(lin_output))


class MLP(object):
    def __init__(self, W_init, b_init, activations):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)
        
        # Initialize lists of layers
        self.layers = []
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        
    def output(self, x):
        '''
        Compute the MLP's output given an input
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        for layer in self.layers:
            x = layer.output(x)
        return x

    def squared_error(self, x, y):
        '''
        Compute the squared euclidean error of the network output against the "true" output y
        
        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - y : theano.tensor.var.TensorVariable
                Theano symbolic variable for desired network output

        :returns:
            - error : theano.tensor.var.TensorVariable
                The squared Euclidian distance between the network output and y
        '''
        return T.sum((self.output(x) - y)**2)


# Gradient descent

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum
    
    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1
   
    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        previous_step = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        step = momentum*previous_step - learning_rate*T.grad(cost, param)
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))
    return updates



# Toy example

import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400                                   # training sample size
feats = 784                               # number of input variables

# generate a dataset: D = (input_values, target_class)
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 100

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(rng.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

# print("Initial model:")
# print(w.get_value())
# print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print("Final model:")
print(w.get_value())
print(b.get_value())
print("target values for D:")
print(D[1])
print("prediction on D:")
print(predict(D[0]))

theano.printing.pydotprint(prediction, outfile="logreg_pydotprint_prediction.png", var_with_name_simple=True)  




#2
# Training data - two randomly-generated Gaussian-distributed clouds of points in 2d space
np.random.seed(0)
# Number of points
N = 1000
# Labels for each cluster
y = np.random.random_integers(0, 1, N)
# Mean of each cluster
means = np.array([[-1, 1], [-1, 1]])
# Covariance (in X and Y direction) of each cluster
covariances = np.random.random_sample((2, 2)) + 1
# Dimensions of each point
X = np.vstack([np.random.randn(N)*covariances[0, y] + means[0, y],
               np.random.randn(N)*covariances[1, y] + means[1, y]]).astype(theano.config.floatX)
# Convert to targets, as floatX
y = y.astype(theano.config.floatX)
# Plot the data
plt.figure(figsize=(8, 8))
plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=3, cmap=plt.cm.cool)
plt.axis([-6, 6, -6, 6])
plt.show()


# First, set the size of each layer (and the number of layers)
# Input layer size is training data dimensionality (2)
# Output size is just 1-d: class label - 0 or 1
# Finally, let the hidden layers be twice the size of the input.
# If we wanted more layers, we could just add another layer size to this list.
layer_sizes = [X.shape[0], X.shape[0]*2, 1]
# Set initial parameter values
W_init = []
b_init = []
activations = []
for n_input, n_output in zip(layer_sizes[:-1], layer_sizes[1:]):
    # Getting the correct initialization matters a lot for non-toy problems.
    # However, here we can just use the following initialization with success:
    # Normally distribute initial weights
    W_init.append(np.random.randn(n_output, n_input))
    # Set initial biases to 1
    b_init.append(np.ones(n_output))
    # We'll use sigmoid activation for all layers
    # Note that this doesn't make a ton of sense when using squared distance
    # because the sigmoid function is bounded on [0, 1].
    activations.append(T.nnet.sigmoid)
# Create an instance of the MLP class
mlp = MLP(W_init, b_init, activations)

# Create Theano variables for the MLP input
mlp_input = T.matrix('mlp_input')
# ... and the desired output
mlp_target = T.vector('mlp_target')
# Learning rate and momentum hyperparameter values
# Again, for non-toy problems these values can make a big difference
# as to whether the network (quickly) converges on a good local minimum.
learning_rate = 0.01
momentum = 0.9
# Create a function for computing the cost of the network given an input
cost = mlp.squared_error(mlp_input, mlp_target)
# Create a theano function for training the network
train = theano.function([mlp_input, mlp_target], cost,
                        updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
# Create a theano function for computing the MLP's output given some input
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))


# Keep track of the number of training iterations performed
iteration = 0
# We'll only train the network with 20 iterations.
# A more common technique is to use a hold-out validation set.
# When the validation error starts to increase, the network is overfitting,
# so we stop training the net.  This is called "early stopping", which we won't do here.
max_iteration = 20
while iteration < max_iteration:
    # Train the network using the entire training set.
    # With large datasets, it's much more common to use stochastic or mini-batch gradient descent
    # where only a subset (or a single point) of the training set is used at each iteration.
    # This can also help the network to avoid local minima.
    current_cost = train(X, y)
    # Get the current network output for all points in the training set
    current_output = mlp_output(X)
    # We can compute the accuracy by thresholding the output
    # and computing the proportion of points whose class match the ground truth class.
    accuracy = np.mean((current_output > .5) == y)
    # Plot network output after this iteration
    plt.figure(figsize=(8, 8))
    plt.scatter(X[0, :], X[1, :], c=current_output,
                lw=.3, s=3, cmap=plt.cm.cool, vmin=0, vmax=1)
    plt.axis([-6, 6, -6, 6])
    plt.title('Cost: {:.3f}, Accuracy: {:.3f}'.format(float(current_cost), accuracy))
    plt.show()
    iteration += 1
