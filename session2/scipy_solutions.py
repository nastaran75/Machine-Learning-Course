import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# slide 7
# A = np.ones(100)
# B = np.zeros(100)
# sio.savemat('myfile.mat',{'A':A, 'B':B})
# data = sio.loadmat('myfile.mat')
# print data.keys()
# print data['A']
# print data['B']
#
# #slide 10
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt
# # #
img = misc.imread('A.jpeg')   # for python3 import imageio then img = imageio.imread('A.jpeg')
# plt.imshow(img)
# plt.show()
# rotated = ndimage.rotate(img,40)
# zoomed = ndimage.zoom(img,(2,2,1))
# shifted = ndimage.shift(img,(50,50,0))
# cropped = img[100:,400:]
# n = 5
#
# plt.subplot(1,5,1)
# plt.title('original')
# plt.imshow(img)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,5,2)
# plt.title('shifted')
# plt.imshow(shifted)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,5,3)
# plt.title('rotated')
# plt.imshow(rotated)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,5,4)
# plt.title('cropped')
# plt.imshow(cropped)
# plt.xticks([])
# plt.yticks([])
# plt.subplot(1,5,5)
# plt.title('zoomed')
# plt.imshow(zoomed)
# plt.xticks([])
# plt.yticks([])
# plt.show()
#
# #slide 12 filtering
blurred=ndimage.gaussian_filter(img,sigma=3)
way_blurred=ndimage.gaussian_filter(img,sigma=5)
local_mean=ndimage.uniform_filter(img,size=(11,11,0))
min_img = ndimage.minimum_filter(img,size=10)
max_img = ndimage.maximum_filter(img,size=10)
med_img = ndimage.median_filter(img,size=10)


n = 7
fig,axes = plt.subplots(1,n)

plt.subplot(1,n,1)
plt.imshow(img)

plt.subplot(1,n,2)
plt.imshow(blurred)

plt.subplot(1,n,3)
plt.imshow(way_blurred)

plt.subplot(1,n,4)
plt.imshow(local_mean)

plt.subplot(1,n,5)
plt.imshow(min_img)

plt.subplot(1,n,6)
plt.imshow(max_img)

plt.subplot(1,n,7)
plt.imshow(med_img)

plt.show()
#
# #slide 17
# from scipy.linalg import *
# #
# A = np.arange(25).reshape(5, 5)  # expected a square matrix
# print eigvals(A)
# print eig(A)
# #
# # # matrix operation
# # A = [[1, 2], [3, 4]]
# # print inv(A)
# # print det(A)
# # print norm(A, ord=2)
# #
# # #solve equation
# # b = [1,2]
# # print solve(A,b)
#
#
# #slide 22
# # integration
from scipy.integrate import quad, dblquad, tplquad


def f(x):
    return x ** 3 + 3 * x + 100


print quad(f, -100, 100)


def integrand(x, n):
    return x ** n + n * x + 100


val, abserr = quad(integrand, -100, 100, args=(3))

print val, abserr

# use lambda
print quad(lambda x: x ** 3 + 3 * x + 100, -100, 100)


# higher dimensional
def integrand(x, y):
    return np.exp(-x ** 2 - y ** 2)


x_lower = 0
x_upper = 10
y_lower = 0
y_upper = 10

val, abserr = dblquad(integrand, x_lower, x_upper, lambda x: y_lower, lambda x: y_upper)

print val, abserr

# #
# # # slide 24 dense matrix
# from scipy.sparse import *
# #
# M = np.array([[1, 0, 0, 0], [0, 3, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
# A = csr_matrix(M)
# B = coo_matrix(M)
# print B
# print A
# print A.todense()
#
# # lil
# A = lil_matrix((4, 4))  # empty 4x4 sparse matrix
# A[0, 0] = 1
# A[1, 1] = 3
# A[2, 2] = A[2, 1] = 1
# A[3, 3] = A[3, 0] = 1
# print A
# print A.todense()

# slide 27 optimization

from scipy import optimize
def f(x):
    return 4 * x ** 3 + (x - 2) ** 2 + x ** 4


fig, ax  = plt.subplots()
x = np.linspace(-5, 3, 100)
ax.plot(x, f(x))
plt.show()
x_min = optimize.fmin_bfgs(f, -2)
print x_min


# solving function
def f(x):
	return x**3+100

fig, ax  = plt.subplots(figsize=(10,4))
x = np.linspace(0, 3, 1000)
y = f(x)
plt.plot(x,f(x))
plt.show()
print optimize.fsolve(f, 0.1)

#slide 31 interpolate
from scipy.interpolate import *
def f(x):
    return np.sin(x)


n = np.arange(10)
x = np.linspace(0, 9, 100)
noisy = f(n) + np.random.random()*0.1

f1 = interp1d(n, noisy, kind='nearest')
f2 = interp1d(n, noisy, kind='previous')
f3 = interp1d(n, noisy, kind='next')
f4 = interp1d(n, noisy, kind='linear')
f5 = interp1d(n, noisy, kind='cubic')

plt.plot(n, noisy, 'bs', label='data points')  # bo g^
plt.plot(x, f(x), label='true function')

plt.plot(x, f1(x), label='nearest')
plt.plot(x, f2(x), label='previous')
plt.plot(x, f3(x), label='next')
plt.plot(x, f4(x), label='linear')
plt.plot(x, f5(x), label='cubic')

plt.legend(loc='best')
plt.show()

#exercise
from scipy import stats
Y = stats.norm()
x = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(3, 1, sharex=True)

# plot the probability distribution function (PDF)
axes[0].plot(x, Y.pdf(x))

# plot the commulative distributin function (CDF)
axes[1].plot(x, Y.cdf(x));

# plot histogram of 1000 random realizations of the stochastic variable Y
axes[2].hist(Y.rvs(size=1000), bins=50);
plt.show()
