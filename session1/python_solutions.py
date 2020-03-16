import numpy as np

# 1
# A = np.zeros(10)
# A[4] =1 
# print A

# 2
# A = np.arange(10,101)
# print A

# 3
# A = np.arange(0,9).reshape(3,3)
# print A

# 4
# A = np.random.random((30,30,30))
# print A.mean(),A.max(),A.min()

# 5
# A = np.ones((10,10))
# A[1:-1,1:-1] = 0
# print A

# 6
# A = np.random.random((5,5))
# A = (A-np.mean(A))/np.std(A)
# print A

# 7
# A = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
# print(A)

# 8  Broadcasting
# A = np.zeros((5,5))
# B = np.arange(5)
# A += B
# print A

# 9
# A = np.random.random(10)
# A[A.argmax()] = 0
# print A

# 10
# A= np.arange(100)
# v = np.random.uniform(1,100)
# index = (np.abs(A-v)).argmin()
# print A[index]

# 11
# A = np.random.randint(0,10,(5,6,7,8))
# mySum = A.sum(axis=(-2,-1))
# print mySum.shape
# print mySum

# 12
# A = np.arange(1000)
# print A[np.argsort(A)[-10:]]

# 13
# A = np.array([3,4,1,2,3,4,1,1,1,4,1,2,4,5])

# print [index for index,val in enumerate(A) if val==1][4]

# 14
# A= np.arange(100).reshape(20,5)
# print A.max(axis=1)

# 15
# A = np.array([20,30,12,0,4,2,19,50,200,12,5,1,5])

# A = np.where(A<10,10,A)
# A = np.where(A>30,30,A)
# print A

# 16
# A = np.array([20,30,12,0,4,2,19,50,200,12,5,1,5])

# B = np.bincount(A)
# print B

# 17
# A= np.arange(20).reshape(4,5)

# B = np.average(A,axis=1,weights=[2,2,3,4,5])
# print B

# 18
# A = np.array([20,30,12,10,4,2,19,50,200,12,5,1,5])

# print np.log(A)
# print np.log10(A)
# print np.log2(A)

# 19
A = np.array([[1,2],[3,4]])   #the inverse might not exist!

print np.linalg.inv(A)


