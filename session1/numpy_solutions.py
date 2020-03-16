# 1

# l=[]
# for i in range(2000, 4001):
#     if (i%7==0) and (i%5!=0):
#         l.append(str(i))

# print l

# 2

# def fact(n):
# 	if n==0 :
# 		return 1
# 	return n * fact(n-1)

# n = int(raw_input())
# print fact(n)

# 3

# n = int(raw_input())
# d = dict()

# for i in range(1,n+1):
# 	d[i] = i*i

# print d

# 4

# def findMax(x,y):
# 	if x>y:
# 		return x
# 	return y

# print findMax(10,4)

# 5

# X, Y = raw_input().split()

# X = int(X)
# Y = int(Y)

# l = [[0 for i in range(X)] for j in range(Y)]
# print l

# for i in range(Y):
# 	for j in range(X):
# 		l[i][j] = i*j

# print l

# 6

# l = raw_input().split(',')
# l.sort()

# print ','.join(l)

# 7
 
# ans = []

# for i in range(1000,3001):
# 	s = str(i)

# 	if int(s[0])%2==0 and int(s[1])%2==0 and int(s[2])%2==0 and int(s[3])%2==0:
# 		ans.append(i)

# print len(ans)
# print ans

# 8

# s = raw_input()
# digits = 0
# alphabet = 0

# for ch in s:
# 	if ch>='0' and ch<='9':
# 		digits+=1
# 	if (ch>='a' and ch<='z') or (ch>='A' and ch<='Z'):
# 		alphabet += 1
# 	else:
# 		continue
# print digits,alphabet

# 9

# class Circle(object):
# 	def __init__(self,r):
# 		self.radius = r

# 	def computeArea(self):
# 		return self.radius**2*3.14

# myC = Circle(5)
# print myC.computeArea()

# 10

# class myClass(object):
# 	@staticmethod
# 	def myStaticMethod():
# 		print 'hello'
# myClass.myStaticMethod()
# #no need
# c = myClass()
# c.myStaticMethod()

# 11

# class Rectangle(object):
# 	def __init__(self,l,w):
# 		self.width = w
# 		self.length = l

# 	def computeArea(self):
# 		return self.width*self.length

# myC = Rectangle(3,4)
# print myC.computeArea()

# 12
# import random
# print random.random()*100+10

# 13

# from random import shuffle

# l = [3,4,2,1]
# shuffle(l)
# print l

# 14

l = [[[0 for i in range(8)] for j in range(5) ]for k in range(8)]


print l







