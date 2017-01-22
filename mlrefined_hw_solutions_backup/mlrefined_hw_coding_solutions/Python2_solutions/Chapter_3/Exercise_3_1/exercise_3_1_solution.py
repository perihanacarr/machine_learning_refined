# This file is associated with the book
# "Machine Learning Refined", Cambridge University Press, 2016.
# by Jeremy Watt, Reza Borhani, and Aggelos Katsaggelos.

import numpy as np
import matplotlib.pyplot as plt
import csv

# import the dataset
reader = csv.reader(open("student_debt_data.csv", "rb"), delimiter=",")
d = list(reader)
data = np.array(d).astype("float")
x = data[:,0]
x.shape = (len(x),1)
y = data[:,1]
y.shape = (len(y),1)

# pad input with ones
o = np.ones((len(x),1))
x_new = np.concatenate((o,x),axis = 1)

# solve linear system of equations for regression fit
A = np.dot(x_new.T,x_new)
b = np.dot(x_new.T,y)
w = np.dot(np.linalg.pinv(A),b)

# print out predicted amount of student debt in 2050
debt_in_2050 = w[0] + w[1]*2050
print 'if this linear trend continues there will be ' + str(debt_in_2050[0]) + ' trillion dollars in student debt in 2050!'

# plot data with linear fit - this is optional
s = np.linspace(np.min(x),np.max(x))
t = w[0] + w[1]*s
plt.plot(s,t,linewidth = 3)
plt.scatter(x,y,linewidth = 2)
plt.xlabel('time')
plt.ylabel('debt in trillions of dollars')