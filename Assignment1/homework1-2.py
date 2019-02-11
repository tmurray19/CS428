# Assignment 1 - Question 2

#Student ID: 15315901
#Name: Taidgh Murray
#CS428/MA500 Homework 1

import math
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, leastsq
from scipy.stats import t

# Defining data as np arrays
Xi1=np.array([7, 18, 5, 14, 11, 5, 23, 9, 16, 5])
Xi2=np.array([5.11, 16.70, 3.20, 7.00, 11.00, 4.00, 22.10, 7.00, 10.60, 4.80])
Yi=np.array([58, 152, 41, 93, 101, 38, 203, 78, 117, 44])
b0=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
bZ = np.array([0.0,0.0,0.0])

# Model definition
def model(Beta, Xi1, Xi2, Error, i):
    return Beta[0] + Beta[1]*Xi1[i] + Beta[2]*Xi2[i] + Error[i]

# Defining the least square estimator
def leastSquareEstimator(b,x, x2):
    return b[0] + b[1]*x[1] + b[2]*x2[2]

# Defining the least square estimator
def func(X, a, b, c):
    Xi1, Xi2 = X
    return a + b*Xi1 + c*Xi2


# Calculating the b0, b1 & b2 values
B = (curve_fit(func, (Xi1, Xi2), Yi, p0=bZ))
print('b0, b1, b2: ', B[0])
B0 = B[0]


# T value with 12 degrees of freedom & a significange level of 0.05
T = t.ppf(0.975, 12)
P = 3

#MSR = (1/p-1)(yTranspose*y - bTranspose * xTranspose * y)


# Defining yHat - The fitted value
def yHat(i):
    return B0[0] + B0[1]*Xi1[i]

# Defining e(i) - The residual value
def e(i):
    return (Yi[i] - yHat(i))

# Defining yMean - The sample mean
yTotal = 0

for i in Yi:
    yTotal += i

yMean = (1/len(Yi))*(yTotal)

# Defining SSTO - The Total Sum of Squares
SSTO = 0
for i in Yi:
    SSTO += (i - yMean)**2

# Defining SSE - The Error Sum of Squares
SSE = 0
count = 0
for j in Yi:
    SSE += (j - float(yHat(count)))**2
    count+=1

# Defining SSR - The Regression Sum of Squares
count = 0
SSR = 0
for k in Yi:
    SSR += (float(yHat(count)) - yMean)**2
    count+=1

# Defining MSR
MSR = (1/P-1) * (Yi.transpose()*Yi - b0.transpose()*Xi1.transpose()*Yi)

# Obtaining residuals
resVals=[]
for i in range(len(Yi)):
    resVals.append(e(i))

# Plotting the residuals
plt.plot(resVals, Yi, 'bo', label='Yi')
plt.plot(resVals, b0)
plt.title('Residuals vs. Yi')
plt.show()

plt.plot(resVals, Xi1, 'bo', label='Yi')
plt.plot(resVals, b0)
plt.title('Residuals vs. Xi1')
plt.show()

plt.plot(resVals, Xi2, 'bo', label='Yi')
plt.plot(resVals, b0)
plt.title('Residuals vs. Xi2')
plt.show()

# Printing MSE
print('MSE:')
MSE = SSE/len(Yi)-P
print(MSE)

# Printing Fstar value as found in the notes
print('F* value:')
FStar = MSR/MSE
print(FStar)
