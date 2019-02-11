#Student ID: 15315901
#Name: Taidgh Murray
#CS428/MA500 Homework 1

import math
import pylab
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization


# Open the data file in a read format
f = open('data.txt', 'r')

# yi = b0+b1*xi+b2*xi^2+error(i)

# Initialise x & y arrays, along with error and beta value

Xs, Ys= [], []
b0 = [0.0, 0.0, 0.0]

# Defining the function y = b0 + b1x + b2x^2
def func(a, b, c, x):
    return a +  b*x + c*x**2


# For-loop
# Iterates through lines in file
# If the number is an x, it's added to the Xi array
# If the number is a y, it's added to the Yi Array
for l in f:
    # Split the lines up by their = sign
    # print(l)
    sp = l.split('=')

    # Splits the x value by its ',' character
    xVal= sp[1].split(',')

    # Adds corresponding x & y values in the line ot the arrays
    Xs.append(float(xVal[0]))
    Ys.append(float(sp[2]))

# Closes file
f.close()


Xs = np.array(Xs)
Ys = np.array(Ys)

# Calculating the b0, b1 & b2 values
linear = np.linspace(0, Xs.max(), 100)
B, _ = optimization.curve_fit(func, Xs, Ys, p0=b0)
y = func(linear, *B)


# Plotting & drawing the graph
plt.plot(Xs, Ys, 'bo', label='Data')
plt.plot(linear, y, 'r--', label='Fit')
plt.title('Least square regression')
plt.show()


# Defining yHat - The fitted value
def yHat(i):
    return B[0] + B[1]*Xs[i]

# Defining yMean - The sample mean
yTotal = 0

for i in Ys:
    yTotal += i

yMean = (1/len(Ys))*(yTotal)

# Defining SSTO - The Total Sum of Squares
SSTO = 0
for i in Ys:
    SSTO += (i - yMean)**2

# Defining SSE - The Error Sum of Squares
SSE = 0
count = 0
for j in Ys:
    SSE += (j - float(yHat(count)))**2
    count+=1

# Defining SSR - The Regression Sum of Squares
count = 0
SSR = 0
for k in Ys:
    SSR += (float(yHat(count)) - yMean)**2
    count+=1

# Showcasing that r^2 = SSR/SSTO = (SSE/SSTO)-1 (Which, they largely are)
print(SSR/SSTO)
print((SSE/SSTO)-1)
