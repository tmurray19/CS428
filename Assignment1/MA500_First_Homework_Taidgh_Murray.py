#Student ID: 15315901
#Name: Taidgh Murray
#CS428/MA500 Homework 1

import math
import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Open the data file in a read format
f = open('data.txt', 'r')

# yi = b0+b1*xi+b2*xi^2+error(i)

# Initialise x & y arrays, along with error and beta value

Xs, Ys= [], []
b0 = [1.0, 1.0, 1.0]

def func(x, a, b, c):
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

# Changing X and Y data to numpy arrays
Xs = np.array(Xs)
Ys = np.array(Ys)
# Creating an arbitrary sigma array filled with ones to indicate error values
sigma = np.ones((100), dtype=float)


# Calculating the b0, b1 & b2 values
linear = np.linspace(Xs.min(), Xs.max(), 100)
B, _ = curve_fit(func, Xs, Ys, b0, sigma)
y = func(linear, *B)


# Plotting & drawing the graph

#Plotting the points
plt.plot(Xs,Ys, 'bo', label='Data')
#Plotting the line of best fit
plt.plot(linear, y, 'r--', label='Fit')
plt.title('Least square regression')
#Drawing graph
plt.show()

print('b0, b1, b2: ', B)

# Defining yHat - The fitted value
def yHat(i):
    return b0[0] + b0[1]*Xs[i]

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
    SSE += (j - (yHat(count)))**2
    count+=1

# Defining SSR - The Regression Sum of Squares
count = 0
SSR = 0
for k in Ys:
    SSR += ((yHat(count)) - yMean)**2
    count+=1

# Showcasing that r^2 = SSR/SSTO = (SSE/SSTO)-1 (Which, they largely are)
print('SSR/SSTo = ', SSR/SSTO)
print('1 - SSE/SSTo = ', (SSE/SSTO)-1)




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


"""
PLEASE NOTE:

I've decided to include the exact formatting of the data in this comment, just in case the program doesn't play nicely with the data in another format.


x_1 = 70,   y_1 = -130
x_2 = 3,   y_2 = 28.1
x_3 = 67,   y_3 = -91.90000000000003
x_4 = 38,   y_4 = 47.59999999999999
x_5 = 46,   y_5 = 36.39999999999998
x_6 = -16,   y_6 = -91.59999999999999
x_7 = 64,   y_7 = -79.60000000000002
x_8 = 10,   y_8 = 38
x_9 = 55,   y_9 = -17.5
x_10 = -17,   y_10 = -115.9
x_11 = 51,   y_11 = 4.899999999999977
x_12 = 23,   y_12 = 72.09999999999999
x_13 = 26,   y_13 = 72.39999999999999
x_14 = 12,   y_14 = 67.59999999999999
x_15 = 34,   y_15 = 68.39999999999999
x_16 = 58,   y_16 = -36.40000000000003
x_17 = 0,   y_17 = 6
x_18 = -18,   y_18 = -108.4
x_19 = 9,   y_19 = 34.9
x_20 = -9,   y_20 = -27.1
x_21 = 50,   y_21 = 10
x_22 = 27,   y_22 = 76.09999999999999
x_23 = 50,   y_23 = 14
x_24 = 48,   y_24 = 31.59999999999999
x_25 = 9,   y_25 = 46.9
x_26 = 26,   y_26 = 72.39999999999999
x_27 = 63,   y_27 = -67.90000000000003
x_28 = 66,   y_28 = -111.6
x_29 = 47,   y_29 = 8.099999999999994
x_30 = 60,   y_30 = -42
x_31 = 37,   y_31 = 62.09999999999999
x_32 = -13,   y_32 = -67.90000000000001
x_33 = 48,   y_33 = 27.59999999999999
x_34 = -10,   y_34 = -38
x_35 = 70,   y_35 = -138
x_36 = 20,   y_36 = 82
x_37 = 24,   y_37 = 80.40000000000001
x_38 = 35,   y_38 = 66.5
x_39 = 28,   y_39 = 71.59999999999999
x_40 = 15,   y_40 = 66.5
x_41 = 60,   y_41 = -58
x_42 = 56,   y_42 = -23.60000000000002
x_43 = 59,   y_43 = -43.10000000000002
x_44 = 23,   y_44 = 72.09999999999999
x_45 = 9,   y_45 = 50.9
x_46 = 48,   y_46 = 15.59999999999999
x_47 = 13,   y_47 = 70.09999999999999
x_48 = 51,   y_48 = 4.899999999999977
x_49 = 49,   y_49 = 6.899999999999977
x_50 = 16,   y_50 = 68.40000000000001
x_51 = 36,   y_51 = 44.40000000000001
x_52 = 12,   y_52 = 55.6
x_53 = 42,   y_53 = 43.59999999999999
x_54 = -8,   y_54 = -32.4
x_55 = -15,   y_55 = -71.5
x_56 = 65,   y_56 = -91.5
x_57 = -19,   y_57 = -113.1
x_58 = 7,   y_58 = 48.1
x_59 = 25,   y_59 = 68.5
x_60 = -16,   y_60 = -79.59999999999999
x_61 = -10,   y_61 = -54
x_62 = 31,   y_62 = 68.89999999999999
x_63 = 39,   y_63 = 64.90000000000001
x_64 = 70,   y_64 = -130
x_65 = 42,   y_65 = 51.59999999999999
x_66 = 53,   y_66 = -9.900000000000034
x_67 = 59,   y_67 = -47.10000000000002
x_68 = -17,   y_68 = -103.9
x_69 = 54,   y_69 = -7.600000000000023
x_70 = -16,   y_70 = -95.59999999999999
x_71 = -17,   y_71 = -103.9
x_72 = 53,   y_72 = 6.099999999999966
x_73 = 42,   y_73 = 51.59999999999999
x_74 = -10,   y_74 = -66
x_75 = 37,   y_75 = 58.09999999999999
x_76 = 69,   y_76 = -113.1
x_77 = 48,   y_77 = 27.59999999999999
x_78 = -8,   y_78 = -32.4
x_79 = 59,   y_79 = -47.10000000000002
x_80 = 28,   y_80 = 71.59999999999999
x_81 = 63,   y_81 = -71.90000000000003
x_82 = 0,   y_82 = 22
x_83 = 64,   y_83 = -83.60000000000002
x_84 = 66,   y_84 = -103.6
x_85 = 50,   y_85 = 10
x_86 = -7,   y_86 = -21.9
x_87 = 39,   y_87 = 68.90000000000001
x_88 = 47,   y_88 = 24.09999999999999
x_89 = 46,   y_89 = 24.39999999999998
x_90 = 53,   y_90 = 10.09999999999997
x_91 = 40,   y_91 = 42
x_92 = -2,   y_92 = -4.4
x_93 = 60,   y_93 = -46
x_94 = -11,   y_94 = -57.1
x_95 = -4,   y_95 = -23.6
x_96 = 0,   y_96 = -2
x_97 = -12,   y_97 = -64.40000000000001
x_98 = 28,   y_98 = 79.59999999999999
x_99 = 57,   y_99 = -33.90000000000003
x_100 = 52,   y_100 = 7.599999999999966
"""
