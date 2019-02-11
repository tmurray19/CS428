import numpy as np
from scipy.optimize import curve_fit


Xs = np.array([0.0,1.0,2.0,3.0,4.0,5.0])
Ys = np.array([0.1,0.9,2.2,2.8,3.9,5.1])


x0 = np.array([0.0,0.0,0.0])


sigma = np.array([1.0,1.0,1.0,1.0,1.0,1.0])

def func(x, a, b, c):
    return a + b*x + c*x*x

B = curve_fit(func, Xs, Ys, x0, sigma)
print(B[0])
b0 = B[0]

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
print('1 - SSE/SSTo = ', 1-(SSE/SSTO))
