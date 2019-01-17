# Open the data file in a read format
f = open('data.txt', 'r')

# y(i) = beta(0)+beta(1)*x(i)+beta(2)*x(i)^2+error(i)

# Initialise x & y arrays
Xi=[]
Yi=[]

# For-loop
# Iterates through lines in file
# If the number is an x, it's added to the Xi array
# If the number is a y, it's added to the Yi Array
for l in f:
    if:
