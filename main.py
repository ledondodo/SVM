import numpy  as np
import random , math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import norm

# generate data: class (A), 10 pts around (1.5, 0.5), 10 pts around (-1.5, 0.5)
#                class (B), 20 pts around (0.0, -0.5)
# stdev: 0.2
# use np.random.seed(100) to get the same random data every run
np.random.seed(100)
classA = np.concatenate((np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
                         np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]
inputs = np.concatenate((classA , classB))
targets = np.concatenate ((np.ones(classA.shape[0]),
                           -np.ones(classB.shape[0])))
N = inputs.shape[0] # Number of rows (samples)
permute=list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

# try this complex data:
# classA = np.concatenate((np.random.randn(10, 2) * 0.5 + [1.5, 0.5],
#                          np.random.randn(10, 2) * 0.4 + [-1.5, 0.5],
#                          np.random.randn(10, 2) * 0.6 + [-3.0, -3.0],
#                          np.random.randn(10, 2) * 0.3 + [-0.5, 3.0]))
# classB = np.concatenate((np.random.randn(10, 2) * 0.6 + [-1.5, -1.5],
#                          np.random.randn(10, 2) * 0.4 + [-1.5, 0.5],
#                          np.random.randn(10, 2) * 0.5 + [1.0, 0.5]))

# global var
C = 100 # high C for low mistakes

# functions
def kernel(x1, x2):
    return linker(x1,x2)

## different kernel functions, to use into kernel(x1,x2)
def linker(x1, x2):
    result = np.dot(x1,x2)
    return result

def polker (x1,x2):
    p = 4 # try to change this parameter
    return pow((np.dot(x1,x2)+1),p)

def RBFker (x1,x2):
    sig = 0.2 # try to change this parameter
    return math.exp(-1*(pow(norm(x1+ (-1*x2)),2)/(2*sig*sig)))



P = [ [ targets[i]*targets[j]*kernel(inputs[i],inputs[j]) for j in range(N) ] for i in range(N) ]

def objective(alpha):
    return 0.5 * np.dot(alpha,np.dot(P,alpha)) - np.sum(alpha)

def zerofun(alpha):
    return np.dot(alpha,targets)


## body
start = np.zeros(N)
B = [(0, C) for b in range(N)]
XC = {'type':'eq', 'fun':zerofun}

ret = minimize( objective , start , bounds=B, constraints=XC )
alpha = ret['x']
if(ret['success']): print('Success !')
else: print('Failure...')

# extract alpha, xi, ti
mask = np.where(alpha > 10e-5)
M = len(mask[0])
extract = [ [ alpha[mask[0][i]],
              inputs[mask[0][i]],
              targets[mask[0][i]] ]
            for i in range(M) ]

# b
s = extract[0][1]
ts = extract[0][2]
b = sum(extract[i][0]*extract[i][2]*kernel(s,extract[i][1])
            for i in range(M)) - ts

# indicator func
def ind(x,y):
    s = np.array([x,y])
    return sum(alpha[i]*targets[i]*kernel(s,inputs[i]) for i in range(N)) - b


## plot
xgrid = np.linspace(-5,5)
ygrid = np.linspace(-4,4)
grid = np.array([ [ ind(x,y) for x in xgrid ] for y in ygrid ])
plt.contour(xgrid, ygrid, grid,
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidths=(1, 3, 1))

plt.plot([p[0] for p in classA],
         [p[1] for p in classA],
         'b.')
plt.plot([p[0] for p in classB],
         [p[1] for p in classB],
         'r.')
plt.axis('equal') # force same scale on both axes
plt.savefig('svmplot.pdf') # save fig in a file
plt.show()



print("Done.")