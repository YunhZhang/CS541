import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return np.dot(A,B) - C

def problem_1c (A, B, C):
    return A*B + np.transposed(C)

def problem_1d (x, y):
    return np.inner(x,y)

def problem_1e (A, x):
    return np.linalg.solve(A,x)

def problem_1f (A, i):
    return np.sum(A[i,0:-1:2])

def problem_1g (A, c, d):
    return np.mean(np.nonzero(A[(A>c) and (A<d)]))

def problem_1h (A, k):
    w, v = np.linalg.eig(A)
    w_idx = np.argsort(w)[::-1][:k]
    return v[w_idx]

def problem_1i (x, k, m, s):
    mean = x + m*np.ones(x.shape[0])
    std = s*np.eye(x.shape[0])
    size = (x.shape[0],k)
    return np.random.multivariate_normal(mean, std, size)

def problem_1j (A):
    n,m = A.shape
    return A[:,np.random.shuffle(m)]

def problem_1k (x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean)/std

def problem_1l (x, k):
    return np.tile(x,(k,1))

def problem_1m (X, Y):
    k,n = X.shape
    k,m = Y.shape
    X_3D = np.tile(X,(m,1,1))
    Y_3D = np.transpose(np.tile(Y,(n,1,1)))
    return np.linalg.norm(X_3D - Y_3D)

def problem_1n (matrices):
    num = 0
    while len(matrices) > 1:
        m0,n0 = matrices[0].shape
        m1,n1 = matrices[1].shape
        num += m0*n0*n1
        matrices[1] = matrices[0]*matrices[1]
        matrices = matrices.pop()
        
    return num

def linear_regression (X_tr, y_tr):
    m,n = X_tr.shape
    ones = np.ones(1,n)
    X_hat = np.vstack(X_tr,ones)
    W = np.dot(np.linalg.solve(np.dot(X_hat.T,X_hat), X_hat.T),y_tr)
    w = W[0:m,:]
    b = W[m:m+1,:]
    return w, b

def fmse(w,b,X,y):
    yhat = np.dot(X,w) + b 
    MSE = np.sum(np.square(yhat-y)) / (2*y.shape[0])
    return MSE

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    w, b = linear_regression(X_tr, y_tr)

    # Report fMSE cost on the training and testing data (separately)
    MSE_tr = fmse(w, b, X_tr, y_tr)
    MSE_te = fmse(w, b, X_te, y_te)
    print(MSE_tr)
    print(MSE_te)

