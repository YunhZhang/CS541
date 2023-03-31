import numpy as np

# Q1
def trainModel():
    pass
def testModel():
    pass

def doDoubleCrossValidation (D, k, H):
    allIdxs = np.arange(len(D))
    # Randomly split dataset into k folds
    idxs = np.random.permutation(allIdxs)
    idxs = idxs.reshape(k, -1)
    accuracies = []
    hyper_accuracy = 0
    for fold in range(k):
        # Get all indexes for this outer fold as validation dataset
        testIdxs_outer = idxs[fold,:]
        # Get all the other outer indexes as training dataset
        trainIdxs_outer = np.array(set(allIdxs) - set(testIdxs_outer)).flatten()
        # Generate the inner fold
        in_index = trainIdxs_outer.reshape(k,-1)
        for in_fold in range(k):
            #Get all indexes for this inner fold as validation dataset
            testIdxs_inner = in_index[in_fold,:]
            #Get all the other inner indexes as training dataset
            trainIdxs_inner = np.array(set(in_index) - set(testIdxs_inner)).flatten()
            # Train the model on the inner training data
            temp_model = trainModel(D[trainIdxs_inner], H)
            # Test the model on the inner testing data and select the best hyper-parameter
            temp_accuracy = testModel(temp_model, D[testIdxs_inner])
            if temp_accuracy > hyper_accuracy:
                hyper_accuracy = in_fold
                hyper_accuracy = temp_accuracy
                model = temp_model
        #test the model to calculate the accuracy         
        accuracies.append(testModel(model, D[trainIdxs_outer]))
        
    return np.mean(accuracies)

#####################################################################
#Q2

def fitting(X, w, b):
    yhat = X.dot(w) + b
    yhat = np.array(yhat)
    return yhat

def totel_loss(yhat, y, w, alpha):
    n = y.shape[0]
    mse = 1/(2*n) * np.sum(np.square(yhat - y))
    w2 = w.dot(w.T)
    L2 = 1/(2*n) * alpha * w2
    return  mse + L2

def random_split(x,y):
    train_per = 0.8
    np.random.permutation(x.shape[0])
    tr_length = x.shape[0] * train_per
    tr_length = int(tr_length)

    x_tr = x[:tr_length, :]
    x_val = x[tr_length:, :]

    y_tr = y[:tr_length]
    y_val = y[tr_length:]

    return x_tr, x_val, y_tr, y_val

def shufflearray(x,y):
    index = np.random.permutation(x.shape[0])
    return x[index],y[index]

def gradient_descent(X, y, w, b, lr, alpha):
    yhat = fitting(X, w, b)
    n = y.shape[0]
    X_T = X.T
    grad_w = 1/n * X_T.dot(yhat-y) + alpha/n * w
    grad_b = 1/n * np.sum((yhat-y))
    
    w = w - lr * grad_w
    b = b - lr * grad_b

    return w,b

def sgd(x, y, b, w, lr, epoch, batch_size, alpha):
    n = x.shape[0]
    x, y = shufflearray(x,y)
    for e in range(epoch - 1):
        for i in range(0,n,batch_size):
            X_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            yhat_batch = fitting(X_batch,w,b)
            loss = totel_loss(yhat_batch, y_batch, X_batch, alpha)
            w,b = gradient_descent(X_batch, y_batch, w, b, lr, alpha)
    return w,b

def iteration(x,y,lr,epoch,batch_size,alpha):
    m = x.shape[1]
    l = np.expand_dims(y, axis=-1)
    w = np.random.rand(m)
    b = np.random.rand(l.shape[1])
    w_new,b_new = sgd(x,y,b,w,lr,epoch,batch_size,alpha)

    return w_new,b_new

def validation(x,y,w,b,alpha):
    yhat = fitting(x, w, b)
    val_loss = totel_loss(yhat,y,w,alpha)
    return val_loss

def dataloader():
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    return X_tr, y_tr, X_te, y_te

def main():
    X_tr, y_tr, X_te, y_te = dataloader()
    X_tr, X_val, y_tr, y_val = random_split(X_tr,y_tr)

    hyperparameters = {
		"lr": [0.5, 0.01, 0.001, 0.0001],
		"epoch": [25, 50, 75, 100],
		"batch_size": [10, 30, 50, 100],
		"alpha": [5, 1, 0.5, 0.1]}	
    
    
    best_loss = 1e7  
    best_lr = -1
    best_epoch = -1
    best_batch_size = -1
    best_alpha = -1


    for lr in hyperparameters["lr"]:
        print("lr: ",lr)
        for epoch in hyperparameters["epoch"]:
            print("epoch:", epoch)
            for batch_size in hyperparameters["batch_size"]:
                for alpha in hyperparameters["alpha"]:
                    w_tr, b_tr = iteration(X_val,y_val,lr,epoch,batch_size,alpha)
                    tr_loss = validation(X_tr, y_tr, w_tr ,b_tr,alpha)

                    if tr_loss < best_loss:
                        best_loss = tr_loss
                        best_lr = lr
                        best_epoch = epoch
                        best_batch_size = batch_size
                        best_alpha = alpha

    best_w,best_b = iteration(X_tr,y_tr,best_lr,best_epoch,best_batch_size,best_alpha)
    te_loss = validation(X_te, y_te, best_w ,best_b,best_alpha)
    print("\n---------------------------------------------------------------------------------------------------------------------------------------------------------")
    print("best training loss: ", best_loss)
    print("best test loss: ", te_loss)
    print("best_learning_rate: ", best_lr)
    print("best_num_of_epoch: ", best_epoch)
    print("best_size_of_batch: ", best_batch_size)
    print("best_alpha: ", best_alpha)
    print("--------------------------------------------------------------------------------------------------------------------------------------------------------- \n")


if __name__ == '__main__':
    main()