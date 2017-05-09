import xgboost as xgb
from keras.datasets import mnist
import numpy

#import data
data= mnist.load_data()
(X_train,y_train), (X_test, y_test)=data

#reshape data to make it suitable for boosting
X_train= X_train.reshape(X_train.shape[0], -1)
X_test= X_test.reshape(X_test.shape[0], -1)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#defining the parameters.
param = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # the training step for each iteration
    'silent': 1,  # logging mode - quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 10}  # the number of classes that exist in this datset
#defining number of rounds
num_round = 200

#training the classifier
bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtest)
best_preds = numpy.asarray([numpy.argmax(line) for line in preds])

#calculating precision score
from sklearn.metrics import precision_score

print (precision_score(y_test, best_preds, average='macro'))