import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

numpy.set_printoptions(threshold=numpy.nan)
# load dataset
dataset = numpy.loadtxt("train.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:13]
Xtraining,Xtest = X[:1665,:], X[1665:,:]
Y = dataset[:,13]
print len(Y)
Ytraining,Ytest = Y[:1665], Y[1665:]
print len(Ytraining)
print len(Ytest)
#print Y
# define base mode
#def baseline_model():
	# create model
seed = 7
numpy.random.seed(seed)
model = Sequential()
model.add(Dense(11, input_dim=13, init='uniform', activation='softplus'))
model.add(Dense(10, init='uniform', activation='softplus'))
model.add(Dense(10, init='uniform', activation='softplus'))
model.add(Dense(9, init='uniform', activation='softplus'))
model.add(Dense(9, init='uniform', activation='softplus'))
model.add(Dense(9, init='uniform', activation='softplus'))
model.add(Dense(8, init='uniform', activation='softplus'))
model.add(Dense(7, init='uniform', activation='softplus'))
model.add(Dense(6, init='uniform', activation='softplus'))
model.add(Dense(6, init='uniform', activation='softplus'))
model.add(Dense(5, init='uniform', activation='softplus'))
model.add(Dense(5, init='uniform', activation='softplus'))
model.add(Dense(4, init='uniform', activation='softplus'))
model.add(Dense(3, init='uniform', activation='softplus'))
model.add(Dense(3, init='uniform', activation='softplus'))
model.add(Dense(2, init='uniform', activation='softplus'))
model.add(Dense(1, init='uniform'))
# Compile model
model.compile(loss='mean_squared_error', optimizer='adam')#,metrics=['accuracy'])
sys.stdout = open('outputval.txt', 'w')
model.fit(Xtraining,Ytraining,validation_split = 0.25, nb_epoch=15000, batch_size=64)

#return model

# fix random seed for reproducibility

# evaluate model with standardized dataset
#estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=1, batch_size=5, verbose=0)
#kfold = KFold(n_splits=10, random_state=seed)
#results = cross_val_score(estimator, X, Y, cv=kfold)


#print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
predictions = model.predict(Xtest)
predictions1 = model.predict(X)
#f = open('output.txt', 'w')
#f.write(predictions[1])
sys.stdout = open('output1.txt', 'w')
print predictions1
sys.stdout = open('output.txt', 'w')
print predictions
sys.stdout = open('actual.txt', 'w')
print Ytest
