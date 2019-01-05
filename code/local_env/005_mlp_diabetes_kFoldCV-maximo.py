# MLP for Pima Indians Dataset with 10-fold cross validation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
import pandas as pd

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
df = pd.read_csv("./data/pima-indians-diabetes_labeled.csv")

# split into input (X) and output (y) variables
X = df.drop(['class'], 1, inplace=False)
y = df['class']

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []

for (train, test) in kfold.split(X, y):
  # create model
  model = Sequential()
  model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
  model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
  model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  # Fit the model
  model.fit(X.iloc[train], y.iloc[train], epochs=150, batch_size=10, verbose=0)

  # evaluate the model
  scores = model.evaluate(X.iloc[test], y.iloc[test], verbose=0)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  cvscores.append(scores[1] * 100)

print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
