from keras.models import Sequential
from keras.layers import Dense
import numpy
import pandas as pd
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility - it allows that no matter if we execute
# the code more than one time, the random values have to be the same
seed = 7
numpy.random.seed(seed)

# Begin data preprocessing
# load pima indians dataset
df = pd.read_csv("../data/pima-indians-diabetes_labeled.csv")

# split into input (X) and output (y) variables
X = df.drop(['class'], 1, inplace=False)
y = df['class']
# End of data preprocessing

# split the dataset into training data and validating data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

# Compile model
# binary_crossentropy = logarithmic loss
# adam = gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=150, batch_size=10)

# Evaluating model with the training data
scores = model.evaluate(X_test, y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
