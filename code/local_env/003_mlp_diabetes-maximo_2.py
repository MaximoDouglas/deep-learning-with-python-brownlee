from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import numpy
import pandas as pd

# fix random seed for reproducibility - it allows that no matter if we execute
# the code more than one time, the random values have to be the same
seed = 7
numpy.random.seed(seed)

# load pima indians dataset
df = pd.read_csv("./data/pima-indians-diabetes_labeled.csv")

# split into input (X) and output (y) variables
#    scale X using sklearn preprocessing
X = preprocessing.scale(numpy.array(df.drop(['class'],1)))
y = df['class']

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
model.fit(X, y, validation_split=0.2, epochs=150, batch_size=10)
