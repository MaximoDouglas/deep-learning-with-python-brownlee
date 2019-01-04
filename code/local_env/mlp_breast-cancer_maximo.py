from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixed random seed for reproducibility - it allows that no matter if we execute
#     the code more than one time, the random values have to be the same
seed = 7
np.random.seed(seed)

# load pima indians dataset
df = pd.read_csv("./data/breast-cancer-wisconsin.csv")

# drop rows with '?' - This can be solved using better methods
df = df[~(df == '?').any(axis=1)]

# drop the id column - The id is not necessary
df.drop(['id'], 1, inplace=True)

# Getting the input X from the df and scaling it using sklearn preprocessing
X = preprocessing.scale(np.array(df.drop(['class'],1)))

# The original dataset have the classes 4 and 2. To turn this problem into a Keras
#     binary classification problem, I had to convert the 4 to 1 and the 2 to 0
df['class'].replace((4, 2), (1, 0), inplace=True)

# Getting the output y from the df, after the transformation
y = df['class']

# create model
model = Sequential()
model.add(Dense(12, input_dim=9, activation="relu", kernel_initializer="uniform"))
model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

# Compile model
#     binary_crossentropy = logarithmic loss
#     adam = gradient descent algorithm
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Fit the model
history = model.fit(X, y, verbose=0, validation_split=0.2, epochs=150, batch_size=10)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='lower right')
plt.savefig('./data/breast-cancer-wisconsin.png')
plt.show()
