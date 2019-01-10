# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Begin data preprocessing
# load pima indians dataset
dataset = numpy.loadtxt("../data/pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
# End data preprocessing

model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(8, kernel_initializer="uniform", activation="relu"))
model.add(Dense(1, kernel_initializer="uniform", activation="sigmoid"))

# Compile model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("Pre-loaded | %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Saving model and weights
# serialize model to JSON
model_json = model.to_json()
with open("../saved_models_and_weights/022.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("../saved_models_and_weights/022.h5")
# End saving model and weights

# Loading model and weights
# load model
json_file = open('../saved_models_and_weights/022.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("../saved_models_and_weights/022.h5")
# End loading model and weights

# evaluate loaded model on test data
loaded_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
score = loaded_model.evaluate(X, Y)
print("Loaded | %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
