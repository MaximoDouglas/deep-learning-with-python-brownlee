# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
import pandas as pd

# Function to create model, required for KerasClassifier
def create_model(optimizer="rmsprop", init="glorot_uniform"):
  # create model
  model = Sequential()
  model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer=init))
  model.add(Dense(8, activation="relu", kernel_initializer=init))
  model.add(Dense(1, activation="sigmoid", kernel_initializer=init))

  # Compile model
  model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
  return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Begin data preprocessing
# load pima indians dataset
df = pd.read_csv("../data/pima-indians-diabetes_labeled.csv")

# split into input (X) and output (y) variables
X = df.drop(['class'], 1, inplace=False)
y = df['class']
# End of data preprocessing

# create model
model = KerasClassifier(build_fn=create_model)

# grid search epochs, batch size and optimizer
optimizers = ['rmsprop', 'adam']
init = ['glorot_uniform', 'normal', 'uniform']
eps = numpy.array([50, 100, 150])
batches = numpy.array([5, 10, 20])

param_grid = dict(optimizer=optimizers, epochs=eps, batch_size=batches, init=init)
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(X, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Result dict of all metrics Cross-Validation
result = grid_result.cv_results_

# List with all the params combinations
params = result['params']
# For each param combination, there are a mean_test_score and a std_test_score, as
#   the result of the Cross-validation made on the model configured with this combination
mean_test_scores = result['mean_test_score']
std_test_scores = result['std_test_score']

# For each params combination, print the respective results
for i in range(len(params)):
    print("%f (%f) with: %r" % (mean_test_scores[i], std_test_scores[i], params[i]))
