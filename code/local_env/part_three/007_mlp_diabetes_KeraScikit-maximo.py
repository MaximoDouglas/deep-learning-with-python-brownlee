# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy
import pandas as pd

# Function to create model, required for KerasClassifier
def create_model():
  # create model
  model = Sequential()
  model.add(Dense(12, input_dim=8, activation="relu", kernel_initializer="uniform"))
  model.add(Dense(8, activation="relu", kernel_initializer="uniform"))
  model.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
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
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, y, cv=kfold)
print("Acc: %.2f%%"%((results.mean())*100))
