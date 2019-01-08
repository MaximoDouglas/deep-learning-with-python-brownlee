# Accuracy: 88.67%
# Standard deviation: 19.10%
# Multiclass Classification with the Iris Flowers Dataset - skfold and scale
import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# ------- Begin data preprocessing

# load dataset
df = pd.read_csv("./data/iris.data", header=None)
X = df.drop(df.columns[-1],1)
y = df[df.columns[-1]]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)

# ------- End data preprocessing

# define baseline model
def create_model():
  # create model
  model = Sequential()
  model.add(Dense(4, input_dim=4, activation="relu", kernel_initializer="normal"))
  model.add(Dense(3, init="normal", activation="sigmoid"))

  # Compile model
  model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
  return model

model = KerasClassifier(build_fn=create_model, epochs=200, batch_size=5)
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, encoded_y, cv=skfold)

# Summarize
print("Accuracy: %.2f%%"%(results.mean()*100))
print("Standard deviation: %.2f%%"%(results.std()*100))
