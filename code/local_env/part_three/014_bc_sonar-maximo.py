# Accuracy: 80.78%
# Standard deviation: 5.35%

# Binary Classification with Sonar Dataset - StratifiedKFold evaluation
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# ----- Begin data preprocessing
# load dataset
df = pd.read_csv("../data/sonar.data", header=None)

# split into input (X) and output (Y) variables
X = df.drop(df.columns[-1],1)
y = df[df.columns[-1]]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_y = encoder.transform(y)
# ----- End data preprocessing

# baseline model
def create_model():
  # create model
  model = Sequential()
  model.add(Dense(60, input_dim=60, kernel_initializer="normal", activation="relu"))
  model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))

  # Compile model
  model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
  return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_model, epochs=100, batch_size=5)
skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_y, cv=skfold)

# Summarize
print("Accuracy: %.2f%%" % (results.mean()*100))
print("Standard deviation: %.2f%%" % (results.std()*100))
