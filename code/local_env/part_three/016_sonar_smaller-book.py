# Binary Classification with Sonar Dataset: Standardized Smaller
import numpy
import pandas
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
numpy.random.seed(seed)

# Begin data preprocessing
# load dataset
dataframe = pandas.read_csv("../data/sonar.data", header=None)
dataset = dataframe.values

# split into input (X) and output (Y) variables
X = dataset[:,0:60].astype(float)
Y = dataset[:,60]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# End data preprocessing

# smaller model
def create_smaller():
    # create model
    model = Sequential()
    model.add(Dense(30, input_dim=60, kernel_initializer="normal", activation="relu"))
    model.add(Dense(1, kernel_initializer="normal", activation="sigmoid"))
    # Compile model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model

estimators = []
estimators.append(("standardize", StandardScaler()))
estimators.append(("mlp", KerasClassifier(build_fn=create_smaller, epochs=100, batch_size=5)))
pipeline = Pipeline(estimators)

skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(pipeline, X, encoded_Y, cv=skfold)
print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
