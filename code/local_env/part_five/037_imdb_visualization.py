# Load and Plot the IMDB dataset
import numpy
from keras.datasets import imdb
from matplotlib import pyplot

# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()

# summarize size
print("Training data: ")
print(X_train.shape)
print(y_train.shape)

# Summarize number of classes
print("Classes: ")
print(numpy.unique(y_train))

# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X_train))))

# Summarize review length
print("Review length: ")
result = []
for val in X_train:
    result.append(len(val))

print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))

# plot review length as a boxplot and histogram
pyplot.subplot(121)
pyplot.boxplot(result)
pyplot.subplot(122)
pyplot.hist(result)
pyplot.show()
