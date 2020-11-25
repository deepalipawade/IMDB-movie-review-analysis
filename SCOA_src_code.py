import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words=10000)
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)
def vectorize(sequences, dimension = 10000):
 results = np.zeros((len(sequences), dimension))
 for i, sequence in enumerate(sequences):
  results[i, sequence] = 1
 return results

data = vectorize(data)
targets = np.array(targets).astype("float32")
test_x = data[:10000]
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]
model = models.Sequential()
# Input - Layer
model.add(layers.Dense(50, activation = "relu", input_shape=(10000, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
# compiling the model
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)
results = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 500,
 validation_data = (test_x, test_y)
)
print("Test-Accuracy:", np.mean(results.history["val_accuracy"]))



import string
import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb
from nltk import word_tokenize
import string
TOP_WORDS = 10000

def Preparing_string(text_string, dimension = TOP_WORDS):
    text_string = text_string.lower()
    table = str.maketrans(dict.fromkeys(string.punctuation))
    text_string = text_string.translate(table)

    word2index = imdb.get_word_index()
    test=[]
    for word in word_tokenize(text_string):
        test.append(word2index[word])

    results = np.zeros(dimension)
    for _ , sequence in enumerate(test):
        if sequence < dimension:
            results[sequence] = 1

    print("\nOriginal string:", text_string,"\n")
    print("\nIndex conversion:", test,"\n")
    results = np.reshape(results,(1, TOP_WORDS))
    print("\nConvert to vectors:", results,"\n")
    return results

data_string = Preparing_string("I don't feel like I know the characters at all. I have no idea why the two soldiers were friends or what they had been through together. The cinematography tried so hard to make this an emotional shocking movie that it had the opposite effect. War scenes with gratuitous up close views of corpses and body parts that don't add anything to the story got old quick.")
print("predict:",model.predict(data_string))
print("predict_classes:",model.predict_classes(data_string))
print("------0 is Bad-------\n")

data_string = Preparing_string("I felt dirty, I felt tired, I felt hungry, I felt a will to succeed and I felt sadness when I was watching the movie. It felt like I was also fighting to reach Colonel MacKenzie for two hours. Several hours later after my emotions are still outside my body. Fantastic photo and music. Good casting of staff. The movie is just perfect!")
print("predict:",model.predict(data_string))
print("predict_classes:",model.predict_classes(data_string))
print("------1 is Good-------\n")

data_string = Preparing_string("I hate this movie")
print("predict:",model.predict(data_string))
print("predict_classes:",model.predict_classes(data_string))
print("------0 is Bad-------\n")


''' 2nd method for prediction '''

from nltk import word_tokenize
import nltk
nltk.download('punkt')
from keras.preprocessing import sequence
word2index = imdb.get_word_index()
test=[]
for word in word_tokenize( "i love this movie"):
     test.append(word2index[word])
test=sequence.pad_sequences([test],maxlen=TOP_WORDS)
print(model.predict(test))
print("predict_classes:",model.predict_classes(test))



