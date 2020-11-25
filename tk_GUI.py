from tkinter import  *
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
from tkinter import messagebox

#--------------------------------------
import numpy as np
from keras.utils import to_categorical
from keras import models
from keras import layers
from keras.datasets import imdb
import string
import numpy as np
from keras import models
from keras import layers
from keras.datasets import imdb
from nltk import word_tokenize
import string

TOP_WORDS = 10000
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
global model
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

def top4():
    global e1,e2,t
    global variable,t
    t=Toplevel()
    t.title('Submit Review')
    t.configure(background='LIGHTGOLDENROD')
    t.geometry("500x500")
    l=['Hindi','Marathi','English']
    Label(t, text="Submit Your Review",bg='LIGHTGOLDENROD',font=("Arial Bold", 15)).place(relx=0.2,rely=0.01)
    Label(t, text="Select Language",bg='LIGHTGOLDENROD',font=("Arial Bold", 10)).place(relx=0.25,rely=0.1)    
    variable = StringVar(t)
    variable.set("Language")
    w = OptionMenu(t,variable,*l,command=ok_t4).place(relx=0.55,rely=0.1)



def ok_t4(value):
    global c,conn,p,d
    p=value
    if p == 'English':
        l=['Shape of water','Spiderman','Interstellar']
    elif p == 'Hindi' :
        l = ['Lunchbix','Dhoom','URI']
    else:
        l = ['Natsamrat','Natrang','Jatra']
    Label(t, text="Select Movie Name",bg='LIGHTGOLDENROD',font=("Arial Bold", 10)).place(relx=0.25,rely=0.2)
    variable = StringVar(t)
    variable.set("Movie")
    w = OptionMenu(t,variable,*l,command=ok1_t4).place(relx=0.55,rely=0.2)


def ok1_t4(value):
    global m,q,textbox_review,e2,e3,id
    m=(value)
    Label(t,text="Write Review : ",bg='LIGHTGOLDENROD',font=("Arial Bold", 10),fg="black").place(relx=0.25,rely=0.3)
    '''------------ REVIEW ENTRY BOX ---------------'''
    textbox_review = Entry(t)
    textbox_review.place(relx=0.55,rely=0.3) 
    z=Button(t,text="Okay",bg="GOLDENROD",font=("Arial Bold", 8),command=predict_sentiment).place(relx=0.35,rely=0.4)
    

def predict_sentiment():
    input_review =str(textbox_review.get())
    print(input_review)
    input_review = Preparing_string(input_review)
    print("predict:",model.predict(input_review))
    
    print("predict_classes:",model.predict_classes(input_review))
    p = model.predict_classes(input_review)[[0]]
    prediction_class = p[[0]]
    if p == 0:
        review = "Movie is BAD"
    else:
        review = "Movie is GOOD"
        
    Label(t,text="Prediction : ",bg='LIGHTGOLDENROD',font=("Arial", 10),fg="black").place(relx=0.25,rely=0.5)
    Label(t,text=review,bg='LIGHTGOLDENROD',font=("Arial Bold", 10)).place(relx=0.55,rely=0.5)
    z=Button(t,text="Done",bg="GOLDENROD",font=("Arial Bold", 12),command=closeT).place(relx=0.35,rely=0.6)

    

def closeT1():
    t1.destroy()

def closeT():
    t.destroy()
   
def close():
    root.destroy()
    
'''---------------START CODE-------------'''

global root
root = Tk()
root.geometry("600x600")
root.configure(background='LIGHTGOLDENROD')
root.title("IMDB Movie Review Sentiment Detection")
l1=Label(root,text="MOVIE   REVIEW  SENTIMENT DETECTION",font=("Arial Bold", 20),bg='LIGHTGOLDENROD',fg="black")
l1.place(relx=0.48, rely=0.05, anchor=CENTER)


b3=Button(root,text="MOVIE REVIEW PRODUCTS",bg="ORANGE",fg="black",height=2,width=25,justify="center",command=top4).place(relx=0.35 ,rely=0.35)
		
b7=Button(root,text="DONE",bg="TOMATO",fg="BLACK",height=3,width=15,command=close)#.grid(row=9,column=3)
b7.place(relx=0.38, rely=0.85)

root.mainloop()






