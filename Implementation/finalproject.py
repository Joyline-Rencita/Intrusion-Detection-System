# ****************************************** Correct output with demo.csv dataset  *****************************************************

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import warnings
warnings.filterwarnings("ignore")


def upload_dataset():   
    global traindata_file_path
    traindata_file_path =  filedialog.askopenfilename(parent=root,initialdir = "/",title = "choose your file",filetypes = (("csv files",".csv"),("all files",".*")))
    print(traindata_file_path)

def binary_train_dataset():
    global traindata_file_path
    global best_algob
    global algorithmsb
    global X_trainb, y_trainb
    global resultsb
    
    print(traindata_file_path)
    data = pd.read_csv(traindata_file_path)
    
    Xb = data.drop('label', axis=1)
    Xb = Xb.drop('attack_cat', axis=1)
    yb = data['label']
    
    #Data processing
    X_trainb, X_testb, y_trainb, y_testb = train_test_split(Xb, yb, test_size = 0.10)


    #Algorithm comparison
    algorithmsb = {"RF":RandomForestClassifier(n_estimators=100, random_state=23),
                  "ET": ExtraTreesClassifier(n_estimators=100, random_state=0),
                  "DT" : DecisionTreeClassifier(),
                  "KNN":KNeighborsClassifier(n_neighbors=23),
                  "LR": linear_model.LogisticRegression()
                  }

    resultsb = {}
    cnt=0
    e3.delete('1.0', END)
    for algob in algorithmsb:
        clfb = algorithmsb[algob]
        clfb.fit(X_trainb, y_trainb)
        scoreb = clfb.score(X_testb, y_testb)
        resultsb[algob] = scoreb
        #print("%s : %f %%" % (algo, score*100))
        scrb = scoreb*100        
        e3.insert(END,algob + " : " + str(scrb) +"%" + "\n")
    best_algob = max(resultsb, key=resultsb.get)
    e9.delete(0,'end')
    e9.insert(0,str(best_algob))
    return resultsb

def multiclass_train_dataset():
    global traindata_file_path
    global best_algo
    global algorithms
    global X_train, y_train
    global results
    
    print(traindata_file_path)
    data = pd.read_csv(traindata_file_path)
    
    X = data.drop('attack_cat', axis=1)
    X = X.drop('label', axis=1)
    y = data['attack_cat']
    
    #Data processing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    #Algorithm comparison
    algorithms = {"RF":RandomForestClassifier(n_estimators=100, random_state=23),
                  "ET": ExtraTreesClassifier(n_estimators=100, random_state=0),
                  "DT" : DecisionTreeClassifier(),
                  "KNN":KNeighborsClassifier(n_neighbors=23),
                  "LR": linear_model.LogisticRegression()
                  }

    results = {}
    cnt=0
    e3.delete('1.0', END)
    for algo in algorithms:
        clf = algorithms[algo]
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        results[algo] = score
        #print("%s : %f %%" % (algo, score*100))
        scr=score*100        
        e3.insert(END,algo + "  : " + str(scr) +"%" + "\n")
    best_algo = max(results, key=results.get)
    e9.delete(0,'end')
    e9.insert(0,str(best_algo))
    return results

def test_file():
    global testdata_file_path
    testdata_file_path =  filedialog.askopenfilename(parent=root,initialdir = "/",title = "choose your file",filetypes = (("csv files",".csv"),("all files",".*")))
    print(testdata_file_path)

def binary_prediction():
    global testdata_file_path
    global algorithmsb
    global best_algob
    global X_trainb, y_trainb
    
    xtestb=pd.read_csv(testdata_file_path)
    #print(xtest)
    classifierb = algorithmsb[best_algob]
    classifierb.fit(X_trainb, y_trainb)
    y_predb = classifierb.predict(xtestb)
    y_predb=str(y_predb)
    y_predb=y_predb.replace('[','').replace(']','')
    e1.delete(0,'end')
    e1.insert(0,y_predb)


def multiclass_prediction():
    global testdata_file_path
    global algorithms
    global best_algo
    global X_train, y_train
    
    xtest=data = pd.read_csv(testdata_file_path)
    #print(xtest)
    classifier = algorithms[best_algo]
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(xtest)
    y_pred=str(y_pred)
    y_pred=y_pred.replace('[','').replace(']','')
    e2.delete(0,'end')
    e2.insert(0,y_pred)

def binary_plot_graph():
    global resultsb

    categories = resultsb.keys()
    values = resultsb.values()
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(categories, values, color='skyblue')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Bar Plot')
    canvas = FigureCanvasTkAgg(fig, master=c6)
    canvas.draw()
    canvas.get_tk_widget().pack()
    
    
def multiclass_plot_graph():
    global results

    categories = results.keys()
    values = results.values()
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.bar(categories, values, color='skyblue')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Bar Plot')
    canvas = FigureCanvasTkAgg(fig, master=c6)
    canvas.draw()
    canvas.get_tk_widget().pack()

def close_plot_graph():
     root.destroy()
     import finalproject


root = Tk() 
root.title('INTRUSION DETECTION SYSTEM')
root.geometry('1920x1200')
root.configure(background='gray')

c1 = Canvas(root,bg='gray',width=1520,height=80)
c1.place(x=5,y=5)
l1=Label(root,text='INTRUSION DETECTION SYSTEM',foreground="white",background='gray',font =('Verdana',20,'bold'))
l1.place(x=500,y=30)

#-------------

c2 = Canvas(root,bg='gray',width=290,height=705)
c2.place(x=5,y=90)

l2=Label(root,text='TRAIN PHASE',foreground="white",background='gray',font =('Verdana',15,'bold'))
l2.place(x=70,y=100)

b0=Button(root,borderwidth=1,relief="flat",text="UPLOAD DATASET",font="verdana 12 bold",bg="lightgray", fg="red",command = upload_dataset)
b0.place(height=50,width=260,x=22,y=130)

b1=Button(root,borderwidth=1,relief="flat",text="BINARY TRAIN",font="verdana 12 bold",bg="lightgray", fg="red",command = binary_train_dataset)
b1.place(height=50,width=260,x=22,y=190)

b11=Button(root,borderwidth=1,relief="flat",text="MULTICLASS TRAIN",font="verdana 12 bold",bg="lightgray", fg="red",command = multiclass_train_dataset)
b11.place(height=50,width=260,x=22,y=250)

l3=Label(root,text='TEST PHASE',foreground="white",background='gray',font =('Verdana',15,'bold'))
l3.place(x=75,y=320)

b2=Button(root,borderwidth=1,relief="flat",text="INPUT TEST FILE", font="verdana 12 bold", bg="lightgray", fg="red",command = test_file)
b2.place(height=50,width=260,x=22,y=350)

b3=Button(root,borderwidth=1,relief="flat",text="BINARY PREDICTION",font="verdana 12 bold", bg="lightgray", fg="red",command = binary_prediction)
b3.place(height=50,width=260,x=22,y=410)

b33=Button(root,borderwidth=1,relief="flat",text="MULTICLASS PREDICTION",font="verdana 12 bold", bg="lightgray", fg="red",command = multiclass_prediction)
b33.place(height=50,width=260,x=22,y=470)

l4=Label(root,text='RESULT PLOTS',foreground="white",background='gray',font =('Verdana',15,'bold'))
l4.place(x=65,y=535)

b4=Button(root,borderwidth=1, relief="flat",text="BINARY PLOT", font="verdana 12 bold",bg="lightgray",fg="red",command = binary_plot_graph)
b4.place(height=50,width=260,x=22,y=565)

b44=Button(root,borderwidth=1, relief="flat",text="MULTICLASS PLOT", font="verdana 12 bold",bg="lightgray",fg="red",command = multiclass_plot_graph)
b44.place(height=50,width=260,x=22,y=630)

b55=Button(root,borderwidth=1, relief="flat",text="CLOSE PLOT", font="verdana 12 bold",bg="lightgray",fg="red",command = close_plot_graph)
b55.place(height=50,width=260,x=22,y=695)

#-------------------

c3 = Canvas(root,bg='gray',width=500,height=300)
c3.place(x=300 ,y=90)

l5=Label(root,text='PREDICTION RESULTS',foreground="white",background='gray',font =('Verdana',15,'bold'))
l5.place(x=425,y=100)

l6=Label(root,text='STATUS - BINARY CLASS PREDICTION',foreground="white",background='gray',font =('Verdana',11,'bold'))
l6.place(x=385,y=150)
e1=Entry(root,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e1.place(height=50,width=330,x=385,y=180)

l7=Label(root,text='CATEGORY - MULTICLASS PREDICTION',foreground="white",background='gray',font =('Verdana',11,'bold'))
l7.place(x=385,y=240)
e2=Entry(root,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e2.place(height=50,width=330,x=385,y=270)

#--------------

c4 = Canvas(root,bg='gray',width=500,height=300) 
c4.place(x=300,y=395)

l8=Label(root,text='ANALYSIS OF MODELS',foreground="white",background='gray',font =('Verdana',15,'bold'))
l8.place(x=425,y=400)


l9=Label(root,text='ALGORITHM WISE ACCURACY',foreground="white",background='gray',font =('Verdana',11,'bold'))
l9.place(x=325,y=450)
e3=Text(root,font=('Verdana',12,'bold'),foreground='RED')
e3.place(height=100,width=455,x=325,y=480)

l15=Label(root,text='BEST ALGORITHM',foreground="white",background='gray',font =('Verdana',11,'bold'))
l15.place(x=325,y=590)
e9=Entry(root,font=('Verdana',12,'bold'),foreground='RED',justify=CENTER)
e9.place(height=50,width=455,x=325,y=620)

#----------------

c5 = Canvas(root,bg='gray',width=720,height=605) 
c5.place(x=805,y=90)

l16=Label(root,text='ACCURACY VS MODELS',foreground="white",background='gray',font =('Verdana',15,'bold'))
l16.place(x=1020,y=120)

c6 = Canvas(root,bg='white',width=580,height=500) 
c6.place(x=870,y=160)

#------

c7 = Canvas(root,bg='gray',width=1225,height=95)
c7.place(x=300,y=700)


mainloop()
