import tkinter
from tkinter import *
from tkinter import filedialog, messagebox
import sklearn
from sklearn import tree
import random
from random import seed, randint
import os
import numpy

from ML_Script import *

root = Tk()

frameTop = Frame(root)
frameTop.pack()
frameMiddle = Frame(root)
frameMiddle.pack()
frameBottom = Frame(root)
frameBottom.pack()

boolDisplayGraph = BooleanVar()
boolDisplayTree = BooleanVar()


def fileBrowse():
    filepath = filedialog.askopenfilename(filetypes=(("csv files", "*.csv"), ("all files", "*.*")))
    entryFilePath.delete(0, END)
    entryFilePath.insert(0, filepath)
    if not(filepath.endswith(".csv")):
        m = messagebox.showinfo(message="Error, the Selected file is not a CSV file")
        entryStatus.delete(0, END)
        entryStatus.insert(0, "Invalid File - Not a (csv)data file.")
    else:
        entryStatus.delete(0, END)
        entryStatus.insert(0, "Valid Data File - Ready for Prediction")
        dataInfo, confMatrix, class_report, acc, accP, bal_acc, y_test, y_pred, classifier = dataLoading(filepath)

        dataInfo = dataInfo.split("\n")
        class_report = class_report.split("\n")

        lstDataInfo.delete(0, END)
        index = 0
        for x in dataInfo:
            lstDataInfo.insert(index, x)
            index = index + 1

        lstAccuracy.delete(0, END)
        lstAccuracy.insert(0, "Balanced Accuracy: " + str(bal_acc))
        lstAccuracy.insert(0, "Accuracy: " + str(accP))
        lstAccuracy.insert(0, "Accuracy: " + str(acc))
        index = 0
        for x in class_report:
            lstAccuracy.insert(index, x)
            index = index + 1

        displayGraph(y_test, y_pred, classifier)



def displayGraph(y_test, y_pred, x):
    if boolDisplayGraph.get() and not boolDisplayTree.get():
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=3)
        ax.set_xlabel('Encoded actual values')
        ax.set_ylabel('Encoded predicted values')
        ax.set_title("Gini Coefficient Prediction Accuracy")
        plt.show()
    elif boolDisplayTree.get() and not boolDisplayGraph.get():
        fig = plt.figure(figsize=(20, 10))
        _ = tree.plot_tree(x, filled=True)
        plt.show()
    elif boolDisplayTree.get() and boolDisplayGraph.get():
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], lw=3)
        ax.set_xlabel('Encoded actual values')
        ax.set_ylabel('Encoded predicted values')
        ax.set_title("Gini Coefficient Prediction Accuracy")
        fig = plt.figure(figsize=(20, 10))
        _ = tree.plot_tree(x, filled=True)
        plt.show()

def fileSave():
    filepath = filedialog.askdirectory()
    entrySaveLocation.insert(0, filepath)

def reset():
    entryFilePath.delete(0, END)
    entrySaveLocation.delete(0, END)
    entryStatus.delete(0, END)
    lstAccuracy.delete(0, END)
    lstDataInfo.delete(0, END)

def save():
    filepath = entrySaveLocation.get()
    if entryFilePath.get() == "":
        m = messagebox.showinfo(message="Please Select a csv file.")
    elif filepath == "":
        m = messagebox.showinfo(message="Please Select a destination to Save the results.")
    else:
        _dir = os.path.join(filepath + '/', 'output/')
        seed(randint(5, 50))
        randInt = randint(0, 25)

        if not os.path.exists(_dir):
            os.makedirs(_dir)

        text = lstDataInfo.get(0, END)

        try:
            with open(_dir + 'Data_Information_' + str(randInt) + '.txt', mode='r+',
                      encoding='UTF-8', errors='strict', buffering=1) as fh:
                fh.truncate(0)
                for x in text:
                    fh.write(x)
                    fh.write('\n')
        except FileNotFoundError:
            with open(_dir + 'Data_Information_' + str(randInt) + '.txt', mode='w+',
                      encoding='UTF-8', errors='strict', buffering=1) as fh:
                for x in text:
                    fh.write(x)
                    fh.write('\n')
        fh.close()

        text = lstAccuracy.get(0, END)

        try:
            with open(_dir + 'Accuracy_Report_' + str(randInt) + '.txt', mode='r+',
                      encoding='UTF-8', errors='strict', buffering=1) as fh:
                fh.truncate(0)
                for x in text:
                    fh.write(x)
                    fh.write('\n')
        except FileNotFoundError:
            with open(_dir + 'Accuracy_Report_' + str(randInt) + '.txt', mode='w+',
                      encoding='UTF-8', errors='strict', buffering=1) as fh:
                for x in text:
                    fh.write(x)
                    fh.write('\n')
        fh.close()


lblSelectFile = Label(frameTop, text="Select a csv file:")
lblModelFile = Label(frameTop, text="Model File:")
lblStatus = Label(frameTop, text="Status:")
lblPrediction = Label(frameMiddle, text="Data Information:")
lblAccuracy = Label(frameMiddle, text="Accuracy:")
lblSaveLocation = Label(frameBottom, text="Choose Where to Save the Results:")

btnBrowse = Button(frameTop, text="Browse", command=fileBrowse)
btnSaveBrowse = Button(frameBottom, text="Browse", command=fileSave)
btnSave = Button(frameBottom, text="Save", command=save)
btnReset = Button(frameBottom, text="Reset", command=reset)
entryFilePath = Entry(frameTop, width=70)
entryStatus = Entry(frameTop, width=70)
entrySaveLocation = Entry(frameBottom, width=70)
lstDataInfo = Listbox(frameMiddle, width=50, height=20)
lstAccuracy = Listbox(frameMiddle, width=50, height=20)
chkBoxDisplayGraph = Checkbutton(frameTop, text='Display Graph', variable=boolDisplayGraph, onvalue=True, offvalue=False)
chkBoxDisplayTree = Checkbutton(frameTop, text='Display Tree', variable=boolDisplayTree, onvalue=True, offvalue=False)

lblSelectFile.grid(row=0, column=0, padx=5, pady=5, sticky=W)
btnBrowse.grid(row=0, column=2, padx=5, pady=5)
entryFilePath.grid(row=0, column=1, padx=5, pady=5, sticky=W)
lblStatus.grid(row=2, column=0, padx=5, pady=5)
entryStatus.grid(row=2, column=1, padx=5, pady=5)
chkBoxDisplayGraph.grid(row=3, column=0, padx=5, pady=5)
chkBoxDisplayTree.grid(row=3, column=1, padx=5, pady=5)
lblPrediction.grid(row=3, column=0, padx=5, pady=15)
lblAccuracy.grid(row=3, column=1, padx=5, pady=15)
lstDataInfo.grid(row=4, column=0, padx=5, pady=5)
lstAccuracy.grid(row=4, column=1, padx=5, pady=5)
lblSaveLocation.grid(row=5, column=0, padx=5, pady=5)
entrySaveLocation.grid(row=6, column=0, padx=5, pady=5)
btnSaveBrowse.grid(row=6, column=1, padx=5, pady=5)
btnReset.grid(row=7, column=0, sticky=E)
btnSave.grid(row=7, column=1)

root.mainloop()
