from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os
import torch
from detecto import core, utils, visualize
#required packages imported.

root = Tk() #creating the root tkinter window
root.geometry("750x600+300+150") #dimensions of the window
root.title("Fracture Detection GUI") # title of the window
root.resizable(width=True, height=True)#if the widow should be resiable which is true meaning it is
#how to get the name of a files path function
def openfn():
	global filename#test
 
	filename = filedialog.askopenfilename(title='open')
	print(filename)#print statment to see if it is getting the correct filepath
	return filename
#fucntion of displaying the image in tkinter
def open_img():
	x = openfn()
	img = Image.open(x)
	print(img)#printing the path of the image
	img = img.resize((250, 250), Image.ANTIALIAS)
	img = ImageTk.PhotoImage(img)
	panel = Label(root, image=img)
	panel.image = img
	panel.place(x=250, y=70)
	btn['state'] = DISABLED#the open image button is disabed.
	searchbtn['state'] = NORMAL # and the search button is now enabled
#function of what happens when the search button is clicked
def search():
	
	btn['state'] = NORMAL#open image button is returened to enabled
	
	fracturedataset = core.Dataset('trainingimagesgui/')#initializes the dataset the parameter is the folder in which to get the images for the dataset
	
	mymodel = core.Model(['crack'])#calls the neural network and it is looking for boxes labelled crack
	print ("model imported") #print function to see if the neural network has been called 
	
	mymodel.fit(fracturedataset)#train the neuralnetwork on the dataset
	print("finished training custom dataset")#prints to show that the neural network has finished training 
	#testing the neural network 
	testimage = utils.read_image(filename) #read the varible testimage which has the location of what is stored in variable filename
	print(filename) #test to see the correct location is being pulled
	endresult = mymodel.predict(testimage)#run the test image on the now trained neural network
	print ("prediction complete") #test to see that the test image has successfully ran on the neural network
	labels, boxes, scores = endresult #all the parameters the test image will have 
	print(labels) #test to see if the correct label is printed
	print(boxes) #this will print the coordiantes of the boxes 
	print(scores) #prints a score of how confident the neural network is
	visualize.show_labeled_image(testimage, boxes, labels)#display the image on the screen


btn = Button(root, text='open image', command=open_img)#creating the open image button 
btn.place(x=300, y=10)
searchbtn = Button(root, text='Search', command=search, state=DISABLED)#creating the search button
searchbtn.place(x=315, y=40)
root.mainloop()
