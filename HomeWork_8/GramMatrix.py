# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 21:44:11 2020

@author: Tanzeel
"""
import numpy as np
import cv2
import glob
from scipy import signal as sig
import matplotlib.pyplot as plt
import re
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

def Class_Gram_features(img_class,kernels,Width=256,Height=256,downsample=16,C=10,Train_Flag=True):
    features_class = []
    if Train_Flag ==True:
        st = 'Training'
        print ('Extracting Gram matrix features from training data for class: {}'.format(img_class))
        images = glob.glob('../imagesDatabaseHW8/training/{}'.format(img_class) + '/*.jpg')
    else:
        st = 'Testing'
        print ('Extracting Gram matrix features from Testing data for class: {}'.format(img_class))
        images = glob.glob('../imagesDatabaseHW8/testing/{}'.format(img_class) + '_*.jpg')
    
    images = sort_alphanum(images)
    num_images = len(images)
    newH,newW = 96,96
    for i in range(num_images):
        image_color = cv2.imread(images[i])
        if image_color is not None:
            h,w = image_color.shape[0:2]
            #features_class.append(h,w)    #Check the size of each image
            #print('Currently running Image number:',images[i])
            image_color=cv2.resize(image_color,(Width,Height),interpolation = cv2.INTER_LINEAR)
            image_gray = cv2.cvtColor(image_color,cv2.COLOR_BGR2GRAY)
            #Do center cropping of the grayscale image--->Choose this or resizing
            #image_gray = image_gray[int(h/2-newH/2):int(h/2+newH/2),int(w/2-newW/2):int(w/2+newW/2)]
            feat_matrix = Get_FeatureMaps(image_gray,kernels,3,downsample,C)
            G = GramMatrix(feat_matrix)
            img_featur=G[np.triu_indices(C)]
            features_class.append((img_featur))
        else:
            print('Skipping Image number:',images[i])
            continue
    
    #Make a plot of last image in the class sequence for showing in report
    plt.imshow(G)
    plt.savefig('Gmatrix_Class_{}_'.format(img_class) + 'C_{}_'.format(int(C)) + '{}_'.format(int(i)) + '{}'.format(st) + '.jpg')
    plt.close()

    return np.array(features_class)

def Get_FeatureMaps(image_gray,kernels,kernel_size=3,downsample=16,C=10):
    # Function for computing the convolving the C kernels with an input grayscale image
    feature_matrix = np.zeros((int(image_gray.shape[0]/downsample),int(image_gray.shape[1]/downsample),C))
    for i in range(C):
        #Generate a random kernel
        #kernel= GenerateRandom_kernel(-1.0,1.0,(kernel_size,kernel_size))
        Ix = sig.convolve2d(image_gray,kernels[:,:,i],mode='same')
        # Possible downsample options
        #Ix = skimage.measure.block_reduce(image_gray, (downsample,downsample), np.mean)
        Ix=cv2.resize(Ix,(downsample,downsample),interpolation = cv2.INTER_LINEAR)
        feature_matrix[:,:,i] = Ix
    return feature_matrix

def GenerateRandom_kernel(low = -1.0,high=1.0,size= 3,C=10):
    kernels = np.zeros((size,size,C))
    for i in range(C):
        kernel = np.random.uniform(low,high,size)
        #Normalize the kernel to obtain zero sum
        kernel = kernel - np.mean(kernel)
        kernels[:,:,i] = kernel
    return kernels

def GramMatrix(input_feature_Matrix):
    # Function for generating the gram matrix
    # Get the size f input matrix
    h,w,c = input_feature_Matrix.shape #h=height, w=width, c= num of feature maps
    # Reshape the input matrix by flattening its spatial dimension
    features = input_feature_Matrix.reshape((c, h * w))
    # Get the gram matrix by taking the dot product
    G = np.dot(features, features.T)
    #Normalize the final output by dividing the total num of elements in the matrix
    G = G/(h*w*c)
    return G

''' Functions for sorting the list of images in alphanumeric order '''
def atoi (string):
    try:
        return int(string)
    except ValueError:
        return string
    
def alphanum_key(string):
    return[atoi(c) for c in re.split('([0-9]+)',string)]

def sort_alphanum(listofImages):
    return (sorted(listofImages,key=alphanum_key))
    
'''----------Main Code for training and testing a classifier-------'''

validation_accuracies = []
training_accuracies = []
best_valid = 0.0
for i in range(100):
    C = i+1
    kernels = GenerateRandom_kernel(-1.0,1.0,3,C)
    # Collect the Gram matrix features of entire training data
    cloudy_features=Class_Gram_features('cloudy',kernels,256,256,1,C,True)
    rain_features=Class_Gram_features('rain',kernels,256,256,1,C,True)
    shine_features=Class_Gram_features('shine',kernels,256,256,1,C,True)
    sunrise_features=Class_Gram_features('sunrise',kernels,256,256,1,C,True)
    #Get the corresponding Y_vector explaining class labels
    Y_C1=np.repeat("cloudy",len(cloudy_features))
    Y_C2=np.repeat("rain",len(rain_features))
    Y_C3=np.repeat("shine",len(shine_features))
    Y_C4=np.repeat("sunrise",len(sunrise_features))
    
    Train_X = np.concatenate((cloudy_features,rain_features,shine_features,sunrise_features),axis=0)
    Train_Y = np.concatenate((Y_C1,Y_C2,Y_C3,Y_C4),axis=0)
    
    # Split the data in training and test 
    X_train, X_val, y_train, y_val = train_test_split(Train_X, Train_Y, test_size=0.3, random_state=42)
    #Train the SVC model
    clf = SVC(C=1,kernel='rbf',gamma='scale')
    clf.fit(X_train,y_train)
    acc_train = clf.score(X_train,y_train)
    acc_val = clf.score(X_val,y_val)
    validation_accuracies.append(acc_val)
    training_accuracies.append(acc_train)
    predicted=clf.predict(X_val)
    print(f"[C-Vale: {i+1}] Training Accuracy: {acc_train:.3f}\t Validation Accuracy: {acc_val:.3f}")

    if acc_val > best_valid:
        best_valid = acc_val
        print("New Best model found")
        # save the model to disk
        filename = 'Best_model.pkl'
        pickle.dump([clf,kernels,X_train,y_train,X_val,y_val], open(filename, 'wb'))
        #disp = metrics.plot_confusion_matrix(clf, X_val, y_val)

#Plot the training and test accuracies
plt.figure()
plt.plot(training_accuracies)
plt.plot(validation_accuracies)
plt.legend (['Train', 'Validation'], loc= 'best',fontsize = 16)

plt.xlabel('C',fontsize = 16)
plt.ylabel('Train / Validation Accuracies',fontsize = 16)
plt.title('Accuraies Vs. parameter C for cropped images',fontsize = 16)

print('Training is finished')

#Open the best saved model with kernels and training and validation data
print('Evaluating the model on test dataset')
filename = 'Best_model.pkl'
loaded_model = pickle.load(open(filename, 'rb'))
best_kernels = loaded_model[1]
C_best=best_kernels.shape[2]
model = loaded_model[0]

cloudy_features_test=Class_Gram_features('cloudy',best_kernels,256,256,16,C_best,False)
rain_features_test=Class_Gram_features('rain',best_kernels,256,256,16,C_best,False)
shine_features_test=Class_Gram_features('shine',best_kernels,256,256,16,C_best,False)
sunrise_features_test=Class_Gram_features('sunrise',best_kernels,256,256,16,C_best,False)
Y_C1_test=np.repeat("cloudy",len(cloudy_features_test))
Y_C2_test=np.repeat("rain",len(rain_features_test))
Y_C3_test=np.repeat("shine",len(shine_features_test))
Y_C4_test=np.repeat("sunrise",len(sunrise_features_test))

Test_X = np.concatenate((cloudy_features_test,rain_features_test,shine_features_test,sunrise_features_test),axis=0)
Test_Y = np.concatenate((Y_C1_test,Y_C2_test,Y_C3_test,Y_C4_test),axis=0)
acc_test = model.score(Test_X,Test_Y)
predicted=model.predict(Test_X)
#Make a confusion matrix
disp = metrics.plot_confusion_matrix(model, Test_X, Test_Y)