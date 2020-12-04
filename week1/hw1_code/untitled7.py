# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 22:17:59 2018

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:09:43 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

#1
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 2
plt.imshow(train_set_x_orig[index])
print (" 'y ='' " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


train_set_x_orig.shape
train_set_y.shape
test_set_x_orig.shape
test_set_y.shape

 #ﬂatten a matrix 
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T        #第幾張圖  , 自動拉成一列
train_set_x_flatten.shape
 
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
test_set_x_flatten.shape

  
 #standardize our dataset
train_set_x = train_set_x_flatten/255.      
test_set_x = test_set_x_flatten/255.

#def sigmoid function
def sigmoid(z): 
    s=1/(1+np.exp(-z))
    return s

sigmoid(0)
sigmoid(1)


    

#Initializing parameters


def initialize_with_normal(dim):         #先把w 跟b 維度列好
    w=np.random.randn(dim,1)
    b=0

    assert(w.shape==(dim,1))
    assert(isinstance(b,float) or isinstance(b,int))

    return w,b  

dim = 5
w, b = initialize_with_normal(dim)
print ("w = " + str(w))
print ("b = " + str(b))




#######助教 定義傳播函式
def propagate(w,b,X,Y):
    m=X.shape[1]
    A=sigmoid(np.dot(w.T,X)+b)
    cost=(-1/m)*(np.dot(Y,np.log(A).T)+np.dot((1-Y),np.log(1-A).T))
    
    dw=np.dot(X,(A-Y).T)/m
    db=np.sum((A-Y))/m
    
    assert(dw.shape==w.shape)
    assert(db.dtype==float)
    cost = np.squeeze(cost)#刪除陣列中為1的那個維度
    assert(cost.shape == ())#cost為實數
    grads={'dw':dw,
           'db':db}
    return grads,cost  



#ex:

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]]) 
grads, cost = propagate(w, b, X, Y) 
print ("dw = " + str(grads["dw"])) 
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))



#定义优化函数，进行梯度下降算法实现，得到最终优化参数W，b：        

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs=[]
    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)
       
        dw=grads['dw']
        db=grads['db']
        w=w-learning_rate*dw
        b=b-learning_rate*db
        if i%100==0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    params={'w':w,
            'b':b}
    grads = {"dw": dw,
             "db": db}  
    return params, grads, costs


params, grads, costs = optimize(w, b, X, Y, num_iterations=100 , learning_rate = 0.009 , print_cost = False)
print ("w = " + str(params["w"])) 
print ("b = " + str(params["b"])) 
print ("dw = " + str(grads["dw"])) 
print ("db = " + str(grads["db"]))


#####

def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)           ########or  助教code
    for i in range(A.shape[1]):              # Y_prediction[0,A[0,:]>0.5]=1 
        if A[0,i]>=0.5:                     #  Y_prediction[0,A[0,:]<=0.5]=0   之後跳到assert那行
            Y_prediction[0,i]=1
        else:
            Y_prediction[0,i]=0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction

#####助教
def model(X_train, Y_train, X_test, Y_test, num_iterations=None, learning_rate=None, print_cost = None):
    ##initalized parameters with zeros
    w, b=initialize_with_normal(X_train.shape[0])
    #Gradient descent
    parameters, grads ,costs =optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost )
    #retrieve parameters w and b from dictionary parameters
    w = parameters["w"]
    b = parameters["b"]
    #predict test / train set
    Y_prediction_test=predict(w,b,X_test)
    Y_prediction_train=predict(w,b,X_train)
    #print train test errors
    print('train accuracy: {} %'.format(100-np.mean(np.abs(Y_prediction_train-Y_train))*100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d={"costs": costs,
        "Y_prediction_test": Y_prediction_test, 
        "Y_prediction_train" : Y_prediction_train, 
        "w" : w, 
        "b" : b,
        "learning_rate" : learning_rate,
        "num_iterations": num_iterations       
    }
    return d




d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.001, print_cost = True)
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.01, print_cost = True)
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.05, print_cost = True)
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.1 , print_cost = True)
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.5, print_cost = True)
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.0001, print_cost = True)


##從測試集看看幾張圖片
index=1
plt.imshow(test_set_x[:,index].reshape((64,64,3))) #使用imshow必须是RGB图像格式，3通道
print('y= '+str(test_set_y[0,index])+ ", you predicted that it is a \""+classes[int(d['Y_prediction_test'][0,index])].decode('utf-8')+"\"picture.")


index=2
plt.imshow(test_set_x[:,index].reshape((64,64,3))) #使用imshow必须是RGB图像格式，3通道
print('y= '+str(test_set_y[0,index])+ ", you predicted that it is a \""+classes[int(d['Y_prediction_test'][0,index])].decode('utf-8')+"\"picture.")

index=3
plt.imshow(test_set_x[:,index].reshape((64,64,3))) #使用imshow必须是RGB图像格式，3通道
print('y= '+str(test_set_y[0,index])+ ", you predicted that it is a \""+classes[int(d['Y_prediction_test'][0,index])].decode('utf-8')+"\"picture.")

index=4
plt.imshow(test_set_x[:,index].reshape((64,64,3))) #使用imshow必须是RGB图像格式，3通道
print('y= '+str(test_set_y[0,index])+ ", you predicted that it is a \""+classes[int(d['Y_prediction_test'][0,index])].decode('utf-8')+"\"picture.")


index=5
plt.imshow(test_set_x[:,index].reshape((64,64,3))) #使用imshow必须是RGB图像格式，3通道
print('y= '+str(test_set_y[0,index])+ ", you predicted that it is a \""+classes[int(d['Y_prediction_test'][0,index])].decode('utf-8')+"\"picture.")

index=6
plt.imshow(test_set_x[:,index].reshape((64,64,3))) #使用imshow必须是RGB图像格式，3通道
print('y= '+str(test_set_y[0,index])+ ", you predicted that it is a \""+classes[int(d['Y_prediction_test'][0,index])].decode('utf-8')+"\"picture.")






####
costs = np.squeeze(d['costs']) 
plt.plot(costs) 
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning rate"])) 
plt.show()


####
learning_rates=[0.01,0.001,0.0001]
models = {} 
for i in learning_rates: 
    print ("learning rate is: " + str(i)) 
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = False) 
    print ('\n' + "-------------------------------------------------------" + '\n') 
for i in learning_rates: 
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
plt.ylabel('cost') 
plt.xlabel('iterations')
legend = plt.legend(loc='upper center', shadow=True) 
frame = legend.get_frame()
frame.set_facecolor('0.90') 
plt.show()








##test our imagE

my_image = "AI.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


my_image = "gargouille.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")


my_image = "la_defense.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")





my_image = "9229d2dc2e9ca559158895766771f3c2.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

my_image = "cat_in_iran.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_predicted_image = predict(d["w"], d["b"], my_image)

plt.imshow(image)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

