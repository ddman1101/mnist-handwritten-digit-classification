# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 01:44:02 2018

@author: HP
"""

#from PIL import Image
#import matplotlib.pyplot as plt
#import numpy as np
#im = Image.open( "C:\\Users\\HP\\Desktop\\碩士班\\類神經網路與深度學習\\training\\9\\*.png" )
#print(im)
#show image information
#print (im.format, im.size, im.mode)
#plt.imshow(im)
#matrix = np.array(im)
#print(matrix)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import skimage.io as io
import winsound
from skimage import data_dir

str0='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/0/*.png'
str1='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/1/*.png'
str2='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/2/*.png'
str3='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/3/*.png'
str4='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/4/*.png'
str5='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/5/*.png'
str6='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/6/*.png'
str7='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/7/*.png'
str8='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/8/*.png'
str9='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/9/*.png'
coll0 = io.ImageCollection(str0)
coll1 = io.ImageCollection(str1)
coll2 = io.ImageCollection(str2)
coll3 = io.ImageCollection(str3)
coll4 = io.ImageCollection(str4)
coll5 = io.ImageCollection(str5)
coll6 = io.ImageCollection(str6)
coll7 = io.ImageCollection(str7)
coll8 = io.ImageCollection(str8)
coll9 = io.ImageCollection(str9)
matrix0 = np.array(coll0)/255
matrix1 = np.array(coll1)/255
matrix2 = np.array(coll2)/255
matrix3 = np.array(coll3)/255
matrix4 = np.array(coll4)/255
matrix5 = np.array(coll5)/255
matrix6 = np.array(coll6)/255
matrix7 = np.array(coll7)/255
matrix8 = np.array(coll8)/255
matrix9 = np.array(coll9)/255
    
def f(x):
    return 1 / (1 + np.exp(-x))

matrix_0 = np.zeros(shape=(4000,784,1))
matrix_1 = np.zeros(shape=(4000,784,1))
matrix_2 = np.zeros(shape=(4000,784,1))
matrix_3 = np.zeros(shape=(4000,784,1))
matrix_4 = np.zeros(shape=(4000,784,1))
matrix_5 = np.zeros(shape=(4000,784,1))
matrix_6 = np.zeros(shape=(4000,784,1))
matrix_7 = np.zeros(shape=(4000,784,1))
matrix_8 = np.zeros(shape=(4000,784,1))
matrix_9 = np.zeros(shape=(4000,784,1))

for i in range(len(coll0)):
    matrix_0[i] = np.transpose(np.matrix(np.hstack((matrix0[i,0,:],matrix0[i,1,:],matrix0[i,2,:],matrix0[i,3,:],matrix0[i,4,:],\
                                                    matrix0[i,5,:],matrix0[i,6,:],matrix0[i,7,:],matrix0[i,8,:],matrix0[i,9,:],\
                                                    matrix0[i,10,:],matrix0[i,11,:],matrix0[i,12,:],matrix0[i,13,:],matrix0[i,14,:],matrix0[i,15,:],\
                                                    matrix0[i,16,:],matrix0[i,17,:],matrix0[i,18,:],matrix0[i,19,:],matrix0[i,20,:],matrix0[i,21,:],\
                                                    matrix0[i,22,:],matrix0[i,23,:],matrix0[i,24,:],matrix0[i,25,:],matrix0[i,26,:],matrix0[i,27,:]))))
    matrix_1[i] = np.transpose(np.matrix(np.hstack((matrix1[i,0,:],matrix1[i,1,:],matrix1[i,2,:],matrix1[i,3,:],matrix1[i,4,:],\
                                                    matrix1[i,5,:],matrix1[i,6,:],matrix1[i,7,:],matrix1[i,8,:],matrix1[i,9,:],\
                                                    matrix1[i,10,:],matrix1[i,11,:],matrix1[i,12,:],matrix1[i,13,:],matrix1[i,14,:],matrix1[i,15,:],\
                                                    matrix1[i,16,:],matrix1[i,17,:],matrix1[i,18,:],matrix1[i,19,:],matrix1[i,20,:],matrix1[i,21,:],\
                                                    matrix1[i,22,:],matrix1[i,23,:],matrix1[i,24,:],matrix1[i,25,:],matrix1[i,26,:],matrix1[i,27,:]))))
    matrix_2[i] = np.transpose(np.matrix(np.hstack((matrix2[i,0,:],matrix2[i,1,:],matrix2[i,2,:],matrix2[i,3,:],matrix2[i,4,:],\
                                                    matrix2[i,5,:],matrix2[i,6,:],matrix2[i,7,:],matrix2[i,8,:],matrix2[i,9,:],\
                                                    matrix2[i,10,:],matrix2[i,11,:],matrix2[i,12,:],matrix2[i,13,:],matrix2[i,14,:],matrix2[i,15,:],\
                                                    matrix2[i,16,:],matrix2[i,17,:],matrix2[i,18,:],matrix2[i,19,:],matrix2[i,20,:],matrix2[i,21,:],\
                                                    matrix2[i,22,:],matrix2[i,23,:],matrix2[i,24,:],matrix2[i,25,:],matrix2[i,26,:],matrix2[i,27,:]))))
    matrix_3[i] = np.transpose(np.matrix(np.hstack((matrix3[i,0,:],matrix3[i,1,:],matrix3[i,2,:],matrix3[i,3,:],matrix3[i,4,:],\
                                                    matrix3[i,5,:],matrix3[i,6,:],matrix3[i,7,:],matrix3[i,8,:],matrix3[i,9,:],\
                                                    matrix3[i,10,:],matrix3[i,11,:],matrix3[i,12,:],matrix3[i,13,:],matrix3[i,14,:],matrix3[i,15,:],\
                                                    matrix3[i,16,:],matrix3[i,17,:],matrix3[i,18,:],matrix3[i,19,:],matrix3[i,20,:],matrix3[i,21,:],\
                                                    matrix3[i,22,:],matrix3[i,23,:],matrix3[i,24,:],matrix3[i,25,:],matrix3[i,26,:],matrix3[i,27,:]))))
    matrix_4[i] = np.transpose(np.matrix(np.hstack((matrix4[i,0,:],matrix4[i,1,:],matrix4[i,2,:],matrix4[i,3,:],matrix4[i,4,:],\
                                                    matrix4[i,5,:],matrix4[i,6,:],matrix4[i,7,:],matrix4[i,8,:],matrix4[i,9,:],\
                                                    matrix4[i,10,:],matrix4[i,11,:],matrix4[i,12,:],matrix4[i,13,:],matrix4[i,14,:],matrix4[i,15,:],\
                                                    matrix4[i,16,:],matrix4[i,17,:],matrix4[i,18,:],matrix4[i,19,:],matrix4[i,20,:],matrix4[i,21,:],\
                                                    matrix4[i,22,:],matrix4[i,23,:],matrix4[i,24,:],matrix4[i,25,:],matrix4[i,26,:],matrix4[i,27,:]))))
    matrix_5[i] = np.transpose(np.matrix(np.hstack((matrix5[i,0,:],matrix5[i,1,:],matrix5[i,2,:],matrix5[i,3,:],matrix5[i,4,:],\
                                                    matrix5[i,5,:],matrix5[i,6,:],matrix5[i,7,:],matrix5[i,8,:],matrix5[i,9,:],\
                                                    matrix5[i,10,:],matrix5[i,11,:],matrix5[i,12,:],matrix5[i,13,:],matrix5[i,14,:],matrix5[i,15,:],\
                                                    matrix5[i,16,:],matrix5[i,17,:],matrix5[i,18,:],matrix5[i,19,:],matrix5[i,20,:],matrix5[i,21,:],\
                                                    matrix5[i,22,:],matrix5[i,23,:],matrix5[i,24,:],matrix5[i,25,:],matrix5[i,26,:],matrix5[i,27,:]))))
    matrix_6[i] = np.transpose(np.matrix(np.hstack((matrix6[i,0,:],matrix6[i,1,:],matrix6[i,2,:],matrix6[i,3,:],matrix6[i,4,:],\
                                                    matrix6[i,5,:],matrix6[i,6,:],matrix6[i,7,:],matrix6[i,8,:],matrix6[i,9,:],\
                                                    matrix6[i,10,:],matrix6[i,11,:],matrix6[i,12,:],matrix6[i,13,:],matrix6[i,14,:],matrix6[i,15,:],\
                                                    matrix6[i,16,:],matrix6[i,17,:],matrix6[i,18,:],matrix6[i,19,:],matrix6[i,20,:],matrix6[i,21,:],\
                                                    matrix6[i,22,:],matrix6[i,23,:],matrix6[i,24,:],matrix6[i,25,:],matrix6[i,26,:],matrix6[i,27,:]))))
    matrix_7[i] = np.transpose(np.matrix(np.hstack((matrix7[i,0,:],matrix7[i,1,:],matrix7[i,2,:],matrix7[i,3,:],matrix7[i,4,:],\
                                                    matrix7[i,5,:],matrix7[i,6,:],matrix7[i,7,:],matrix7[i,8,:],matrix7[i,9,:],\
                                                    matrix7[i,10,:],matrix7[i,11,:],matrix7[i,12,:],matrix7[i,13,:],matrix7[i,14,:],matrix7[i,15,:],\
                                                    matrix7[i,16,:],matrix7[i,17,:],matrix7[i,18,:],matrix7[i,19,:],matrix7[i,20,:],matrix7[i,21,:],\
                                                    matrix7[i,22,:],matrix7[i,23,:],matrix7[i,24,:],matrix7[i,25,:],matrix7[i,26,:],matrix7[i,27,:]))))
    matrix_8[i] = np.transpose(np.matrix(np.hstack((matrix8[i,0,:],matrix8[i,1,:],matrix8[i,2,:],matrix8[i,3,:],matrix8[i,4,:],\
                                                    matrix8[i,5,:],matrix8[i,6,:],matrix8[i,7,:],matrix8[i,8,:],matrix8[i,9,:],\
                                                    matrix8[i,10,:],matrix8[i,11,:],matrix8[i,12,:],matrix8[i,13,:],matrix8[i,14,:],matrix8[i,15,:],\
                                                    matrix8[i,16,:],matrix8[i,17,:],matrix8[i,18,:],matrix8[i,19,:],matrix8[i,20,:],matrix8[i,21,:],\
                                                    matrix8[i,22,:],matrix8[i,23,:],matrix8[i,24,:],matrix8[i,25,:],matrix8[i,26,:],matrix8[i,27,:]))))
    matrix_9[i] = np.transpose(np.matrix(np.hstack((matrix9[i,0,:],matrix9[i,1,:],matrix9[i,2,:],matrix9[i,3,:],matrix9[i,4,:],\
                                                    matrix9[i,5,:],matrix9[i,6,:],matrix9[i,7,:],matrix9[i,8,:],matrix9[i,9,:],\
                                                    matrix9[i,10,:],matrix9[i,11,:],matrix9[i,12,:],matrix9[i,13,:],matrix9[i,14,:],matrix9[i,15,:],\
                                                    matrix9[i,16,:],matrix9[i,17,:],matrix9[i,18,:],matrix9[i,19,:],matrix9[i,20,:],matrix9[i,21,:],\
                                                    matrix9[i,22,:],matrix9[i,23,:],matrix9[i,24,:],matrix9[i,25,:],matrix9[i,26,:],matrix9[i,27,:]))))
matrix = np.zeros(shape = (40000,784,1))
matrix[0:4000] = matrix_0
matrix[4000:8000] = matrix_1
matrix[8000:12000] = matrix_2
matrix[12000:16000] = matrix_3
matrix[16000:20000] = matrix_4
matrix[20000:24000] = matrix_5
matrix[24000:28000] = matrix_6
matrix[28000:32000] = matrix_7
matrix[32000:36000] = matrix_8
matrix[36000:40000] = matrix_9
np.random.shuffle(matrix)
iteration = 20
hidden_layers_nod = 30
eta = 0.5
consequence = 10

#weight = np.zeros(shape=(number_of_hidden_layers,hidden_layers_nod,hidden_layers_nod))
weight_input = np.matrix(np.random.uniform(-1,1,(hidden_layers_nod,784)))
weight_output = np.matrix(np.random.uniform(-1,1,(consequence,hidden_layers_nod)))

aim = np.zeros(shape = (40000,10,1)) 
aim[0:4000] = np.array([[1],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
aim[4000:8000] = np.array([[0],[1],[0],[0],[0],[0],[0],[0],[0],[0]])
aim[8000:12000] = np.array([[0],[0],[1],[0],[0],[0],[0],[0],[0],[0]])
aim[12000:16000] = np.array([[0],[0],[0],[1],[0],[0],[0],[0],[0],[0]])
aim[16000:20000] = np.array([[0],[0],[0],[0],[1],[0],[0],[0],[0],[0]])
aim[20000:24000] = np.array([[0],[0],[0],[0],[0],[1],[0],[0],[0],[0]])
aim[24000:28000] = np.array([[0],[0],[0],[0],[0],[0],[1],[0],[0],[0]])
aim[28000:32000] = np.array([[0],[0],[0],[0],[0],[0],[0],[1],[0],[0]])
aim[32000:36000] = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[1],[0]])
aim[36000:40000] = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0],[1]])

allmatrix = np.zeros(shape = (40000,794,1)) 

for m in range(40000):
    allmatrix[m] = np.vstack((aim[m],matrix[m]))
    
np.random.shuffle(allmatrix)

for l in range(40000):
    aim[l] = allmatrix[l,0:10]

for u in range(40000):
    matrix[u] = allmatrix[u,10:40000]
#for k in range(number_of_hidden_layers):
#    weight[k] = np.matrix(np.random.uniform(-1,1,(hidden_layers_nod,hidden_layers_nod)))
    
#def forward_input(k):
#    hidden_layer_output[k] = np.matrix( np.array(weight_list)[0] * matrix_0[d])

def f(x):
    return 1 / (1 + np.exp(-x))
#隱藏層 = 1,forward,"0"
#訓練模型
for c in range(iteration):
    for d in range(40000):
        hidden_layer1 = np.matrix( weight_input * matrix[d] )
        hidden_layer1_output = f(hidden_layer1)
       #hidden_layer2 = np.matrix( weight * hidden_layer1_output)
       #hidden_layer2_output = f(hidden_layer2)
        final = weight_output * hidden_layer1_output
        final_output = f(final)
        
        if (1/2)*(np.asscalar((sum(aim[d] - final)))**2) < 0.001:
            print("it's good")
        else:
            error_outputlayer = np.array((aim[d] - final_output)) * np.array(final_output) * np.array(np.transpose(np.matrix([1,1,1,1,1,1,1,1,1,1])) - final_output)
            error_hiddenlayer = np.array(hidden_layer1_output) * np.array((1 - hidden_layer1_output)) * np.transpose(np.array(np.transpose(error_outputlayer) * weight_output ))
            delta_weight_hidden_layer = eta * matrix[d] * np.transpose(error_hiddenlayer)
            delta_weight_output_layer = eta * hidden_layer1_output * np.transpose(error_outputlayer)
            weight_input = weight_input + np.transpose(delta_weight_hidden_layer)
            weight_output = weight_output + np.transpose(delta_weight_output_layer)            

#back propogation
       

#class NN():
#    def __init__(self,layers,learning_rate):
#        self.layers = layers 
#        self.learning_rate = learning_rate
       
#    def activation_function(self, x):
#        return 1 / (1 + math.exp(-x))  

#hidden_layers1 = 5   
#weight_num = 784*hidden_layers1
#weight1 = np.matrix(np.random.rand(28,hidden_layers1))
#weight2 = np.matrix(np.random.rand(5,10))
#print(weight1)
#print(weight2)
#len(weight1)
#test1 = matrix0 * weight1
        
#測試
def cost_function(x):
    return (1/2)*((1 - x)**2)


str0='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/0/*.png'
str1='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/1/*.png'
str2='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/2/*.png'
str3='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/3/*.png'
str4='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/4/*.png'
str5='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/5/*.png'
str6='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/6/*.png'
str7='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/7/*.png'
str8='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/8/*.png'
str9='C:/Users/HP/Desktop/碩士班/類神經網路與深度學習/training/9/*.png'
coll0 = io.ImageCollection(str0)
coll1 = io.ImageCollection(str1)
coll2 = io.ImageCollection(str2)
coll3 = io.ImageCollection(str3)
coll4 = io.ImageCollection(str4)
coll5 = io.ImageCollection(str5)
coll6 = io.ImageCollection(str6)
coll7 = io.ImageCollection(str7)
coll8 = io.ImageCollection(str8)
coll9 = io.ImageCollection(str9)
matrix0 = np.array(coll0)/255
matrix1 = np.array(coll1)/255
matrix2 = np.array(coll2)/255
matrix3 = np.array(coll3)/255
matrix4 = np.array(coll4)/255
matrix5 = np.array(coll5)/255
matrix6 = np.array(coll6)/255
matrix7 = np.array(coll7)/255
matrix8 = np.array(coll8)/255
matrix9 = np.array(coll9)/255

for i in range(len(coll0)):
    matrix_0[i] = np.transpose(np.matrix(np.hstack((matrix0[i,0,:],matrix0[i,1,:],matrix0[i,2,:],matrix0[i,3,:],matrix0[i,4,:],\
                                                    matrix0[i,5,:],matrix0[i,6,:],matrix0[i,7,:],matrix0[i,8,:],matrix0[i,9,:],\
                                                    matrix0[i,10,:],matrix0[i,11,:],matrix0[i,12,:],matrix0[i,13,:],matrix0[i,14,:],matrix0[i,15,:],\
                                                    matrix0[i,16,:],matrix0[i,17,:],matrix0[i,18,:],matrix0[i,19,:],matrix0[i,20,:],matrix0[i,21,:],\
                                                    matrix0[i,22,:],matrix0[i,23,:],matrix0[i,24,:],matrix0[i,25,:],matrix0[i,26,:],matrix0[i,27,:]))))
    matrix_1[i] = np.transpose(np.matrix(np.hstack((matrix1[i,0,:],matrix1[i,1,:],matrix1[i,2,:],matrix1[i,3,:],matrix1[i,4,:],\
                                                    matrix1[i,5,:],matrix1[i,6,:],matrix1[i,7,:],matrix1[i,8,:],matrix1[i,9,:],\
                                                    matrix1[i,10,:],matrix1[i,11,:],matrix1[i,12,:],matrix1[i,13,:],matrix1[i,14,:],matrix1[i,15,:],\
                                                    matrix1[i,16,:],matrix1[i,17,:],matrix1[i,18,:],matrix1[i,19,:],matrix1[i,20,:],matrix1[i,21,:],\
                                                    matrix1[i,22,:],matrix1[i,23,:],matrix1[i,24,:],matrix1[i,25,:],matrix1[i,26,:],matrix1[i,27,:]))))
    matrix_2[i] = np.transpose(np.matrix(np.hstack((matrix2[i,0,:],matrix2[i,1,:],matrix2[i,2,:],matrix2[i,3,:],matrix2[i,4,:],\
                                                    matrix2[i,5,:],matrix2[i,6,:],matrix2[i,7,:],matrix2[i,8,:],matrix2[i,9,:],\
                                                    matrix2[i,10,:],matrix2[i,11,:],matrix2[i,12,:],matrix2[i,13,:],matrix2[i,14,:],matrix2[i,15,:],\
                                                    matrix2[i,16,:],matrix2[i,17,:],matrix2[i,18,:],matrix2[i,19,:],matrix2[i,20,:],matrix2[i,21,:],\
                                                    matrix2[i,22,:],matrix2[i,23,:],matrix2[i,24,:],matrix2[i,25,:],matrix2[i,26,:],matrix2[i,27,:]))))
    matrix_3[i] = np.transpose(np.matrix(np.hstack((matrix3[i,0,:],matrix3[i,1,:],matrix3[i,2,:],matrix3[i,3,:],matrix3[i,4,:],\
                                                    matrix3[i,5,:],matrix3[i,6,:],matrix3[i,7,:],matrix3[i,8,:],matrix3[i,9,:],\
                                                    matrix3[i,10,:],matrix3[i,11,:],matrix3[i,12,:],matrix3[i,13,:],matrix3[i,14,:],matrix3[i,15,:],\
                                                    matrix3[i,16,:],matrix3[i,17,:],matrix3[i,18,:],matrix3[i,19,:],matrix3[i,20,:],matrix3[i,21,:],\
                                                    matrix3[i,22,:],matrix3[i,23,:],matrix3[i,24,:],matrix3[i,25,:],matrix3[i,26,:],matrix3[i,27,:]))))
    matrix_4[i] = np.transpose(np.matrix(np.hstack((matrix4[i,0,:],matrix4[i,1,:],matrix4[i,2,:],matrix4[i,3,:],matrix4[i,4,:],\
                                                    matrix4[i,5,:],matrix4[i,6,:],matrix4[i,7,:],matrix4[i,8,:],matrix4[i,9,:],\
                                                    matrix4[i,10,:],matrix4[i,11,:],matrix4[i,12,:],matrix4[i,13,:],matrix4[i,14,:],matrix4[i,15,:],\
                                                    matrix4[i,16,:],matrix4[i,17,:],matrix4[i,18,:],matrix4[i,19,:],matrix4[i,20,:],matrix4[i,21,:],\
                                                    matrix4[i,22,:],matrix4[i,23,:],matrix4[i,24,:],matrix4[i,25,:],matrix4[i,26,:],matrix4[i,27,:]))))
    matrix_5[i] = np.transpose(np.matrix(np.hstack((matrix5[i,0,:],matrix5[i,1,:],matrix5[i,2,:],matrix5[i,3,:],matrix5[i,4,:],\
                                                    matrix5[i,5,:],matrix5[i,6,:],matrix5[i,7,:],matrix5[i,8,:],matrix5[i,9,:],\
                                                    matrix5[i,10,:],matrix5[i,11,:],matrix5[i,12,:],matrix5[i,13,:],matrix5[i,14,:],matrix5[i,15,:],\
                                                    matrix5[i,16,:],matrix5[i,17,:],matrix5[i,18,:],matrix5[i,19,:],matrix5[i,20,:],matrix5[i,21,:],\
                                                    matrix5[i,22,:],matrix5[i,23,:],matrix5[i,24,:],matrix5[i,25,:],matrix5[i,26,:],matrix5[i,27,:]))))
    matrix_6[i] = np.transpose(np.matrix(np.hstack((matrix6[i,0,:],matrix6[i,1,:],matrix6[i,2,:],matrix6[i,3,:],matrix6[i,4,:],\
                                                    matrix6[i,5,:],matrix6[i,6,:],matrix6[i,7,:],matrix6[i,8,:],matrix6[i,9,:],\
                                                    matrix6[i,10,:],matrix6[i,11,:],matrix6[i,12,:],matrix6[i,13,:],matrix6[i,14,:],matrix6[i,15,:],\
                                                    matrix6[i,16,:],matrix6[i,17,:],matrix6[i,18,:],matrix6[i,19,:],matrix6[i,20,:],matrix6[i,21,:],\
                                                    matrix6[i,22,:],matrix6[i,23,:],matrix6[i,24,:],matrix6[i,25,:],matrix6[i,26,:],matrix6[i,27,:]))))
    matrix_7[i] = np.transpose(np.matrix(np.hstack((matrix7[i,0,:],matrix7[i,1,:],matrix7[i,2,:],matrix7[i,3,:],matrix7[i,4,:],\
                                                    matrix7[i,5,:],matrix7[i,6,:],matrix7[i,7,:],matrix7[i,8,:],matrix7[i,9,:],\
                                                    matrix7[i,10,:],matrix7[i,11,:],matrix7[i,12,:],matrix7[i,13,:],matrix7[i,14,:],matrix7[i,15,:],\
                                                    matrix7[i,16,:],matrix7[i,17,:],matrix7[i,18,:],matrix7[i,19,:],matrix7[i,20,:],matrix7[i,21,:],\
                                                    matrix7[i,22,:],matrix7[i,23,:],matrix7[i,24,:],matrix7[i,25,:],matrix7[i,26,:],matrix7[i,27,:]))))
    matrix_8[i] = np.transpose(np.matrix(np.hstack((matrix8[i,0,:],matrix8[i,1,:],matrix8[i,2,:],matrix8[i,3,:],matrix8[i,4,:],\
                                                    matrix8[i,5,:],matrix8[i,6,:],matrix8[i,7,:],matrix8[i,8,:],matrix8[i,9,:],\
                                                    matrix8[i,10,:],matrix8[i,11,:],matrix8[i,12,:],matrix8[i,13,:],matrix8[i,14,:],matrix8[i,15,:],\
                                                    matrix8[i,16,:],matrix8[i,17,:],matrix8[i,18,:],matrix8[i,19,:],matrix8[i,20,:],matrix8[i,21,:],\
                                                    matrix8[i,22,:],matrix8[i,23,:],matrix8[i,24,:],matrix8[i,25,:],matrix8[i,26,:],matrix8[i,27,:]))))
    matrix_9[i] = np.transpose(np.matrix(np.hstack((matrix9[i,0,:],matrix9[i,1,:],matrix9[i,2,:],matrix9[i,3,:],matrix9[i,4,:],\
                                                    matrix9[i,5,:],matrix9[i,6,:],matrix9[i,7,:],matrix9[i,8,:],matrix9[i,9,:],\
                                                    matrix9[i,10,:],matrix9[i,11,:],matrix9[i,12,:],matrix9[i,13,:],matrix9[i,14,:],matrix9[i,15,:],\
                                                    matrix9[i,16,:],matrix9[i,17,:],matrix9[i,18,:],matrix9[i,19,:],matrix9[i,20,:],matrix9[i,21,:],\
                                                    matrix9[i,22,:],matrix9[i,23,:],matrix9[i,24,:],matrix9[i,25,:],matrix9[i,26,:],matrix9[i,27,:]))))
matrix_test = np.zeros(shape = (40000,784,1))
matrix_test[0:4000] = matrix_0
matrix_test[4000:8000] = matrix_1
matrix_test[8000:12000] = matrix_2
matrix_test[12000:16000] = matrix_3
matrix_test[16000:20000] = matrix_4
matrix_test[20000:24000] = matrix_5
matrix_test[24000:28000] = matrix_6
matrix_test[28000:32000] = matrix_7
matrix_test[32000:36000] = matrix_8
matrix_test[36000:40000] = matrix_9


#讀取資料
string_path='C:/Users/HP/Desktop/test/data/*.png'
coll_final = io.ImageCollection(string_path)
matrix_final = np.array(coll_final)/255
matrix_test_final = np.zeros(shape=(len(matrix_final),784,1))
for n in range(len(coll_final)):
    matrix_test_final[n] = np.transpose(np.matrix(np.hstack((matrix_final[n,0,:],matrix_final[n,1,:],matrix_final[n,2,:],matrix_final[n,3,:],matrix_final[n,4,:],\
                                                    matrix_final[n,5,:],matrix_final[n,6,:],matrix_final[n,7,:],matrix_final[n,8,:],matrix_final[n,9,:],\
                                                    matrix_final[n,10,:],matrix_final[n,11,:],matrix_final[n,12,:],matrix_final[n,13,:],matrix_final[n,14,:],matrix_final[n,15,:],\
                                                    matrix_final[n,16,:],matrix_final[n,17,:],matrix_final[n,18,:],matrix_final[n,19,:],matrix_final[n,20,:],matrix_final[n,21,:],\
                                                    matrix_final[n,22,:],matrix_final[n,23,:],matrix_final[n,24,:],matrix_final[n,25,:],matrix_final[n,26,:],matrix_final[n,27,:]))))
def f(x):
    return 1 / (1 + np.exp(-x))
answer = np.zeros(shape = (len(matrix_final),1)) 
for p in range(10000):
    hidden_layer1 = np.matrix( weight_input * matrix_test_final[p] )
    hidden_layer1_output = f(hidden_layer1)
    final = weight_output * hidden_layer1_output
    final_output = f(final)
    temp = cost_function(np.array(final_output))
    temp[0] = round(np.asscalar(temp[0]),5)
    temp[1] = round(np.asscalar(temp[1]),5)
    temp[2] = round(np.asscalar(temp[2]),5)
    temp[3] = round(np.asscalar(temp[3]),5)
    temp[4] = round(np.asscalar(temp[4]),5)
    temp[5] = round(np.asscalar(temp[5]),5)
    temp[6] = round(np.asscalar(temp[6]),5)
    temp[7] = round(np.asscalar(temp[7]),5)
    temp[8] = round(np.asscalar(temp[8]),5)
    temp[9] = round(np.asscalar(temp[9]),5)
    temp_list = temp.tolist()
    answers = temp_list.index([np.asscalar(min(temp))])
    answer[p] = answers
    print(answers)

answer1 = np.transpose(np.array(answer,dtype = int))
f = open('C:\\Users\\HP\\Desktop\\碩士班\\類神經網路與深度學習\\A.txt', 'w', encoding = 'UTF-8') 
for y  in range(10000):
   f.write('{0:05}'.format(y+1)+ " "+ str(answer1[0,y])+"\n")        
f.close()








