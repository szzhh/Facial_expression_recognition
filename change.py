import cv2
import numpy as np
import os

EXPRESSIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']
lst=[]
for i in range(7):
    #st='D:/AI/facial_expression_recognition/emoji/'+EXPRESSIONS[i]+'.png'
    img=cv2.imread('D:/AI/facial_expression_recognition/emoji/'+EXPRESSIONS[i]+'.png')
    print('D:/AI/facial_expression_recognition/emoji/'+EXPRESSIONS[i]+'.png')
    lst.append(img)
#arr=np.array(lst)
np.save('D:/AI/facial_expression_recognition/emoji.npy',lst)

l=np.load('D:/AI/facial_expression_recognition/emoji.npy')
for i in range(7):
    cv2.imshow('img',l[i])
    cv2.waitKey(0)