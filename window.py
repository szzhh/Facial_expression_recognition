import os
import sys
import  cv2
from PySide2.QtWidgets import *
from PySide2.QtCore import *
from PySide2.QtGui import  *
from PySide2.QtUiTools import QUiLoader
import resnet
import torch
import numpy as np
import time
import random
from playsound import playsound

EXPRESSIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']
num=0
lst=[]
tim=10
for i in range(7):
    #st='D:/AI/facial_expression_recognition/emoji/'+EXPRESSIONS[i]+'.png'
    imge=cv2.imread(os.path.split(os.path.realpath(__file__))[0]+'/emoji/'+EXPRESSIONS[i]+'.png')
    lst.append(imge)
class Window(QWidget):
    def __init__(self):
        super().__init__()
        # 从文件中加载UI定义
        qfile_stats=QFile(os.path.split(os.path.realpath(__file__))[0]+'/window.ui')
        qfile_stats.open(QFile.ReadOnly)
        qfile_stats.close()
        # 从 UI 定义中动态 创建一个相应的窗口对象
        # 注意：里面的控件对象也成为窗口对象的属性了
        # 比如 self.ui.button , self.ui.textEdit
        self.ui = QUiLoader().load(qfile_stats)
        self.net = torch.load(os.path.split(os.path.realpath(__file__))[0]+'/fer2013_resnet_lr_3.pkl', map_location='cpu')
        self.face_cascade = cv2.CascadeClassifier(os.path.split(os.path.realpath(__file__))[0]+'/haarcascade_frontalface_default.xml')  # 加载人脸特征库
        self.ui.pushButton.clicked.connect(self.run)
        self.ui.pushButton_2.clicked.connect(self.close)
        self.ui.checkBox.stateChanged.connect(self.check)
        self.ui.checkBox_2.stateChanged.connect(self.check)
        self.ui.checkBox_3.stateChanged.connect(self.check)
        self.ui.show()
        
    def run(self):
        global tim
        global num
        print('开始')
        num+=1
        st='第'+str(num)+'轮：'
        self.ui.textEdit.append(st)
        self.ui.textEdit.moveCursor(QTextCursor.End)
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        # while (not cap.isOpened()):
        #     v_num+=1
        #     cap = cv2.VideoCapture(v_num)
        lstn=[]
        score=0
        for i in range(5):
            output=9
            n=random.randint(0,6)
            while (n in lstn):
                n=random.randint(0,6)
            lstn.append(n)
            st=str(i+1)+'、请做出'+EXPRESSIONS[n]+'表情'
            self.ui.textEdit.append(st)
            self.ui.textEdit.moveCursor(QTextCursor.End)
            #展示表情
            show = cv2.cvtColor(lst[n], cv2.COLOR_BGR2RGB)
            showImage = QImage(show.data,show.shape[1],show.shape[0],QImage.Format_RGB888)
            self.ui.label_2.setPixmap(QPixmap.fromImage(showImage).scaled(self.ui.label_2.width(), self.ui.label_2.height()))
            time_start=time.time()
            while (True):
                ret, frame = cap.read()  # 读取一帧的图像
                if ret:  # 保证摄像头有读取到图片时才进行下列操作
                    frame = cv2.flip(frame,1)
                    faces = self.face_cascade.detectMultiScale(frame, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))  # 检测人脸
                    if len(faces) > 0:  # 有人脸时再执行下列操作
                        with torch.no_grad():
                            for (x, y, w, h) in faces:
                                face = frame[y:y + h, x:x + w, 0]  # 裁剪出人脸,注意，这里x和y反过来了
                                face = cv2.resize(face, (48, 48))
                                face = np.reshape(face, (1, 1, 48, 48))  # 转成网络满足的格式
                                input = torch.tensor(face / 255, dtype=torch.float32)
                                output = np.argmax(self.net(input).numpy(), axis=1)[0]
                                # print(EXPRESSIONS[output])
                                # print(x,y,w,h)
                                # plt.imshow(face,cmap='gray')
                                # plt.pause(0.01)
                                # print(face.shape)
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用矩形圈出人脸
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                cv2.putText(frame, EXPRESSIONS[int(output)], (x, y), fontFace=font, fontScale=1.0, color=(0, 0, 255),
                                        thickness=2)
                    #cv2.imshow('Face Recognition', frame)    # end(if ret)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    heigt, width = frame.shape[:2]
                    pixmap = QImage(frame, width, heigt, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(pixmap).scaled(self.ui.label.width(), self.ui.label.height())
                    self.ui.label.setPixmap(pixmap)
                time_end=time.time()
                t=time_end-time_start
                if cv2.waitKey(50) == ord('q') or t>tim or int(output)==n:
                    if int(output)==n:
                        st=EXPRESSIONS[n]+'识别成功，得分+1'
                        playsound(os.path.split(os.path.realpath(__file__))[0]+'/music/yes.mp3')
                        score+=1
                        self.ui.textEdit.append(st)
                        self.ui.textEdit.moveCursor(QTextCursor.End)
                    if t>tim:
                        st=EXPRESSIONS[n]+'识别失败'
                        playsound(os.path.split(os.path.realpath(__file__))[0]+'/music/no.mp3')
                        self.ui.textEdit.append(st)
                        self.ui.textEdit.moveCursor(QTextCursor.End)
                    break
                #end(while)
            
        cap.release()  # 释放摄像头
        cv2.destroyAllWindows()
        st='总得分：'+str(score)
        self.ui.textEdit.append(st)
        self.ui.textEdit.moveCursor(QTextCursor.End)
    def check(self):
        global tim
        if  self.ui.checkBox.isChecked():
            tim=10
        if  self.ui.checkBox_2.isChecked():
            tim=5
        if  self.ui.checkBox_3.isChecked():
            tim=3
    def close(self):
        self.ui.close()

if __name__ == '__main__':
    app = QApplication([])
    app.setWindowIcon(QIcon(os.path.split(os.path.realpath(__file__))[0]+'/icon/brain.ico'))
    window= Window()
    app.exec_()