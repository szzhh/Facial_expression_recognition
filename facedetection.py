'''
    日期：2021/4/21

    人脸检测加
    首先由opencv的库框出人脸，会返回位置和大小，从原图截取这部分图片，转化为神经网络需要的格式，输入神经网络，得到表情识别结果

'''
import resnet
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt


EXPRESSIONS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'normal']
# print(cv2.__file__)



def expressionRecognition(net):     # 表情识别
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')  # 加载人脸特征库
    while (True):
        ret, frame = cap.read()  # 读取一帧的图像
        if ret:  # 保证摄像头有读取到图片时才进行下列操作
            frame = cv2.flip(frame,1)
            faces = face_cascade.detectMultiScale(frame, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))  # 检测人脸
            if len(faces) > 0:  # 有人脸时再执行下列操作
                with torch.no_grad():
                    for (x, y, w, h) in faces:
                        face = frame[y:y + h, x:x + w, 0]  # 裁剪出人脸,注意，这里x和y反过来了
                        face = cv2.resize(face, (48, 48))
                        face = np.reshape(face, (1, 1, 48, 48))  # 转成网络满足的格式
                        input = torch.tensor(face / 255, dtype=torch.float32).cuda()
                        output = np.argmax(net(input).cpu().numpy(), axis=1)[0]
                        # print(EXPRESSIONS[output])
                        # print(x,y,w,h)
                        # plt.imshow(face,cmap='gray')
                        # plt.pause(0.01)
                        # print(face.shape)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 用矩形圈出人脸
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, EXPRESSIONS[output], (x, y), fontFace=font, fontScale=1.0, color=(0, 0, 255),
                                thickness=2)
            cv2.imshow('Face Recognition', frame)    # end(if ret)
        if cv2.waitKey(1) == ord('q'):
            break
        # end(while)
    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()

def main():
    net = torch.load('fer2013_resnet_lr_3.pkl')
    expressionRecognition(net)


if __name__ == '__main__':
    main()