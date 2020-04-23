# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:37:53 2020

@author: Rodrigo
"""

import streamlit as st
import face_recognition
import cv2
from PIL import Image

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

    
def images(filename,filename1):   
    image = face_recognition.load_image_file(filename)
    image1 = face_recognition.load_image_file(filename1)
    known_face_encoding = face_recognition.face_encodings(image)
    unknown_face_encodings = face_recognition.face_encodings(image1)
    
    facematch = False

    if known_face_encoding and unknown_face_encodings:
        match_results = face_recognition.compare_faces([known_face_encoding[0]], unknown_face_encodings[0])
        if match_results[0]:
            facematch = True
    
        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"
    
        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
    
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"
    
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
        genderList = ['Male', 'Female']
    
        # Load network
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)
        faceNet = cv2.dnn.readNet(faceModel, faceProto)
    
        # Open a video file or an image file or a camera stream
        
        
        
        cap = face_recognition.load_image_file(filename1)
        
        
        padding = 20
    
    
        frameFace, bboxes = getFaceBox(faceNet, cap)
        if not bboxes:
            st.write("No face Detected, Checking next frame")
            
        for bbox in bboxes:
            # print(bbox)
            face = cap[max(0,bbox[1]-padding):min(bbox[3]+padding,cap.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, cap.shape[1]-1)]
    
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
    
            if facematch == True:
                st.markdown("As **fotos são da mesma** pessoa :sunglasses:")
                if gender=='Male':
                    st.write("Existe uma probabilidade de",100*genderPreds[0].max(),"% do gênero ser masculino")
                else:
                    st.write("Existe uma probabilidade de",100*genderPreds[0].max(),"% do gênero ser feminino")
                st.write(st.write("Existe uma probabilidade de",100*agePreds[0].max(),"% da idade da pessoa estar no intervalo", age))
                st.balloons()
            else:
               st.markdown("As **fotos não são da mesma** pessoa :confused:") 
            
    else:
        st.markdown("Ei! coloca uma foto com um rosto aí :rage:")
 
st.title('Face Match')
image = Image.open('machine-learning.jpg')
st.image(image, use_column_width=True)

st.markdown('''No processo de confirmação de identidade, Face Match é um recurso cada vez mais utilizado pelas empresas por oferecer mais agilidade através da tecnologia. Nessa demonstração, vamos mostrar um exemplo de face match e dois modelos um que estima o gênero e outro que estima a idade da pessoa da foto. 

## O que é Face Match?

De maneira simples e direta, Face Match é a tecnologia de reconhecimento facial que utiliza inteligência artificial. Muitas vezes tido como um assunto de ficção científica, hoje já é utilizado por diversas áreas e mecanismos, como equipamentos de segurança, smartphones, caixas eletrônicos, entre outros. 

Por vezes, Face Match é associada à ideia de excesso de vigilância e a coleta exagerada de informações pessoais. No entanto, seu trabalho é muito mais dar proteção para o usuário e evitar a ações intrusivas. 

Além disso, é muito mais prático utilizar essa ferramenta para confirmar sua identidade. O usuário irá poupar o tempo de digitar uma senha para desbloquear o seu smartphone ou até mesmo para realizar um saque com um caixa eletrônico. Além disso, simplifica o processo de confirmação de identidade para abrir uma conta, por exemplo. confira o código dessa aplicação em https://github.com/rasiqueira/face_recognition

## Como Face Match funciona?

Face Match é uma tecnologia complexa e é preciso entender seu funcionamento para aplicar valor ao seu negócio e para seu cliente.

Essa técnica de identificação biométrica, assim como a impressão digital, mapeia as características faciais de uma pessoa. Através de um conjunto complexo de algoritmos, compara com imagens de banco de dados ou especificamente de uma imagem digital da mesma pessoa para reconhecer sua identidade (ou negar). 

Esses algoritmos dividem uma imagem em diversos pontos e pixels, mapeando o rosto em busca dos pontos nodais – características de marca a distinção de uma pessoa para outra. Existem cerca de 80 desses pontos na face. Alguns deles são:

* formato do queixo,
* comprimento da linha da mandíbula,
* distância entre os olhos,
* profundidade das órbitas oculares,
* largura do nariz,
* entre tantos outros.''')
st.header('Demo de face match, estimativa de gênero e idade')
st.subheader('documento')
file1 = st.file_uploader("faça o upload de uma foto do seu documento", type=['pgn','jpg','jpge'])

st.subheader('selfie')
file2 = st.file_uploader("faça o upload de uma selfie", type=['pgn','jpg','jpge'])

if file1 and file2:
    images(file1,file2)

    