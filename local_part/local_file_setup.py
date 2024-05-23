#!/usr/bin/env python
# coding: utf-8

from ultralytics import YOLO
import cv2
import math
import os
import json
import argparse
import queue
import sys
import sounddevice as sd
import threading
from datetime import datetime
from playsound import playsound
import winsound
from whisper_mic import WhisperMic
import torch
import subprocess
import mutagen 
from mutagen.wave import WAVE
from time import time, sleep
import paramiko
import smtplib
from email.message import EmailMessage
# def email_alert(subject,body,to):
#     msg= EmailMessage()
#     msg.set_content(body)
import os
import ssl
from smtplib import SMTP_SSL
def email_alert(body):
    email_sender="adil70722156@gmail.com"
    email_password='tozc rpif wmir ztpe'
    email_receiver='adil.zhiyenbayev@nu.edu.kz'
    subject='We need an ambulance!'
    body = "It was determined that the person is in the emergency situation and requires immediate attention. Here is the suggestion from the model:\n" + body
    em= EmailMessage()
    em["From"]=email_sender
    em["To"]=email_receiver
    em["Subject"]=subject
    em.set_content(body)

    context=ssl.create_default_context()
    with SMTP_SSL("smtp.gmail.com",465,context=context) as smtp:
        smtp.login(email_sender,email_password)
        smtp.sendmail(email_sender,email_receiver,em.as_string())



def create_ssh_client(server, port, user, key_file):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, username=user, key_filename=key_file)
    return client

    # Function to execute a command on the remote server
def execute_command(ssh_client, command):
    stdin, stdout, stderr = ssh_client.exec_command(command)
    return stdout.read().decode(), stderr.read().decode()
gateway = create_ssh_client('87.255.216.119', 11223, 'adil_zhiyenbayev', 'C:/Users/Adil/.ssh/id_rsa')
transport = gateway.get_transport()
dest_addr = ('10.10.25.12', 22)
local_addr = ('127.0.0.1', 222)
channel = transport.open_channel("direct-tcpip", dest_addr, local_addr)
remote_client = paramiko.SSHClient()
remote_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
remote_client.connect('10.10.25.12', username='adil_zhiyenbayev', sock=channel)

class VisualAudioAssistant:
    def __init__(self):
        self.model = YOLO("yolo-Weights/yolov8n.pt")
        self.mic = WhisperMic(model='small', english=True, device='cuda', save_file=False, pause=3, mic_index=0)
        self.classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
#         self.gateway = create_ssh_client('87.255.216.119', 11223, 'adil_zhiyenbayev', 'C:/Users/Adil/.ssh/id_rsa')

#         self.transport = gateway.get_transport()
#         self.dest_addr = ('10.10.25.12', 22)
#         self.local_addr = ('127.0.0.1', 222)
#         self.channel = self.transport.open_channel("direct-tcpip", self.dest_addr, self.local_addr)

#         self.remote_client = paramiko.SSHClient()
#         self.remote_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#         self.remote_client.connect('10.10.25.12', username='adil_zhiyenbayev', sock=self.channel)

        # self.setup_dir = "C:/Users/adil/SETUP_UPDATED/"
        # self.remote_server = "adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/"

    # def delete_files(self, filenames):
    #     for filename in filenames:
    #         file_path = os.path.join(self.setup_dir, filename)
    #         if os.path.isfile(file_path):
    #             os.remove(file_path)
    # Function to create connection
#     def create_ssh_client(self,server, port, user, key_file):
#         client = paramiko.SSHClient()
#         client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#         client.connect(server, port, username=user, key_filename=key_file)
#         return client

#     # Function to execute a command on the remote server
#     def execute_command(self,ssh_client, command):
#         stdin, stdout, stderr = ssh_client.exec_command(command)
#         return stdout.read().decode(), stderr.read().decode()

    def mode(self):
        start=time()
        with open('C:/Users/adil/SETUP_UPDATED/mode.txt', 'w') as f: #open the file
                    f.write('0')
        os.system(r"scp .\mode.txt adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/") 
        end=time()
        print("Time for sending the mode.txt: ",end-start)

    def vqa(self):
        #sleep(20)
        #while not os.path.isfile("C:/Users/adil/SETUP_UPDATED/vqa.txt"):
            #print("Trying to find VQA")
            #sleep(1)
        
        if os.path.isfile("C:/Users/Adil/SETUP_UPDATED/vqa.txt"):
            os.remove("C:/Users/Adil/SETUP_UPDATED/vqa.txt")
        while True:
            directory_to_list = '/raid/adil_zhiyenbayev/'
            output, error = execute_command(remote_client, f'ls -l {directory_to_list}')
            #print(output)
            if 'vqa.txt' in output and 'vqa_tmp.txt' in output:
                #sleep(2)
                start=time()
                sftp = remote_client.open_sftp()
                remote_file_path = '/raid/adil_zhiyenbayev/vqa_tmp.txt'
                local_file_path = 'C:/Users/Adil/SETUP_UPDATED/vqa.txt'
                sftp.get(remote_file_path, local_file_path)
                end=time()
                print("Time to send vqa.txt to local: ",end-start)
                sftp.remove("/raid/adil_zhiyenbayev/vqa_tmp.txt")
                sftp.remove("/raid/adil_zhiyenbayev/vqa.txt")
                sftp.close()
                print("Yes")
                
                break
            else:
                print("No")
        #sleep(1)
#         sftp = remote_client.open_sftp()
#         remote_file_path = '/raid/adil_zhiyenbayev/vqa.txt'
#         local_file_path = 'C:/Users/Adil/vqa.txt'
#         sftp.get(remote_file_path, local_file_path)
#         sftp.close()
        #if os.path.isfile("adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/vqa.txt"):
        #os.system(r"scp adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/vqa.txt C:\Users\adil\SETUP_UPDATED")
        sleep(1)
        # os.system(r"scp adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/vqa.txt C:\Users\adil\SETUP_UPDATED")
        start=time()
        with open('C:/Users/Adil/SETUP_UPDATED/vqa.txt') as f:
            st = f.read()
            print(st.count('Yes'))
        if (st.count('Yes')) > 0:
            check = "Dear user, please, confirm that you need help. Say 'Yes' if you need any help and 'No' if you do not."
            os.system("echo " + check + " | .\piper.exe -m .\en_US-ryan-medium.onnx -f check.wav")
            audio_check = WAVE(r"check.wav") 
            audio_check_info = audio_check.info 
            length_aud_check = int(audio_check_info.length)
            winsound.PlaySound(r'check.wav', winsound.SND_ASYNC)
            sleep(length_aud_check + 1)
            end=time()
            print("time to voice the audio for help : ",end-start)
            aud_beep = WAVE(r"beep.wav")
            aud_beep_info = aud_beep.info
            len_aud_beep = int(aud_beep_info.length)
            winsound.PlaySound(r'beep.wav', winsound.SND_ASYNC)
            sleep(len_aud_beep)
            start=time()
            result_check = self.mic.listen(3)
            end=time()
            print("time to say yes i need help: ", end-start)
            winsound.PlaySound(r'beep.wav', winsound.SND_ASYNC)
            sleep(len_aud_beep)
            print(result_check)
            winsound.PlaySound(r'beep.wav', winsound.SND_ASYNC)
            os.remove("check.wav")
            start=time()
            if result_check.lower().count('yes')>0 or result_check.count('Timeout')>0 or result_check=="":

                with open('mode.txt', 'w') as f: #open the file
                    f.write('1')
                os.system(r"scp mode.txt adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/")
                end=time()
                print("time for sending the mode.txt with 1 to server: ",end-start)
                os.remove("vqa.txt")
                os.remove("mode.txt")
                return True
            # with open('mode.txt', 'w') as f: #open the file
            #     f.write('1')
            # os.system(r"scp .\mode.txt adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/")
            # os.remove("vqa.txt")
            # # print("Return True")
            # return True
        os.remove("vqa.txt")
        return False

    def voicing(self):
        for i in range(5):
                if i>3:
                    #while not os.path.isfile("C:/Users/adil/SETUP_UPDATED/suggestion.txt"):
                    while True:
                        directory_to_list = '/raid/adil_zhiyenbayev/'
                        output, error = execute_command(remote_client, f'ls -l {directory_to_list}')
                        
#                         if 'suggestion.txt' in output:
#                             print("Yes")
#                             break
                        yes_mode = 0
                        #print(i)
                        if 'yes.txt' in output:
                            yes_mode = 1
                        if 'suggestion.txt' in output and 'suggestion_tmp.txt' in output:

                            sftp = remote_client.open_sftp()
                            remote_file_path = '/raid/adil_zhiyenbayev/suggestion_tmp.txt'
                            local_file_path = 'C:/Users/Adil/SETUP_UPDATED/suggestion.txt'
                            sftp.get(remote_file_path, local_file_path)
                            sftp.close()
                            print("Yes")
                            
                            break
                        else:
                            print("No")
#             else:
#                 print("No")
                #os.system(r"scp adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/suggestion.txt C:\Users\adil\SETUP_UPDATED")
                else:
                    questname = "question" + str(i + 1)
                    #while not os.path.isfile("C:/Users/adil/SETUP_UPDATED/question.txt"):
                    while True:
                        directory_to_list2 = '/raid/adil_zhiyenbayev/'
                        output2, error2 = execute_command(remote_client, f'ls -l {directory_to_list2}')
                        checksame = ''
                        # if os.path.isfile("C:/Users/adil/SETUP_UPDATED/question.txt"):
                        #     with open("C:/Users/adil/SETUP_UPDATED/question.txt", "r") as f:
                        #         checksame = f.read()
                        #     if os.path.isfile("C:/Users/adil/SETUP_UPDATED/question.txt"):
                        #         os.remove("C:/Users/adil/SETUP_UPDATED/question.txt")
                        if questname + '.txt' in output2:
                            start=time()
                            sftp = remote_client.open_sftp()
                            remote_file_path = '/raid/adil_zhiyenbayev/' + questname + '_tmp.txt'
                            local_file_path = 'C:/Users/Adil/SETUP_UPDATED/' + questname + '.txt'
                            sftp.get(remote_file_path, local_file_path)
                            end=time()
                            print("time for sending question.txt: ",end-start)
                            with open("C:/Users/adil/SETUP_UPDATED/" + questname + ".txt","r") as f2:
                                checkq = f2.read()
                            if checkq == checksame:
                                continue
                            sftp.close()
                            print("Yes")
                            break
                        else:
                            print("No")
                        #print(i)
                    #os.system(r"scp adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/question.txt C:\Users\adil\SETUP_UPDATED")
                
                if i<4:

                    with open(questname + '.txt') as f:
                        line = f.read()
                else:
                    with open('suggestion.txt') as f:
                        line = f.read()
                    if yes_mode == 1:
                        line2 = 'The ambulance is called and will arrive soon'
                        start=time()
                        os.system("echo " + line2 + " | .\piper.exe -m .\en_US-ryan-medium.onnx -f yes.wav")
                        audio2 = WAVE(r"yes.wav") 
                        audio_info2 = audio.info 
                        length_aud2 = int(audio_info.length)
                        winsound.PlaySound(r'yes.wav', winsound.SND_ASYNC)
                        sleep(length_aud2)
                        end=time()
                        print("time for vocing the decision of the model: ",end-start)
                        email_alert(line)
                        #output2, error2 = execute_command(remote_client, f'ls -l {directory_to_list}')

                    

                
                
                
                line = line.replace('</s>', '')
                print(line)
                f.close()
                start=time()
                os.system("echo " + line + " | .\piper.exe -m .\en_US-ryan-medium.onnx -f test1.wav")
                winsound.PlaySound(r'test1.wav', winsound.SND_ASYNC)
                audio = WAVE(r"test1.wav") 
                audio_info = audio.info 
                length_aud = int(audio_info.length)
                #os.remove(questname + '.txt')
                if os.path.isfile("C:/Users/adil/SETUP_UPDATED/suggestion.txt"):
                    os.remove("C:/Users/adil/SETUP_UPDATED/suggestion.txt")
                sleep(length_aud+1)
                end=time()
                print("Time for voicng the question or suggestion: ",end-start)
                if i < 4:
                    aud_beep = WAVE(r"beep.wav")
                    aud_beep_info = aud_beep.info
                    len_aud_beep = int(aud_beep_info.length)
                    winsound.PlaySound(r'beep.wav', winsound.SND_ASYNC)
                    sleep(len_aud_beep)
                    start=time()
                    result = self.mic.listen(3)
                    end=time()
                    print("Time for listening the user: ",end-start)
                    print(result)
                    winsound.PlaySound(r'beep.wav', winsound.SND_ASYNC)
                    sleep(len_aud_beep)
                    winsound.PlaySound(r'beep.wav', winsound.SND_ASYNC)
                    with open('answer.txt', 'w') as f: #open the file
                        f.write(result)
                    f.close()
                    start=time()
                    os.system('scp answer.txt adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/')
                    end=time()
                    print("time for sending answer.txt to server: ",end-start)
                if os.path.exists("C:/Users/adil/SETUP_UPDATED/test1.wav"):
                    os.remove('test1.wav')
                # if i >= 4:
                #     output2, error2 = execute_command(remote_client, f'ls -l {directory_to_list}')
                #     if 'yes.txt' in output2:
                #         line2 = 'The ambulance is called and will arrive soon'
                #         os.system("echo " + line2 + " | .\piper.exe -m .\en_US-ryan-medium.onnx -f yes.wav")
                #         winsound.PlaySound(r'yes.wav', winsound.SND_ASYNC)
                #         email_alert(line)
                
    def process_detection(self,results,img):
        """
        Process each detection and perform actions like drawing bounding box, checking confidence, etc.
        """
        #crop = results.crop(save=True)
        # coordinates
        start=time()
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
            # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values
                # put box in cam
                #cv2.rectangle(img, (y1, y2), (x2, x1), (255, 0, 255), 3)
                # class name
                cls = int(box.cls[0])
                if self.classNames[cls] == 'person':
                    
                    # confidence
                    print(x1-10, x2+10, y1-10, y2+10)
                    confidence = math.ceil((box.conf[0]*100))/100
                    print(confidence)
                    if confidence <= 0.8:
                        continue
                    end=time()
                    print("Time for the model to detect the person: ",end-start)
                    #print(x1, x2, y1, y2)
                    start=time()
                    img_height, img_width = img.shape[:2]

                    # Adjust bounding box coordinates to ensure they are within image boundaries
                    x1 = max(0, x1 - 50)
                    y1 = max(0, y1 - 50)
                    x2 = min(img_width, x2 + 50)
                    y2 = min(img_height, y2 + 50)

                    cropped_img = img[y1:y2, x1:x2]
                    cv2.imwrite('C:/Users/adil/SETUP_UPDATED/help_image.jpg', cropped_img)
                    end=time()
                    print("Time to crop the image:  ",end-start)
                    #cv2.imwrite('C:/Users/adil/SETUP_UPDATED/help_image.jpg', img)
                    print('YOLO Person is detected ==> Saving the frame')
                    start=time()
                    os.system('scp C:/Users/adil/SETUP_UPDATED/help_image.jpg adil_zhiyenbayev@remote_server:/raid/adil_zhiyenbayev/')
                    end=time()
                    print("time to send the image to the server: ", end-start)
                    print("SAVED!")
                    return True
        return False

       
    # print("CHECK FOR IMAGE")
    # if(classNames[cls]!='person'):
    #     print("NOT a PERSON")
    #     return False
        

    # if confidence<=0.85:
    #     print("CONFIDENCE is SMALL")
    #     return False
    # return True

    def cleanup(self):
        for i in range(4):
            if os.path.exists("C:/Users/adil/SETUP_UPDATED/question" + str(i+1) + ".txt"):
                os.remove("C:/Users/adil/SETUP_UPDATED/question" + str(i+1) + ".txt")
        # Delete the file if it exists
        if os.path.exists("C:/Users/adil/SETUP_UPDATED/decision.txt"):
            os.remove("C:/Users/adil/SETUP_UPDATED/decision.txt")
        if os.path.exists("C:/Users/adil/SETUP_UPDATED/question.txt"):
            os.remove("C:/Users/adil/SETUP_UPDATED/question.txt")
        if os.path.exists("C:/Users/adil/SETUP_UPDATED/help_image.jpg"):
            os.remove("C:/Users/adil/SETUP_UPDATED/help_image.jpg")
        if os.path.exists("C:/Users/adil/SETUP_UPDATED/test1.wav"):
            os.remove("C:/Users/adil/SETUP_UPDATED/test1.wav")
        if os.path.exists("C:/Users/adil/SETUP_UPDATED/vqa.txt"):
            os.remove("C:/Users/adil/SETUP_UPDATED/vqa.txt")
        if os.path.exists("C:/Users/adil/SETUP_UPDATED/answer.txt"):
            os.remove("C:/Users/adil/SETUP_UPDATED/answer.txt")
        if os.path.isfile("C:/Users/adil/SETUP_UPDATED/suggestion.txt"):
            os.remove("C:/Users/adil/SETUP_UPDATED/suggestion.txt")
        if os.path.exists("C:/Users/adil/SETUP_UPDATED/yes.wav"):
            os.remove("C:/Users/adil/SETUP_UPDATED/yes.wav")

    def main_loop(self):
        
        cap = cv2.VideoCapture(0)
        try:
            while True:
                start=time()
                success, img = cap.read() 
                end=time()
                print("To take a photo: ",end-start)
                #img=cv2.imread("C:/Users/Adil/Downloads/ex_sit.jpg")   
                results = self.model(img, stream=True)
                # Results
                #crops = results.crop(save=True) 
                
                if self.process_detection(results,img)==False:
                    continue
                self.mode()
                if self.vqa()==False:
                    continue
                self.voicing()
                self.cleanup()
        except KeyboardInterrupt:
            print("KeyboardInterrupt: Cleaning up...")
            self.cleanup()
                
    
if __name__ == "__main__":
    assistant = VisualAudioAssistant()
    startstr = input("enter")
    if startstr == "":
        assistant.main_loop()
    #assistant.main_loop()



