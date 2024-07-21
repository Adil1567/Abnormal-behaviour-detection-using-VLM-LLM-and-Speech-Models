# Abnormal-behaviour-detection-using-VLM-LLM-and-Speech-Models
This project is aimed at detecting the abnormal behaviour or emergency cases using vision-language model (VLM), large language model (LLM), human detection model, text-to-speech (TTS) and speech-to-text models (STT).  The  framework can detect the subtle sings of emergency and actively interact with the user to make an accurate decision.

# Multi-Modal System for Real-Time Emergency Response

This repository contains the code and instructions necessary to replicate the multi-modal system for real-time emergency response, as described in our research.

## Overview

The system integrates various technologies, including Vision-Language Models (VLM), Large Language Models (LLM), Text-to-Speech (TTS), and Speech-to-Text (STT) models. It is designed for continuous monitoring, emergency detection, and real-time interaction with users.

## Key Components

### Hardware

- **NVIDIA DGX V100 Server**: Used for running the LLaVA model and data storage (can be run on one NVIDIA V100 GPU).
- **Intel RealSense d455 Camera**: Positioned strategically for extensive coverage of the living space, connected to a laptop.
- **Rode Wireless GO Microphones**: Worn by participants for precise audio capture during interactions.
- **Local Laptop**: Equipped with an audio speaker, this laptop facilitates remote server access, handles image and text file transfers, and runs Piper TTS, human detection model, and Whisper STT models.

### Models

- **YOLOv8**: A deep learning model for human detection, monitoring camera footage to identify individuals in the living space.
- **LLaVA (Large Language-and-Vision Assistant)**: Central to identifying emergencies, generating context-specific questions, and analyzing responses.
- **Piper TTS**: Converts LLaVA’s questions into natural voice prompts.
- **Whisper STT**: Transcribes user responses accurately to facilitate analysis.

## System Workflow

### Continuous Monitoring

1. **Human Detection**: YOLOv8 continuously monitors the camera footage.
2. **Image Capturing**: When a person is detected, the camera captures frames and the goes to the preporocessing, file transfer and LLaVA model initialization.
3. **Preprocessing**: *Preprocessing**: The captured frame undergoes preprocessing, which includes:
   - Calculating the bounding boxes around the detected person.
   - Expanding the bounding boxes to capture nearby objects.
   - Cropping the image to focus on the person and the surrounding objects for zooming and better model comprehension.
4. **File Transfer**: Preprocessed images and a "mode.txt" (for initialization) file are sent to the server.

### Abnormality Detection and Thresholding

1. **VQA Session**: The system utilizes a set of predefined questions to identify early signs of abnormalities.
2. **Thresholding**: Collected user responses are tallied to generate an abnormality score, representing the count of "Yes" answers. If this score surpasses a predetermined threshold (the non-anomalous action threshold), the system initiates real-time dialogue with the user.
3. **User Confirmation**: Before activating the user-model interaction block, the system requests the user to confirm the need for further interaction. If the user affirms assistance is required ("Yes" response), the system updates the "mode.txt" file to 1, signaling the activation of the interaction block. If the user remains unresponsive, the system automatically triggers the interaction block.

### User-Model Interaction Block

1. **Engagement**: The system engages directly with the individual, posing questions tailored to the visual context and preserving the interaction history.
2. **Question Generation**: LLaVA dynamically generates a set of questions tailored to the user's visual context and prior responses. These questions are stored in a "question.txt" file and sent to the local part for processing by speech models.
3. **Activation of TTS and STT**: 
   - **TTS**: Piper TTS audibly presents the questions to the user.
   - **STT**: Whisper STT captures and transcribes the user's responses into the "answers.txt" file on the local side. Whisper STT halts recording after three seconds of silence, which is customizable to manage conversational pauses.
4. **Interaction Analysis**: The "answers.txt" file is transmitted back to the server for analysis by the LLaVA model. Based on the user's responses and previous image embeddings, the system gauges the severity of the situation.

### Decision-Making Process

1. **Emergency Determination**: If the "answers.txt" file is empty, indicating no response from the individual, the system interprets it as unresponsiveness and triggers an immediate ambulance call. If responses are present, LLaVA scrutinizes them alongside historical data and image analysis to ascertain the necessity for emergency services.
2. **Iterative Process**: The iterative process of generating questions and analyzing user responses continues until the LLaVA model determines whether to call an ambulance.
3. **Final Decision**: Upon detecting an emergency, the model summarizes the situation, provides recommendations for the user, and forwards relevant information to medical professionals or caregivers for further assistance.

## Installation and Setup

### Prerequisites

- **NVIDIA DGX V100 Server**
- **Intel RealSense d455 Camera**
- **Rode Wireless GO Microphones**
- **Python 3.x**
- **CUDA 11.x**
- **PyTorch**
- **YOLOv8**
- **LLaVA**
- **Piper TTS**
- **Whisper STT**

### Steps

1. **Clone the Repository**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
