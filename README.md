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
2. **Image Capturing**: When a person is detected, the camera captures frames at set intervals.
3. **Preprocessing**: The captured frame undergoes preprocessing to isolate the individual and their immediate surroundings.
4. **File Transfer**: Preprocessed images and a "mode.txt" file are sent to the server.

### Abnormality Detection

1. **VQA Session**: The system utilizes a set of predefined questions to identify early signs of abnormalities.
2. **User-Model Interaction**: If abnormalities are detected, the system engages in real-time interactions to gather detailed responses.
3. **Decision Making**: Based on the contextual information, the system decides on actions such as calling an ambulance.

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
