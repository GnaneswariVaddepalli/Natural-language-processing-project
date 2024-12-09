# Image Captioning with Visual Attention


## Project Overview


Image Captioning with Visual Attention is a deep learning project designed to generate descriptive textual captions for images. By leveraging a visual attention mechanism, the model dynamically focuses on salient regions of an image while generating captions, resulting in more contextually accurate and relevant descriptions. This project incorporates a ResNet50-based encoder, an LSTM-based decoder, and an interactive Streamlit application for real-time caption generation.


## Features

Attention Mechanism: Enhances caption generation by focusing on critical image regions dynamically.
Encoder-Decoder Architecture:
Encoder: Pre-trained ResNet50 extracts high-level image features.
Decoder: LSTM generates captions word-by-word using attention-guided context vectors.
Streamlit Application: Provides an intuitive interface for real-time image captioning.
Evaluation Metrics: BLEU Score, F1-Score, Accuracy, and Loss.


## Installation and Setup

### 1. Clone the Repository

Clone the repository to your local machine or directly download the project files:

bash
git clone https://github.com/yourusername/ImageCaptioning.git
cd ImageCaptioning


### 2. Install Dependencies

Install the required Python libraries using pip:

bash
pip install -r requirements.txt

### 3. Set Up the Dataset

Download the Flickr8k dataset and organize it as follows:

flickr8k/
├── Flickr8k_Subset/
│   ├── Images/
│   └── captions.txt

### 4. Pre-Trained Model

Download the pre-trained model image_captioning_model.pth and place it in the model/ directory.

### 5. Launch the Application
Run the Streamlit application for real-time caption generation:

streamlit run streamlit_app.py

Upload an image to generate a caption in real-time.








