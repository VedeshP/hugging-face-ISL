from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "cdsteameight/ISL-SignLanguageTranslation"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

import cv2

def extract_frames(video_path, frame_rate=1):
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        # Capture frames at the specified frame rate
        if i % frame_rate == 0:
            frames.append(frame)
    
    cap.release()
    return frames

import torch
from torchvision import transforms

def preprocess_frames(frames):
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  # Resize to the input size expected by the model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    
    tensor_frames = [preprocess(frame) for frame in frames]
    return torch.stack(tensor_frames)  # Stack into a single tensor

def translate_sign_language(video_path):
    frames = extract_frames(video_path)
    preprocessed_frames = preprocess_frames(frames)
    
    # Add a batch dimension
    input_tensor = preprocessed_frames.unsqueeze(0)  # Shape: (1, num_frames, 3, 224, 224)
    
    with torch.no_grad():
        outputs = model.generate(input_tensor)
    
    # Decode the output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

video_path = "path_to_your_video.mp4"
translated_output = translate_sign_language(video_path)
print(translated_output)