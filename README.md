# Vision to Voice

An assistive system that turns images into spoken descriptions in multiple languages, built to help visually impaired users perceive their surroundings.

## What this is

This was my final year capstone project at VIT (B.Tech CSE, AI & Robotics). The idea is straightforward: a visually impaired person points a camera at something, and the system describes what it sees out loud, in their preferred language.

The full hardware version uses ultrasonic sensors to trigger image capture when an obstacle is detected. This repo contains the software pipeline and web interface. You can upload an image or capture one from your webcam, and the system will generate a spoken description in both English and Tamil (or any other language supported by M2M100).

I co-authored an IEEE format research paper on this project with my faculty supervisor.

## How it works

The pipeline chains four models together:

1. **Scene captioning** with BLIP2 FLAN T5 XL (Salesforce). Takes the image and generates a natural language description using beam search with repetition penalty.

2. **Face aware captioning**. If the scene caption mentions a person ("man", "woman", "child"), a fine tuned BLIP model (trained on a Kaggle faces dataset with AdamW, lr 1e-5, cross entropy loss) generates a more specific caption. It only replaces the original if confidence exceeds 90%.

3. **Translation** with Facebook's M2M100 (418M parameters). Translates the English caption to the target language. Defaults to Tamil but supports any M2M100 language pair.

4. **Text to speech** with Google TTS. Generates audio files for both the English and translated captions.

The web interface is a Flask app. Upload an image or use the webcam capture button, and you get back the image alongside both captions with playable audio.

## Stack

Python · PyTorch · HuggingFace Transformers · BLIP2 · FLAN T5 · M2M100 · gTTS · Flask · PIL

## How to run

```bash
git clone https://github.com/kautum/vision-to-voice.git
cd vision-to-voice

# Install dependencies
pip install torch transformers Pillow flask gtts huggingface-hub

# Run the web app
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

Note: first run will download several large models (BLIP2 FLAN T5 XL is ~15GB, M2M100 is ~1.8GB). You'll need a machine with at least 16GB RAM. GPU is optional but recommended.

The fine tuned face model weights are downloaded automatically from HuggingFace Hub (`Duke29/Face_Finetuned_Salesforce_Blip_Image_Captioning_Base`).
