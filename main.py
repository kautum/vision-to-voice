import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from gtts import gTTS
import os
from PIL import Image
from huggingface_hub import hf_hub_download

# Global Model Setup

# BLIP2 for general image captioning
blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
blip2_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")


# BLIP for face detection captioning
face_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
face_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
face_model.load_state_dict(torch.load(hf_hub_download(
    repo_id="Duke29/Face_Finetuned_Salesforce_Blip_Image_Captioning_Base",
    filename="finetuned_blip_captioning.pth"), map_location=device))
face_model.eval()


transmodel_name = "facebook/m2m100_418M"
transtokenizer = AutoTokenizer.from_pretrained(transmodel_name)
transmodel = AutoModelForSeq2SeqLM.from_pretrained(transmodel_name)



# Directory for saving audio files
AUDIO_FOLDER = os.path.join("static", "audio")
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Function Definitions

def generate_caption(image):
    image = image.convert("RGB")
    text_prompt = "Describe the content in the image."
    inputs = blip2_processor(images=image, text=text_prompt, return_tensors="pt")
    outputs = blip2_model.generate(
        **inputs,
        max_length=300,
        num_beams=5,
        length_penalty=2.0,
        repetition_penalty=2.0
    )
    caption = blip2_processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return caption

def generate_caption_for_face(image):
    image = image.convert("RGB")
    inputs = face_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = face_model.generate(**inputs, output_scores=True, return_dict_in_generate=True)
    caption = face_processor.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
    confidence_score = torch.max(probs).item() * 100
    return caption, confidence_score

def translate_caption(text, target_language="ta"):

    transtokenizer.src_lang = "en"
    inputs = transtokenizer(text, return_tensors="pt")

    # Generate translation
    output = transmodel.generate(**inputs, forced_bos_token_id=transtokenizer.lang_code_to_id[target_language])

    # Decode and return
    translated_text = transtokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return translated_text
def text_to_speech(text, filename, language="en"):
    tts = gTTS(text=text, lang=language)
    audio_path = os.path.join(AUDIO_FOLDER, filename)
    tts.save(audio_path)
    return audio_path

def process_image(image, target_language="ta"):
    caption = generate_caption(image)
    if any(word in caption.lower() for word in ["man", "woman", "child"]):
        custom_caption, confidence_score = generate_caption_for_face(image)
        if confidence_score >= 90:
            caption = custom_caption
    translated_caption = translate_caption(caption, target_language)
    audio_path_en = text_to_speech(caption, "caption_en.mp3", language="en")
    audio_path_translated = text_to_speech(translated_caption, "caption_translated.mp3", language=target_language)
    return caption, translated_caption, audio_path_en, audio_path_translated

print("Image processing module ready.")
