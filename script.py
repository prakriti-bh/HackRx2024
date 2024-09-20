import os
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from gtts import gTTS
from moviepy.editor import *
import random
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load Airoboros model
def load_airoboros_model():
    model_name = "jondurbin/airoboros-l2-70b-gpt4-1.4.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    return model, tokenizer

# 1. Input Processing
def extract_text_from_pdf(pdf_path):
    images = convert_from_path(pdf_path)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    text = ""
    for image in images:
        result = ocr.ocr(np.array(image))
        for line in result:
            text += line[1][0] + "\n"
    return text

def chunk_text(text, max_length=2000):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks if chunks else [text]  # Return the original text if it's smaller than max_length

def summarize_with_airoboros(model, tokenizer, text):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        prompt = f"Summarize the following text concisely:\n\n{chunk}\n\nSummary:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, temperature=0.7)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary.split("Summary:")[1].strip())
    return " ".join(summaries)

def generate_script_with_airoboros(model, tokenizer, summary):
    prompt = f"Based on the following summary, create a detailed script for a video presentation:\n\n{summary}\n\nScript:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
    script = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return script.split("Script:")[1].strip()

def extract_key_points(model, tokenizer, summary, n=5):
    prompt = f"Extract {n} key points from the following summary:\n\n{summary}\n\nKey points:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    key_points = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return key_points.split("Key points:")[1].strip().split("\n")

# 2. Prompt Generation
def generate_prompts(key_points):
    prompts = []
    for point in key_points:
        prompt = f"Create a photorealistic image depicting {point}. High quality, detailed, 4K resolution."
        prompts.append(prompt)
    return prompts

# 3. Visual Creation
def generate_image(pipe, prompt):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
    return image

def generate_images(prompts):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    images = []
    with ThreadPoolExecutor(max_workers=2) as executor:  # Limit concurrent generations
        futures = [executor.submit(generate_image, pipe, prompt) for prompt in prompts]
        for future in as_completed(futures):
            images.append(future.result())
    
    return images

# 4. Voiceover Generation
def generate_voiceover(script):
    tts = gTTS(text=script, lang='en', slow=False)
    tts.save("temp_voiceover.mp3")
    
    sound = AudioSegment.from_mp3("temp_voiceover.mp3")
    sound.export("voiceover.wav", format="wav")
    
    y, sr = librosa.load("voiceover.wav")
    
    # Optimized noise reduction
    y_reduced_noise = librosa.effects.remix(y, intervals=librosa.effects.split(y, top_db=30))
    
    # Faster time stretching
    y_fast = librosa.effects.time_stretch(y_reduced_noise, rate=1.05)
    
    wavfile.write("processed_voiceover.wav", sr, (y_fast * 32767).astype(np.int16))
    
    return "processed_voiceover.wav"

# 5. Video Generation
def create_video(images, voiceover_path, output_path):
    audio = AudioFileClip(voiceover_path)
    duration = audio.duration

    clips = []
    for img in images:
        img_array = np.array(img)
        img_clip = ImageClip(img_array).set_duration(duration/len(images))
        zoom = lambda t: 1 + 0.05*t  # Reduced zoom effect for smoother transitions
        clips.append(img_clip.resize(zoom))

    concat_clip = concatenate_videoclips(clips, method="compose")
    final_clip = concat_clip.set_audio(audio)
    
    txt_clip = TextClip("Generated with AI", fontsize=30, color='white')
    txt_clip = txt_clip.set_pos('bottom').set_duration(final_clip.duration)
    
    final_clip = CompositeVideoClip([final_clip, txt_clip])
    
    final_clip.write_videofile(output_path, fps=24, threads=4)  # Use multiple threads for faster encoding

# 6. Quiz Generation
def generate_quiz_with_airoboros(model, tokenizer, script, num_questions=5):
    sentences = script.split('.')
    num_questions = min(num_questions, len(sentences))  # Ensure we don't ask for more questions than sentences
    
    prompt = f"Based on the following script, generate {num_questions} quiz questions with their answers:\n\n{script}\n\nQuiz:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
    quiz = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return quiz.split("Quiz:")[1].strip().split("\n")

# 7. Analytics Dashboard
class AnalyticsTracker:
    def __init__(self):
        self.analytics = {}

    def track_event(self, video_id, event_type, value):
        if video_id not in self.analytics:
            self.analytics[video_id] = {
                "playtime": 0,
                "quiz_score": 0,
                "rewinds": 0,
                "fast_forwards": 0,
                "pause_duration": 0,
                "inactivity": 0
            }
        self.analytics[video_id][event_type] += value

    def get_analytics(self, video_id):
        return self.analytics.get(video_id, {})

# Main process
def main(pdf_path):
    print("Loading Airoboros model...")
    model, tokenizer = load_airoboros_model()

    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    print("Summarizing text and generating script...")
    summary = summarize_with_airoboros(model, tokenizer, text)
    script = generate_script_with_airoboros(model, tokenizer, summary)
    key_points = extract_key_points(model, tokenizer, summary)

    print("Generating prompts and images...")
    prompts = generate_prompts(key_points)
    images = generate_images(prompts)

    print("Generating voiceover...")
    voiceover_path = generate_voiceover(script)

    print("Creating video...")
    create_video(images, voiceover_path, "output_video.mp4")

    print("Generating quiz...")
    quiz = generate_quiz_with_airoboros(model, tokenizer, script)

    analytics_tracker = AnalyticsTracker()

    print("Video created successfully!")
    print(f"Quiz questions: {quiz}")

    # Simulate some analytics events
    analytics_tracker.track_event("video_001", "playtime", 120)
    analytics_tracker.track_event("video_001", "quiz_score", 80)
    analytics_tracker.track_event("video_001", "rewinds", 2)
    analytics_tracker.track_event("video_001", "fast_forwards", 1)
    analytics_tracker.track_event("video_001", "pause_duration", 5)
    analytics_tracker.track_event("video_001", "inactivity", 10)

    print("Analytics:", analytics_tracker.get_analytics("video_001"))

if __name__ == "__main__":
    main("input.pdf")