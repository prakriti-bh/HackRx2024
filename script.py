import os
import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from gtts import gTTS
from moviepy.editor import *
import random
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    chunks = [text[i:i+1024] for i in range(0, len(text), 1024)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return " ".join(summaries)

def generate_script(text):
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    inputs = tokenizer([text], max_length=1024, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=100, max_length=300)
    script = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return script

def extract_key_points(text, n=5):
    sentences = text.split('.')
    vectorizer = TfidfVectorizer().fit_transform(sentences)
    similarities = cosine_similarity(vectorizer)
    scores = similarities.sum(axis=1)
    top_idx = scores.argsort()[-n:][::-1]
    return [sentences[i].strip() for i in top_idx]

# 2. Prompt Generation
def generate_prompts(key_points):
    prompts = []
    for point in key_points:
        prompt = f"Create a photorealistic image depicting {point}. High quality, detailed, 4K resolution."
        prompts.append(prompt)
    return prompts

# 3. Visual Creation
def generate_images(prompts):
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    images = []
    for prompt in prompts:
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
        images.append(image)
    return images

# 4. Voiceover Generation
def generate_voiceover(script):
    tts = gTTS(text=script, lang='en', slow=False)
    tts.save("temp_voiceover.mp3")
    
    # Convert mp3 to wav for further processing
    sound = AudioSegment.from_mp3("temp_voiceover.mp3")
    sound.export("voiceover.wav", format="wav")
    
    # Load the wav file
    y, sr = librosa.load("voiceover.wav")
    
    # Noise reduction
    y_reduced_noise = librosa.effects.remix(y, intervals=librosa.effects.split(y, top_db=20))
    
    # Speed up the audio slightly
    y_fast = librosa.effects.time_stretch(y_reduced_noise, rate=1.1)
    
    # Save the processed audio
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
        
        # Add zoom effect
        zoom = lambda t: 1 + 0.1*t
        clips.append(img_clip.resize(zoom))

    concat_clip = concatenate_videoclips(clips, method="compose")
    final_clip = concat_clip.set_audio(audio)
    
    # Add text overlay
    txt_clip = TextClip("Generated with AI", fontsize=30, color='white')
    txt_clip = txt_clip.set_pos('bottom').set_duration(final_clip.duration)
    
    final_clip = CompositeVideoClip([final_clip, txt_clip])
    
    final_clip.write_videofile(output_path, fps=24)

# 6. Quiz Generation
def generate_quiz(script, num_questions=5):
    sentences = script.split('.')
    questions = []
    for _ in range(num_questions):
        sentence = random.choice(sentences)
        words = sentence.split()
        blank_word = random.choice([w for w in words if len(w) > 3])
        question = f"Fill in the blank: {sentence.replace(blank_word, '______')}"
        questions.append((question, blank_word))
    return questions

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
    # Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)

    # Summarize text and generate script
    print("Summarizing text and generating script...")
    summary = summarize_text(text)
    script = generate_script(summary)
    key_points = extract_key_points(summary)

    # Generate prompts and images
    print("Generating prompts and images...")
    prompts = generate_prompts(key_points)
    images = generate_images(prompts)

    # Generate voiceover
    print("Generating voiceover...")
    voiceover_path = generate_voiceover(script)

    # Create video
    print("Creating video...")
    create_video(images, voiceover_path, "output_video.mp4")

    # Generate quiz
    print("Generating quiz...")
    quiz = generate_quiz(script)

    # Initialize analytics tracker
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