import argparse
import os
import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from gtts import gTTS
from moviepy.editor import *
import random
from pydub import AudioSegment
from scipy.io import wavfile
import librosa
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load a lightweight model for summarization
def load_summarization_model():
    model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return pipeline("summarization", model=model, tokenizer=tokenizer)

# Load Airoboros model with reduced precision
def load_airoboros_model():
    model_name = "jondurbin/airoboros-l2-70b-gpt4-1.4.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, low_cpu_mem_usage=True)
    return model, tokenizer

# 1. Improved Input Processing
def extract_text_from_pdf(pdf_path):
    text = ""
    for page_layout in extract_pages(pdf_path):
        for element in page_layout:
            if isinstance(element, LTTextContainer):
                text += element.get_text() + "\n"
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
    return chunks if chunks else [text]

def summarize_with_pipeline(summarizer, text, max_length=150):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    return " ".join(summaries)

def generate_script_with_airoboros(model, tokenizer, summary, max_length=500):
    prompt = f"Based on the following summary, create a detailed script for a video presentation:\n\n{summary}\n\nScript:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_length, temperature=0.7)
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
def generate_image(pipe, prompt, num_inference_steps, guidance_scale):
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
    return image

def generate_images(prompts, model_id="CompVis/stable-diffusion-v1-4", num_inference_steps=50, guidance_scale=7.5, max_workers=2):
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    images = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_image, pipe, prompt, num_inference_steps, guidance_scale) for prompt in prompts]
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
    
    y_reduced_noise = librosa.effects.remix(y, intervals=librosa.effects.split(y, top_db=30))
    y_fast = librosa.effects.time_stretch(y_reduced_noise, rate=1.05)
    
    wavfile.write("processed_voiceover.wav", sr, (y_fast * 32767).astype(np.int16))
    
    return "processed_voiceover.wav"

# 5. Video Generation
def create_video(images, voiceover_path, output_path, fps=24, zoom_factor=0.05, threads=4):
    audio = AudioFileClip(voiceover_path)
    duration = audio.duration

    clips = []
    for img in images:
        img_array = np.array(img)
        img_clip = ImageClip(img_array).set_duration(duration/len(images))
        zoom = lambda t: 1 + zoom_factor*t
        clips.append(img_clip.resize(zoom))

    concat_clip = concatenate_videoclips(clips, method="compose")
    final_clip = concat_clip.set_audio(audio)
    
    txt_clip = TextClip("Generated with AI", fontsize=30, color='white')
    txt_clip = txt_clip.set_pos('bottom').set_duration(final_clip.duration)
    
    final_clip = CompositeVideoClip([final_clip, txt_clip])
    
    final_clip.write_videofile(output_path, fps=fps, threads=threads)

# 6. Quiz Generation
def generate_quiz_with_airoboros(model, tokenizer, script, num_questions=5):
    sentences = script.split('.')
    num_questions = min(num_questions, len(sentences))
    
    prompt = f"Based on the following script, generate {num_questions} quiz questions with their answers:\n\n{script}\n\nQuiz:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.7)
    quiz = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return quiz.split("Quiz:")[1].strip().split("\n")

# 7. Analytics Dashboard (unchanged)
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

def parse_arguments():
    parser = argparse.ArgumentParser(description="Automated Video Generation from Text")
    
    # Input and output
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("--output", default="output_video.mp4", help="Path for the output video file")
    
    # Text processing
    parser.add_argument("--chunk_size", type=int, default=2000, help="Maximum size of text chunks for processing")
    parser.add_argument("--summary_length", type=int, default=150, help="Maximum length of the generated summary")
    parser.add_argument("--script_length", type=int, default=500, help="Maximum length of the generated script")
    
    # Image generation
    parser.add_argument("--num_images", type=int, default=5, help="Number of images to generate")
    parser.add_argument("--image_model", default="CompVis/stable-diffusion-v1-4", help="Stable Diffusion model to use")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of inference steps for image generation")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for image generation")
    
    # Video creation
    parser.add_argument("--fps", type=int, default=24, help="Frames per second for the output video")
    parser.add_argument("--zoom_factor", type=float, default=0.05, help="Zoom factor for image transitions")
    
    # Quiz generation
    parser.add_argument("--num_questions", type=int, default=5, help="Number of quiz questions to generate")
    
    # Performance
    parser.add_argument("--max_workers", type=int, default=2, help="Maximum number of workers for concurrent image generation")
    parser.add_argument("--video_threads", type=int, default=4, help="Number of threads to use for video encoding")
    
    return parser.parse_args()

def main(args):
    print("Loading models...")
    summarizer = load_summarization_model()
    airoboros_model, airoboros_tokenizer = load_airoboros_model()

    print(f"Extracting text from PDF: {args.input_pdf}")
    text = extract_text_from_pdf(args.input_pdf)

    print("Summarizing text and generating script...")
    summary = summarize_with_pipeline(summarizer, text, max_length=args.summary_length)
    script = generate_script_with_airoboros(airoboros_model, airoboros_tokenizer, summary, max_length=args.script_length)
    key_points = extract_key_points(airoboros_model, airoboros_tokenizer, summary, n=args.num_images)

    print("Generating prompts and images...")
    prompts = generate_prompts(key_points)
    images = generate_images(prompts, model_id=args.image_model, num_inference_steps=args.inference_steps, 
                             guidance_scale=args.guidance_scale, max_workers=args.max_workers)

    print("Generating voiceover...")
    voiceover_path = generate_voiceover(script)

    print(f"Creating video: {args.output}")
    create_video(images, voiceover_path, args.output, fps=args.fps, zoom_factor=args.zoom_factor, threads=args.video_threads)

    print("Generating quiz...")
    quiz = generate_quiz_with_airoboros(airoboros_model, airoboros_tokenizer, script, num_questions=args.num_questions)

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
    args = parse_arguments()
    main(args)