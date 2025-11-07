import os
import json
import requests
import google.generativeai as genai
from langdetect import detect
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from pydub import AudioSegment
from dotenv import load_dotenv

class EmpathyEngine:
    def __init__(self):
        load_dotenv()
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
        self.murf_api_key = os.getenv('ELEVENLABS_API_KEY')  # Actually Murf API key
        genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
        self.gemini_model = genai.GenerativeModel('gemini-1.5-pro')
        
    def detect_language(self, text):
        return detect(text)
    
    def split_sentences(self, text):
        return sent_tokenize(text)
    
    def detect_emotion(self, sentence):
        results = self.emotion_classifier(sentence)
        emotion_scores = results if isinstance(results, list) else []
        if emotion_scores and isinstance(emotion_scores[0], dict):
            primary = max(emotion_scores, key=lambda x: x['score'])
            emotion = primary['label']
            intensity = primary['score']
        else:
            emotion = 'neutral'
            intensity = 0.5
        return emotion, intensity
    
    def map_emotion_to_prosody(self, emotion, intensity):
        """Use Gemini to get Murf.ai voice parameters based on emotion"""
        # Check if Gemini API key is available
        if not os.getenv('GOOGLE_API_KEY'):
            print("Gemini API key not found. Using fallback parameters.")
            return self._get_fallback_prosody(emotion, intensity)

        prompt = f"Given the emotion '{emotion}' with intensity {intensity:.2f} (0-1 scale), suggest voice style settings for Murf.ai. Respond with just a JSON object with key 'style' (string value like 'Conversational', 'Promo', 'Narrative', etc.)."
        try:
            response = self.gemini_model.generate_content(prompt)
            content = response.text.strip()
            try:
                params = json.loads(content)
            except:
                params = self._get_fallback_prosody(emotion, intensity)
        except Exception as e:
            print(f"Gemini API failed: {e}. Using fallback parameters.")
            params = self._get_fallback_prosody(emotion, intensity)
        return params
    
    def _get_fallback_prosody(self, emotion, intensity):
        """Get fallback style parameters based on emotion"""
        emotion_styles = {
            'joy': 'Conversational',
            'sadness': 'Narrative',
            'anger': 'Promo',
            'fear': 'Narrative',
            'surprise': 'Conversational',
            'neutral': 'Conversational',
            'disgust': 'Narrative'
        }
        return {"style": emotion_styles.get(emotion, 'Conversational')}
    
    def generate_tts(self, sentence, emotion, prosody_params):
        """Generate TTS with Murf.ai API"""
        if not self.murf_api_key:
            raise ValueError("Murf API key not found in environment variables.")
        
        # Choose voice based on emotion
        voice_mapping = {
            'joy': "en-US-natalie",
            'sadness': "en-US-natalie", 
            'anger': "en-US-natalie",
            'fear': "en-US-natalie",
            'surprise': "en-US-natalie",
            'disgust': "en-US-natalie",
            'neutral': "en-US-natalie"
        }
        voice_id = voice_mapping.get(emotion, "en-US-natalie")
        
        try:
            response = requests.post(
                "https://api.murf.ai/v1/speech/generate",
                headers={
                    "api-key": self.murf_api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "text": sentence,
                    "voiceId": voice_id,
                    "style": prosody_params.get("style", "Conversational")
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Murf API error: {response.status_code} - {response.text}")
            
            result = response.json()
            audio_url = result.get("audioFile")
            
            if not audio_url:
                raise Exception("No audio file URL in response")
            
            # Download the audio file
            audio_response = requests.get(audio_url)
            output_file = f"temp_{abs(hash(sentence))}.wav"
            
            with open(output_file, "wb") as f:
                f.write(audio_response.content)
            
            return output_file
            
        except Exception as e:
            raise Exception(f"Murf.ai TTS failed: {e}")
    
    def combine_audio(self, audio_files):
        """Combine audio segments with crossfade"""
        if not audio_files:
            return None
            
        combined = AudioSegment.from_wav(audio_files[0])
        
        # Crossfade between segments (100ms smooth transition)
        for file in audio_files[1:]:
            next_segment = AudioSegment.from_wav(file)
            combined = combined.append(next_segment, crossfade=100)
        
        output_file = "output_expressive.wav"
        combined.export(output_file, format="wav")
        
        # Cleanup temp files
        for file in audio_files:
            if os.path.exists(file):
                os.remove(file)
                
        return output_file
    
    def process(self, text):
        """Main processing pipeline"""
        # Detect language (for multilingual support)
        lang = self.detect_language(text)
        
        # Split into sentences
        sentences = self.split_sentences(text)
        
        # Process each sentence
        audio_files = []
        for sentence in sentences:
            emotion, intensity = self.detect_emotion(sentence)
            prosody = self.map_emotion_to_prosody(emotion, intensity)
            audio_file = self.generate_tts(sentence, emotion, prosody)
            audio_files.append(audio_file)

            print(f"Sentence: {sentence[:50]}...")
            print(f"  Emotion: {emotion} (intensity: {intensity:.2f})")
            print(f"  Style: {prosody}")
        
        # Combine with crossfade
        final_audio = self.combine_audio(audio_files)
        return final_audio

if __name__ == "__main__":
    engine = EmpathyEngine()
    text = "Hi Mike,have you been? I am really excited to work with you on this project. However, I am a bit nervous about the deadlines. But I believe we can do it together!"
    output = engine.process(text)
    print(f"\nGenerated expressive audio: {output}")