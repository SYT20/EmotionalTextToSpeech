# Empathy Engine: Emotion-Driven Text-to-Speech Pipeline

An AI-powered pipeline that transforms plain text into expressive, emotion-aware speech by analyzing sentiment and dynamically adjusting voice parameters for natural, empathetic audio output.

## Features

- **Emotion Detection**: Uses Hugging Face transformers to identify emotions (joy, sadness, anger, fear, surprise, disgust, neutral) with intensity scores
- **Dynamic Voice Parameterization**: Leverages Google Gemini AI to intelligently map emotions to appropriate voice styles
- **Expressive TTS Generation**: Integrates with Murf.ai API for high-quality, emotion-responsive speech synthesis
- **Audio Post-Processing**: Combines sentence segments with smooth crossfades for seamless audio output
- **Multilingual Support**: Language detection for potential future expansion

## Architecture

```
Text Input → Language Detection → Sentence Splitting → Emotion Analysis → Voice Parameter Mapping → TTS Generation → Audio Combination → Output
```

## Prerequisites

- Python 3.8+
- Virtual environment (recommended)
- API Keys for:
  - Hugging Face (for emotion model access)
  - Google Gemini (for voice parameter generation)
  - Murf.ai (for text-to-speech)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd empathy-engine
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install transformers torch langdetect nltk pydub requests python-dotenv google-generativeai
```

### 4. Download NLTK Data

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=ap2_8e5a5719-dabf-4f91-8e67-8a5eb205a3bb
ELEVENLABS_API_KEY=AIzaSyDqLOHeSWSe6lvJwumgI6fWHwYwryk5wUY
```

**Note**: The `ELEVENLABS_API_KEY` variable actually contains your Murf.ai API key.

### 6. Get API Keys

- **Hugging Face**: Sign up at [huggingface.co](https://huggingface.co) and create an API token
- **Google Gemini**: Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Murf.ai**: Register at [murf.ai](https://murf.ai) and obtain your API key

## Usage

### Basic Usage

```python
from empathy_engine import EmpathyEngine

# Initialize the engine
engine = EmpathyEngine()

# Process text
text = "I'm so excited about this new project! It's going to be amazing."
output_file = engine.process(text)

print(f"Generated audio: {output_file}")
```

### Command Line

```bash
python empathy_engine.py
```

This will process the sample text and generate `output_expressive.wav`.

## Design Choices

### Emotion Detection

- **Model**: `j-hartmann/emotion-english-distilroberta-base` from Hugging Face
- **Why**: Provides reliable emotion classification with intensity scores across 7 emotion categories
- **Processing**: Analyzes each sentence individually for granular emotion mapping

### Voice Parameter Mapping

The system uses Google Gemini AI to dynamically generate voice style parameters based on detected emotions. This approach provides:

- **Contextual Intelligence**: AI understands emotional nuance beyond simple rule-based mapping
- **Flexibility**: Can adapt to new emotions or intensity levels without code changes
- **Consistency**: Maintains coherent voice style decisions across different text inputs

#### Emotion-to-Style Mapping Logic

The Gemini model is prompted to suggest Murf.ai voice styles based on emotion and intensity:

- **Joy**: Conversational, Promo (energetic, engaging)
- **Sadness**: Narrative (calm, storytelling)
- **Anger**: Promo (assertive, commanding)
- **Fear**: Narrative (measured, cautious)
- **Surprise**: Conversational (dynamic, expressive)
- **Disgust**: Narrative (detached, reflective)
- **Neutral**: Conversational (balanced, natural)

#### Fallback System

If Gemini API is unavailable, the system uses predefined emotion-style mappings with intensity scaling for robust operation.

### TTS Generation

- **Provider**: Murf.ai API
- **Voice Selection**: Currently uses "en-US-natalie" for all emotions (can be extended to emotion-specific voices)
- **Style Application**: Applies Gemini-generated style parameters to voice synthesis
- **Audio Processing**: Downloads generated audio and combines segments with crossfade transitions

### Audio Post-Processing

- **Crossfade Duration**: 100ms between sentence segments
- **Format**: WAV output for high quality
- **Cleanup**: Automatically removes temporary audio files

## Configuration

### Voice Styles

Murf.ai supports various voice styles that affect delivery:
- **Conversational**: Natural, friendly tone
- **Promo**: Energetic, promotional style
- **Narrative**: Storytelling, calm delivery

### Emotion Intensity Scaling

Voice parameters are scaled based on emotion intensity (0-1):
- Higher intensity = more pronounced style application
- Lower intensity = more subtle adjustments

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure all required API keys are set in `.env`
   - Check API key validity and permissions

2. **NLTK Data Missing**
   - Run: `python -c "import nltk; nltk.download('punkt_tab')"`

3. **Model Download Issues**
   - Ensure stable internet connection for Hugging Face model download
   - Check available disk space

4. **Audio Generation Failures**
   - Verify Murf.ai API key and account status
   - Check API rate limits

### Performance Notes

- First run may take longer due to model downloads
- Gemini API calls add ~2-3 seconds per emotion analysis
- Audio generation time depends on text length and Murf.ai processing

## Future Enhancements

- Support for multiple languages
- Emotion-specific voice selection
- Real-time streaming TTS
- Custom voice training integration
- Advanced audio effects (reverb, EQ)

## Dependencies

- `transformers`: Hugging Face models
- `torch`: PyTorch backend
- `langdetect`: Language detection
- `nltk`: Text tokenization
- `pydub`: Audio processing
- `requests`: HTTP client
- `python-dotenv`: Environment management
- `google-generativeai`: Gemini AI integration

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]


