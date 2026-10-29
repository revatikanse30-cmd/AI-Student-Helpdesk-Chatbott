# AI Text Summarizer

A simple AI-powered web application that turns long text into concise summaries using Natural Language Processing (NLP).

## Features

- Summarizes long English text with a pretrained AI model
- Short, Balanced, and Detailed summary settings
- Standard paragraph or bullet-point output
- Original and summary word counts
- Estimated text reduction percentage
- Download the generated summary as a `.txt` file
- Sample text for quick testing
- Friendly Streamlit web interface

## Tech stack

- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- `sshleifer/distilbart-cnn-12-6` pretrained summarization model

## Requirements

- Python 3.10–3.12 recommended
- Internet connection for the first model download
- Enough disk space and memory to load the model

## Run locally

### 1. Create and activate a virtual environment

**Windows Command Prompt**
```bat
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

The first run downloads the pretrained model and may take a few minutes.

## How it works

1. The user enters text.
2. The app tokenizes the input and applies a safe input-length limit.
3. A pretrained Transformer model generates a summary.
4. The app displays the result, word counts, and a download option.

## Testing checklist

- [ ] App opens in the browser.
- [ ] Empty/short input displays a warning.
- [ ] A long paragraph produces a summary.
- [ ] Summary settings can be changed.
- [ ] Bullet-point output works.
- [ ] Download button saves a text file.
- [ ] Output is reviewed against the source for accuracy.

## Limitations

- The model is intended for English summarization.
- Very long text may be truncated to fit the model's input limit.
- AI summaries can omit important details or make mistakes. Always check important information against the original text.
- The first launch needs internet access to download the model.

## Project details

**Internship:** Skillinf Tech India  
**Project:** Artificial Intelligence — Week 3  
**Title:** AI Text Summarizer
