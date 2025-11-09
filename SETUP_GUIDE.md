# 🚀 GenAI Setup Guide

## Quick Setup (5 minutes)

### Step 1: Get Your FREE Google Gemini API Key

1. Go to: **https://makersuite.google.com/app/apikey**
2. Sign in with your Google account
3. Click **"Create API Key"** or **"Get API Key"**
4. Copy the API key (starts with "AIza...")

### Step 2: Add API Key to Your Project

1. Open the file: `.streamlit/secrets.toml`
2. Replace `"your-api-key-here"` with your actual API key:

```toml
GEMINI_API_KEY = "AIzaSyC_your_actual_key_here"
```

3. Save the file

### Step 3: Run the Application

```powershell
streamlit run simple_clean_genai.py
```

## 🎉 That's It!

Your app now has **REAL GenAI** powered by Google Gemini!

## How to Test GenAI is Working

1. Upload a CSV file
2. Go to the AI Chat section at the bottom
3. Ask: **"What can you tell me about my dataset?"**
4. If you get a smart, context-aware response → GenAI is working! ✅
5. If you get a generic response → Still using fallback mode (check API key)

## Troubleshooting

### "Import google.generativeai could not be resolved"
- Already fixed! Package installed automatically

### GenAI not responding
- Check your API key in `.streamlit/secrets.toml`
- Make sure the key starts with "AIza"
- Restart the Streamlit app after adding the key

### API Key Issues
- Get a new key from: https://makersuite.google.com/app/apikey
- Free tier: 60 requests per minute (plenty for testing!)

## What's Different Now?

### Before (Rule-Based):
- Pre-programmed responses
- Limited understanding
- Same answers every time

### After (Real GenAI):
- Google Gemini LLM responses
- Natural language understanding
- Context-aware, intelligent answers
- Adapts to YOUR specific dataset

## Cost

**100% FREE** for personal/educational use!
- Google Gemini API free tier: 60 requests/minute
- Perfect for capstone projects and demonstrations

## Example Questions to Ask the AI

- "How would you improve the quality of my data?"
- "What's the best approach for handling these duplicates?"
- "Explain the cleaning process in simple terms"
- "What machine learning algorithms are you using?"
- "Is my data ready for analysis?"

Enjoy your **REAL GenAI-powered** data cleaning assistant! 🚀
