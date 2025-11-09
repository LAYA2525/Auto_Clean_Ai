# 🤖 GenAI-Powered Data Cleaning Assistant

An intelligent data cleaning application powered by **Google Gemini AI** for automated data preprocessing and quality enhancement.

## ✨ Key Features

- 🤖 **Real GenAI Integration**: Powered by Google Gemini LLM for intelligent chat responses
- 🔍 **Automatic Analysis**: Instantly analyzes data quality, missing values, and duplicates
- 🧹 **One-Click Cleaning**: Automated cleaning with median/mode imputation and duplicate removal
- � **AI Chat Assistant**: Natural language conversation about your data using Gemini AI
- � **Quality Metrics**: Real-time quality score and comprehensive statistics
- 📥 **Dual Downloads**: Get both cleaned data and detailed cleaning report
- � **Clean Interface**: Professional, minimalist design perfect for presentations

## 🚀 Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key (FREE)

1. Visit: https://makersuite.google.com/app/apikey
2. Click "Create API Key"
3. Copy your API key

### 3. Configure API Key

Edit `.streamlit/secrets.toml` and add your API key:

```toml
GEMINI_API_KEY = "your-actual-api-key-here"
```

### 4. Run the Application

```bash
streamlit run simple_clean_genai.py
```

## 🎯 GenAI Features

### Real AI-Powered Chat
- Uses Google Gemini Pro model for natural language understanding
- Context-aware responses based on your actual dataset
- Intelligent answers to any data cleaning question
- Falls back to rule-based responses if API is unavailable

### Intelligent Analysis
- Automated quality scoring
- Smart duplicate detection
- Missing value identification
- Statistical preprocessing

## 📖 Usage

1. **Upload Dataset**: Upload CSV file through the interface
2. **View Analysis**: See automatic quality metrics (rows, columns, missing, duplicates, quality score)
3. **Clean Data**: Click "Clean Data" button for AI-powered preprocessing
4. **View Results**: Expand "Results" and "View Details" sections
5. **Download**: Get cleaned data (CSV) and comprehensive report (TXT)
6. **Chat with AI**: Ask questions about your data, cleaning methods, or algorithms

## 🤖 What Makes This GenAI?

- **Google Gemini Integration**: Real LLM API calls for chat responses
- **Natural Language Processing**: Understands and responds to user queries intelligently
- **Context-Aware AI**: Responses adapt based on your dataset's actual statistics
- **Automated Intelligence**: AI-powered decision making for data cleaning strategies

## 📊 Supported Data Formats

- CSV files (.csv)
- Standardizes text formatting
- Caps outliers using IQR method
- Provides detailed processing reports

Enjoy automated data preprocessing! 🚀