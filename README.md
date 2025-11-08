# 🤖 AI Data Preprocessing Chatbot

An intelligent chatbot that automatically detects data quality issues and provides one-click fixes for your datasets.

## ✨ Key Features

- 🔍 **Automatic Issue Detection**: Instantly identifies missing values, duplicates, outliers, and formatting issues
- 🔧 **One-Click Fixes**: Fix individual issues with dedicated "Fix This Issue" buttons
- 📊 **Real-Time Analysis**: Live dataset preview and statistics
- 💬 **Interactive Chat**: Ask questions about your data and get instant insights
- 📥 **Download Results**: Get your cleaned dataset with detailed cleaning history

## 🎯 What Issues It Detects & Fixes

- **Missing Values** (High/Medium/Low priority based on percentage)
- **Duplicate Rows** (Removes identical rows)  
- **Outliers** (IQR-based detection and capping)
- **Text Formatting Issues** (Standardizes case and removes extra spaces)

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the chatbot:
```bash
streamlit run chatbot_app.py
```

3. Upload your CSV/Excel file and start chatting!

## Usage

- Upload your dataset using the sidebar
- Ask the chatbot to "analyze data" or "preprocess data"
- Download your cleaned dataset
- Chat naturally about your data needs

## Supported Formats

- CSV files
- Excel files (.xlsx, .xls)

## Automatic Processing

The chatbot automatically:
- Removes duplicate rows
- Fills missing values (median for numbers, mode for text)
- Standardizes text formatting
- Caps outliers using IQR method
- Provides detailed processing reports

Enjoy automated data preprocessing! 🚀