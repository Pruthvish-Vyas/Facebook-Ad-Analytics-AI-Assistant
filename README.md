# Facebook Ad Analytics & AI Assistant

## 🚀 Overview
An advanced Streamlit application integrating machine learning predictions with GPT-powered insights for Facebook advertising campaigns. Built with Streamlit and OpenAI's GPT models, this tool helps advertisers predict conversion rates and get AI-powered campaign optimization suggestions.

## ✨ Key Features

### 📊 Prediction Module
- Real-time conversion predictions
- Campaign performance metrics (CTR, CPC, CVR)
- Interactive dashboard with key metrics
- Campaign summary visualization
- Date-based analysis

### 🤖 AI Assistant Features
- GPT-powered campaign analysis
- Multiple model options:
  - GPT-3.5-turbo
  - GPT-4
  - GPT-4-turbo-preview
  - GPT-3.5-turbo-16k
- Contextual recommendations
- Chat interface with history
- Example questions provided

## 💻 Technical Details

### Dependencies
```python
streamlit
pandas
datetime
openai
python-dotenv
```

### Project Structure
```
fb/
├── app.py # Main Streamlit application
├── src/
│ ├── pipeline/
│ │ └── predict_pipeline.py
│ ├── components/
│ │ ├── data_processor.py
│ │ └── model.py
│ ├── utils/
│ │ └── helpers.py
│ └── logger/
│ └── init.py
├── models/ # Saved model files
├── data/ # Data directory
├── tests/ # Unit tests
├── config/ # Configuration files
├── requirements.txt
└── README.md
```

## 🎯 Features

### Campaign Metrics Input
- Campaign ID & Facebook Campaign ID
- Date range selection
- Demographic targeting (age groups, gender)
- Interest targeting (up to 3 interests)
- Performance metrics:
  - Impressions
  - Clicks
  - Spend
  - Total conversions

### Performance Analytics
- CTR (Click-Through Rate)
- CPC (Cost per Click)
- CVR (Conversion Rate)
- Approved conversions prediction

### AI Assistant Capabilities
- Performance analysis
- Campaign optimization tips
- Budget recommendations
- Targeting suggestions
- KPI interpretation
- Best practices

## 🚀 Getting Started

1. Clone and install dependencies:
```bash
git clone <repository>
cd fb
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Configure:
   - Enter OpenAI API key for AI assistant
   - Input campaign metrics
   - Choose between prediction tool and AI chat

## 💡 Usage Tips

### Prediction Tool
1. Enter campaign details in sidebar
2. Input performance metrics
3. Click "Predict Approved Conversions"
4. View results and KPIs

### AI Assistant
1. Enter OpenAI API key
2. Select preferred model
3. Ask questions about:
   - Campaign performance
   - Optimization strategies
   - Metric interpretation
   - Best practices

## 🔒 Security
- API keys are session-only
- No data persistence
- Secure processing

## 📝 Notes
- CTR = (clicks/impressions) × 100
- CPC = spent/clicks
- CVR = (conversions/clicks) × 100
- Keep API key confidential

## 📄 License
MIT License - Feel free to use and modify as needed.
