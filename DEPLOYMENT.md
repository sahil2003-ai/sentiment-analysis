# Streamlit Cloud Deployment Guide

## Live App Deployment Steps

This guide will help you deploy the Sentiment Analysis Streamlit app to Streamlit Cloud for free hosting.

## 🚀 Quick Start Deployment (Recommended)

### Prerequisites
- GitHub account with this repository forked
- Streamlit Cloud account (free at https://share.streamlit.io)

### Step 1: Prepare Your Repository

Your repository should have the following structure:
```
sentiment-analysis/
├── sentiment.py                 # Main Streamlit app
├── pipeline.pkl                 # Trained ML model
├── requirements.txt             # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

✅ All these files are already in the repository!

### Step 2: Deploy to Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - **Repository**: sahil2003-ai/sentiment-analysis
   - **Branch**: main
   - **Main file path**: sentiment.py
5. Click "Deploy"

### Step 3: Wait for Deployment

Streamlit Cloud will:
- Install all dependencies from `requirements.txt`
- Download and cache your model (`pipeline.pkl`)
- Start the Streamlit server
- Provide you with a live URL

**Typical deployment time**: 3-5 minutes

## 📱 Live App URL

Once deployed, your app will be available at:
```
https://share.streamlit.io/sahil2003-ai/sentiment-analysis/main/sentiment.py
```

Or a custom short URL if you configure it in your Streamlit Cloud dashboard.

## ⚙️ Configuration Details

### .streamlit/config.toml

The config file includes:
- **Client settings**: Prevents directory delete warnings
- **Theme**: Custom blue color scheme matching the app
- **Server settings**: Headless mode for cloud deployment

### requirements.txt

Includes all necessary dependencies:
- Data science: pandas, numpy, scikit-learn, xgboost
- NLP: nltk, spacy, textblob, wordcloud, langdetect, deep-translator
- Web: streamlit, joblib
- Visualization: matplotlib, seaborn

## 🔧 Troubleshooting

### App Takes Too Long to Deploy
- Check the "Manage app" logs in Streamlit Cloud
- Ensure all dependencies in `requirements.txt` are correctly spelled
- Try redeploying by clicking "Reboot" in app settings

### Model File Not Found Error
- Ensure `pipeline.pkl` is committed to the repository
- File size should be ~10-15 MB (compressed)
- Check git LFS if file size exceeds limits

### Language Detection Issues
- The app downloads spacy models on first run
- This is cached in Streamlit Cloud
- First prediction may take 10-15 seconds

### Memory Issues
- The app is memory-optimized for free tier (~1GB RAM)
- Pipeline is cached using @st.cache_resource
- Simultaneous users may experience slowdowns during peak times

## 🌐 Sharing Your App

### Share Options:
1. **Direct Link**: Send the app URL to anyone
2. **Embed**: Use iframe to embed in websites
3. **Social Media**: Share on LinkedIn, Twitter, etc.

### Example Sharing:
```
"Check out my Sentiment Analysis ML App: [Your App URL]

Features:
- 8 different ML models
- Multi-language support
- Real-time predictions
- Beautiful UI with Streamlit"
```

## 📊 Monitoring & Updates

### View App Logs
1. Go to https://share.streamlit.io
2. Select your app
3. Click "Manage app" → "Logs"

### Deploy Updates
- Push changes to GitHub main branch
- Streamlit Cloud auto-deploys (unless disabled)
- Updates typically deploy within 1-2 minutes

### Disable Auto-Deploy
1. Click "Manage app"
2. Settings → Uncheck "Auto-rerun on source file change"

## 🎯 Performance Tips

1. **Cache Heavy Operations**
   ```python
   @st.cache_resource
   def load_pipeline():
       return joblib.load('pipeline.pkl')
   ```

2. **Optimize Image Display**
   - Use st.image() with width parameter
   - Compress images before uploading

3. **Session State for User Input**
   - Use st.session_state to remember user inputs
   - Reduces re-computation on reruns

4. **Lazy Loading**
   - Load models only when needed
   - Use conditional rendering

## 🔐 Security Notes

- Streamlit Cloud provides HTTPS by default
- No sensitive data stored in the app
- Model is read-only (no fine-tuning in cloud)
- Secrets can be managed via Environment Variables

## 📞 Support & Community

- **Streamlit Docs**: https://docs.streamlit.io
- **Streamlit Cloud Docs**: https://docs.streamlit.io/streamlit-community-cloud
- **Community Forum**: https://discuss.streamlit.io
- **GitHub Issues**: Create issues in this repository

## 🎓 Next Steps

1. Deploy the app to Streamlit Cloud
2. Share the live URL with others
3. Collect feedback and iterate
4. Consider adding features:
   - Upload text files for batch analysis
   - Export results as CSV
   - Analytics dashboard
   - User authentication

## 📝 Deployment Checklist

- [ ] Repository is public on GitHub
- [ ] All files committed (sentiment.py, pipeline.pkl, requirements.txt)
- [ ] requirements.txt has all dependencies
- [ ] .streamlit/config.toml is configured
- [ ] README.md is updated with features
- [ ] No hardcoded secrets or API keys
- [ ] App runs locally without errors
- [ ] Streamlit Cloud account created
- [ ] App deployed successfully
- [ ] Live URL shared with team/community

---

**Happy Deploying! 🚀**

Your Sentiment Analysis app is now live and accessible to the world!
