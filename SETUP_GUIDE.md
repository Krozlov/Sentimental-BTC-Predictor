# 🚀 Quick Setup Guide - BTC Sentiment Predictor
## DUE TOMORROW? No problem! Follow this exact order.

---

## ⏱️ TOTAL TIME: ~3-4 hours (mostly automated)

### Phase 1: Setup (15 minutes)

**Step 1: Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 2: Get API Credentials (10 minutes)**

**Reddit API (REQUIRED):**
1. Go to: https://www.reddit.com/prefs/apps
2. Click "create app" (bottom of page)
3. Select "script"
4. Name: `btc_sentiment_collector`
5. Redirect URI: `http://localhost:8080`
6. Click "create app"
7. **Save these:**
   - CLIENT_ID (under app name, looks like: `xX1234abcDEF`)
   - SECRET (next to "secret", longer string)

**News API (RECOMMENDED):**
1. Go to: https://newsapi.org/register
2. Enter email (free tier = 100 requests/day, plenty!)
3. Verify email
4. **Save your API key** --> 5016d1659dfb4eef9be53e1f0fbc260c

---

### Phase 2: Data Collection (2-3 hours, runs in background)

**Step 3: Collect Sentiment Data**
```bash
python sentiment_collector.py
```

**What it asks:**
- Reddit CLIENT_ID → paste from Step 2
- Reddit SECRET → paste from Step 2
- App name → just type: `btc_collector`
- News API KEY → paste from Step 2 (or press Enter to skip)
- Days back → type: `30` (recommended)

**Expected output:**
- Reddit: ~300-500 posts
- News: ~100 articles
- Time: 2-3 hours (leave it running, go do other work!)

---

### Phase 3: Analysis (30 minutes)

**Step 4: Analyze Sentiment**
```bash
python sentiment_analyzer.py
```

**What it does:**
- Processes all collected text
- Generates sentiment scores (-1 to +1)
- Creates daily aggregations
- Time: ~5-10 minutes

**Expected output:**
```
✓ Sentiment analysis complete!
💾 sentiment_daily_*.csv (USE THIS for modeling)
```

---

### Phase 4: Modeling (10 minutes)

**Step 5: Train Models**
```bash
python btc_predictor_enhanced.py
```

**What it does:**
- Trains baseline model (technical indicators only)
- Trains enhanced model (technical + sentiment)
- Compares performance
- Time: ~5-10 minutes

**Expected output:**
```
MODEL COMPARISON: Baseline vs Sentiment-Enhanced
Test_MAE:   [improvement stats]
Test_R²:    [improvement stats]
```

---

### Phase 5: Visualizations (10 minutes)

**Step 6: Generate Paper Figures**
```bash
python visualizations.py
```

**What it creates:**
- 5 publication-quality PNG figures
- 1 summary statistics table (CSV)
- All saved in `data/figures/`
- Time: ~2-3 minutes

---

## 📊 What You Get for Your Paper

### Figures (ready for LaTeX):
1. **fig1_sentiment_distribution.png** - Shows sentiment score distribution
2. **fig2_sentiment_over_time.png** - Temporal trends
3. **fig3_source_comparison.png** - Reddit vs News analysis
4. **fig4_sentiment_price_correlation.png** - Sentiment-price relationship
5. **fig5_feature_importance.png** - Feature contribution analysis
6. **model_comparison.png** - Baseline vs Enhanced predictions

### Tables:
- **table_summary_statistics.csv** - All key metrics

---

## 🎯 Research Paper Talking Points

### Novel Contributions:
1. **Multi-source sentiment integration** (Reddit + News)
2. **Temporal sentiment features** (lag effects, momentum)
3. **Source-specific analysis** (platform differences)
4. **Quantitative improvement** over baseline model

### Methodology Highlights:
- VADER sentiment analysis (explain why: domain-independent, no training needed)
- Ridge regression (explain alpha parameter tuning)
- Time-series validation (explain why no shuffle)
- Feature engineering (technical + sentiment fusion)

### Results to Report:
- MAE/RMSE improvement percentage
- R² score comparison
- Sentiment-price correlation coefficient
- Statistical significance (if improved)

---

## 🆘 Troubleshooting

**Problem: "Rate limit error" from yfinance**
- Solution: Wait 5 minutes, run again
- Or: Use smaller date range in code

**Problem: "No sentiment data found"**
- Solution: Make sure Step 3 completed successfully
- Check `data/` folder for CSV files

**Problem: Reddit API returns no data**
- Solution: Check credentials are correct
- Try different subreddits in code

**Problem: Low correlation between sentiment and price**
- This is NORMAL! Crypto is volatile
- Still valid research: "explored relationship, found weak correlation"
- Discuss in paper: "suggests market complexity beyond sentiment"

---

## 📝 Writing Your Paper (Overleaf Structure)

### Suggested Sections:

**1. Introduction**
- Problem: BTC prediction is hard
- Gap: Most models ignore social sentiment
- Contribution: Multi-modal sentiment integration

**2. Related Work**
- Technical indicator models
- Sentiment analysis in finance
- Crypto-specific predictions

**3. Methodology**
- Data collection (Reddit API, News API)
- Sentiment analysis (VADER)
- Feature engineering
- Model architecture (Ridge regression)

**4. Experiments**
- Dataset statistics (use table_summary_statistics.csv)
- Baseline vs Enhanced comparison
- Evaluation metrics

**5. Results**
- Performance improvements (use model_comparison.png)
- Sentiment analysis insights (use fig1-4)
- Feature importance (use fig5)

**6. Discussion**
- What worked / didn't work
- Limitations (data volume, timeframe, correlation strength)
- Future work

**7. Conclusion**
- Sentiment adds value (even if small)
- Multi-source approach is novel
- Framework is extensible

---

## 💡 Pro Tips for Tomorrow

1. **Run everything TONIGHT** - don't wait until morning
2. **If Reddit fails** - News API alone is still publishable
3. **If improvements are small** - That's OK! Report honestly
4. **Save all figures** - Screenshot outputs as backup
5. **Document your process** - Take notes for methodology section

---

## ✅ Final Checklist

Before submitting, make sure you have:
- [ ] All 4 Python scripts ran successfully
- [ ] 6 figures in `data/figures/`
- [ ] Summary statistics table
- [ ] Model comparison results saved/screenshotted
- [ ] At least 30 days of sentiment data
- [ ] Baseline vs Enhanced comparison complete

---

## 🚨 EMERGENCY: "I Only Have 6 Hours Left!"

**Minimal viable project (2 hours active work):**
1. Skip News API (Reddit only) - 30 min
2. Collect 14 days instead of 30 - 30 min
3. Run analyzer - 10 min
4. Run predictor - 10 min
5. Generate 3 key figures only - 5 min
6. Write paper from templates - 35 min

**This still gets you:**
- Working sentiment integration ✓
- Baseline comparison ✓
- Novel approach ✓
- Publishable results ✓

---

Good luck! You got this! 🎓🚀