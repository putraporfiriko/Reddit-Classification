## ML-Based Reddit Flair Classification

Scrapes data from Reddit using praw and trains 5 machine learning models to classify Reddit posts into different flairs, and compares different models performance with matplotlib visualization. also does model tuning and hyperparameter optimization. ensemble learning :soontm:

used machine learning models:
- K-Nearest Neighbors
- Random Forest
- Gradient Boosting
- XGBoost
- AdaBoost

# How to Run:
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Reddit-Classification.git
   ```
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Scrape data from reddit:
   ```
   python redditpull.py
   ```
4. Cleanup scraped data:
   ```
   python filecleanup.py
   ```
5. Run the training and evaluation script:
   ```
   python actuallyai.py
   ```

# Acknowledgements
* [PRAW (Python Reddit API Wrapper)](https://github.com/praw-dev/praw) by Bryce Boe
* [Matplotlib](https://matplotlib.org/) by John Hunter, Darren Dale, Eric Firing, Michael Droettboom, and the Matplotlib development team
* [Scikit-learn](https://scikit-learn.org/) by the Scikit-learn developers
* [Numpy](https://numpy.org/) by the NumPy developers
* [Sentiment Analysis NLP with Python](https://github.com/yrtnsari/Sentiment-Analysis-NLP-with-Python) by yrtnsari
