<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/fashion_brands.jpg" width=80%>

# Aspect Based Sentiment Analysis for Fashion Products
The fashion industry is characterized by rapidly changing trends and consumer preferences. Consumers express their emotions through product reviews and social networks such as Instagram, Facebook and Twitter. The ability to detect consumer satisfaction from user-generated content online can be very useful for fashion brands to swiftly respond to customer needs accordingly.

While sentiment analysis can help identify the sentiment behind an opinion or statement, there might be several aspects that have triggered the identified sentiment. Aspect Based Sentiment Analysis (ABSA) is a technique that takes into consideration the underlying aspects and identifies the sentiment associated with each aspect.

The main objective of this project is to develop a novel ABSA pipeline for identifying aspects from fashion product reviews and predicting the sentiment toward each aspect.

## [Data Wrangling](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/Data_Wrangling_and_EDA.ipynb)
Amazon Fashion is the apparel department on Amazon that focuses on fashion products and services. The dataset was retrieved from the [Amazon Review Data (2018)](https://nijianmo.github.io/amazon/). It contains product reviews and metadata from Amazon Fashion, including reviews spanning 2002 - 2018.

Some major data wrangling steps including:
* **Removing duplicates and missing values** - Rows with duplicate or missing reviews were removed.
* **Natural language processing**
  * Remove urls
  * Remove html tags
  * Remove extra white spaces
  * Remove accented characters
  * Expand contractions
  * Convert all characters to lowercase
  * Remove special characters
  * Lemmatize tokens to their base forms
  * Extract nouns and adjectives for the aspect extraction step

## [Exploratory Data Analysis](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/Data_Wrangling_and_EDA.ipynb)
### Review Word Clouds
We can clearly see what are the most frequently used words in the reviews for a specific product from its word cloud. For the shoe insole, the words with the highest frequencies are *foot, shoe* and *insole*. Some other words that are frequently used in the reviews are *good, support, great* and *arch*. For the yoga pants, the words with the highest frequencies are *fit, love, great* and *pant*. Some other words that are frequently used in the reviews are *color, wear* and *pair*.

<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/word_cloud_1.png" width=50%><img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/word_cloud_2.png" width=50%>

### Review Sentiment Distribution
The stacked bar chart shows the review sentiment distribution from 2002 to 2018. We can see that the largest lift in the positive review percentage took place between 2005 and 2006, an increase from 62% to 76% to be specific.

<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/bar_plot_sentiment_distribution.png">

### Hypothesis Testing
>**H<sub>0</sub>: The positive review percentage for the current year is lower than or equal to the previous year (p<sub>year</sub> ≤ p<sub>year - 1</sub>)**  
>**H<sub>1</sub>: The positive review percentage for the current year is higher than the previous year (p<sub>year</sub> > p<sub>year - 1</sub>)**

We select 5% as the significance level α and use two proportions z-test to test our hypothesis for each year from 2003 to 2018. The table below summarizes the results of the tests. The positive percentages highlighted in red indicate that it has increased compared to the previous year, and p values highlighted in yellow indicate that the result is statistically significant to reject the null hypothesis. We conclude that the positive review percentages for 2006, 2016, 2017 and 2018 are higher than their previous years.

<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/table_hypothesis_testing.PNG">

## [Aspect Extraction](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/Aspect_Extraction.ipynb)
we provide a list of six aspects the fashion industry is commonly interested in: *color, design, material, price, quality, and sizing*, and use an unsupervised approach to extract aspects from each review. The main steps are:
* Extract nouns and adjectives from the text corpus
* Embed each extracted word and each aspect from the given list, compute the semantic similarity between each word-aspect pair
* Examine the word-aspect pairs with the highest semantic similarities, and select the similarity threshold for the aspect extraction step of the ABSA pipeline.

## [Sentiment Prediction](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/tree/main/Sentiment_Analysis)
Sentiment prediction is a classification problem and the label is the underlying sentiment of a review. We map reviews with ratings of 5 or 4 as Positive, 3 as Neutral, 2 or 1 as Negative.

### Feature Engineering
We embed tokens as vectors using below common techniques depending on the characteristics of the algorithms:
* Bag of Words (BOW) with unigram and bigram features
* Term Frequency-Inverse Document Frequency (TF-IDF) with unigram and bigram features
* Word2Vec using the pre-trained model on Google News dataset

### Model Training
We evaluate below classical [machine learning algorithms](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/Sentiment_Analysis/Sentiment_Analysis_Machine_Learning.ipynb):
* Logistic Regression
* Multinomial Naive Bayes
* Random Forest

We evaluate below neural network algorithms:
* [LSTM](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/Sentiment_Analysis/Sentiment_Analysis_LSTM.ipynb)
  * LSTM
  * Bidirectional LSTM
  * LSTM with attention layer
* [Transformer (BERT model)](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/Sentiment_Analysis/Sentiment_Analysis_Transformer.ipynb)

### Model Evaluation
The transformer model has the highest weighted accuracy and F1, as well as F1 for negative reviews. The logistic regression model trained on TF-IDF features also has decent performance, and the prediction time on the test dataset is significantly faster than the transformer model.

<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/table_model_performance_comparison.PNG">

Our recommendations are twofold for this ABSA project and fashion brands should adopt the machine learning model that best suits their needs. We recommend the logistic regression model for social listening purposes because companies need a model that can detect any negative sentiment quickly so that they can respond before the word-of-mouth starts damaging their brands. We recommend the transformer model for other common purposes like customer experience management and product development because companies need a model that can detect review aspects and sentiment accurately as they aim to extract actionable insights for long-term planning.

## [ABSA Pipeline](https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/ABSA_Pipeline.ipynb)
We create a novel ABSA pipeline for the fashion industry. The pipeline preprocesses the textual data, extracts the underlying aspect(s), and predicts the sentiment for the extracted aspect(s) in a streamlined process.

### [Positive Review Example](https://www.amazon.com/gp/customer-reviews/R2D9JK13WPNO5V/)
<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/review_example_1.png" width=60%>

**Pipeline output**
><img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/output_example_1.png">

### [Neutral Review Example](https://www.amazon.com/gp/customer-reviews/R2RNIJ7N09BNE8/)
<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/review_example_2.png" width=60%>

**Pipeline output**
><img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/output_example_2.png">

### [Negative Review Example](https://www.amazon.com/gp/customer-reviews/R3E1GV6HPZ32XE/)
<img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/review_example_3.png" width=60%>

**Pipeline output**
><img src="https://github.com/amythemirror/Aspect_Based_Sentiment_Analysis_for_Fashion_Products/blob/main/README_files/output_example_3.png" width=50%>

## Future Improvements
* Extract quantities, measurements and their units from numbers in the preprocessing step, and utilize them to improve aspect extraction and sentiment prediction results.
* Train a model on annotated data for better aspect extraction.
* Include emoticons as features for the sentiment prediction.
* Tune hyperparameters for the transformer model.
* Train the LSTM model on the entire Amazon Fashion review dataset.
