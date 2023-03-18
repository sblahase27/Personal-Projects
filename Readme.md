# Youtube Data Analysis


## Overview

The aim of this project is to perform data analysis on the US YouTube data for videos uploaded. This project utilizes various libraries such as NumPy, Pandas, Seaborn, Matplotlib, NLTK, Wordcloud, TextBlob, Emoji, Plotly, etc.

## Installation

This project is implemented in a Conda environment. The environment file is available in project repo. To use the project, clone the repository, and install the dependencies using the following command:

`
conda env create -f environment.yml
`

## Data

The project uses a CSV file, UScomments.csv, which contains US YouTube data for videos uploaded.

## Steps Involved

The project involves working on the following major steps:

### Step 1: Sentiment Analysis

- In this step, sentiment analysis is performed using TextBlob, which is an NLP library built on top of NLTK. 
- Part B includes sentiment analysis using NLTK Vader SentimentAnalyzer. 
- An alternate approach is also provided here to solve the problem effectively by writing a function.

```python 
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from nltk.tokenize import RegexpTokenizer
import re

def getting_polarity(category):
    tags=df[df['category_columns']==category]['tags']
    tags=tags.str.lower()
    tags_all=tags.str.cat()
    ### ^-except than A-Za-z  ,whatever we have just replace it with space
    tags_word = re.sub('[^A-Za-z]+', ' ', tags_all)
    ### lets create a list from tags_word so that we can remove stopwords from tags_word
    tokens_word = word_tokenize(tags_word)
    
    eng_stopwords = list(stopwords.words('english'))
    ge_stopwords = list(stopwords.words('german'))   
    fra_stopwords = list(stopwords.words('french'))   
    rus_stopwords= list(stopwords.words('russian')) 
    
    eng_stopwords.extend(ge_stopwords)
    eng_stopwords.extend(fra_stopwords)
    eng_stopwords.extend(rus_stopwords)
    filtered_words = [w for w in tokens_word if not w in eng_stopwords]
    ### lets consider those words who have atleast 3 chars or more..

    without_single_double_chr = [word for word in filtered_words if len(word) > 2]
    
    # Remove numbers
    cleaned_data_title = [word for word in without_single_double_chr if not word.isdigit()]

    ##Calculate frequency distribution
    word_dist = nltk.FreqDist(cleaned_data_title)
    
    dist = pd.DataFrame(word_dist.most_common(100),
                columns=['Word', 'Frequency'])
    polarities=[]
    compound = .0
    for word in dist['Word']:
        compound += SentimentIntensityAnalyzer().polarity_scores(word)['compound']
    return compound
    
    
```

### Step 2: Wordcloud Analysis

- In this step, we perform EDA for positive and negative sentences. 
- We found that for negative comments, users are emphasizing more on terrible, worst, horrible, boring, disgusting, etc. 
- For positive comments, users are emphasizing more on best, awesome, perfect, beautiful, etc.

### Step 3: Perform Emoji Analysis

- In this step, we perform analysis using the emoji dictionary to compute the frequency of each emoji in the comments. 
- We use the 'emoji' library to process the emojis in the comments.

### Step 4: Collect Entire Data of Youtube

- In this step, we combine all the youtube data from different countries, which are in the form of CSV and JSON. 
- We use the 'pandas' library to read and combine data from different sources.

### Step 5: Video Category with Maximum Likes

- In this step, we find out the top 5 categories of YouTube videos with the maximum number of likes. 
- We use the 'pandas' library to group the data by category and compute the total number of likes in each category. Then we plot a bar graph to visualize the results.

### Step 6: Audience Engagement

- In this step, we find out whether the audience is engaged or not by analyzing the correlation between views, likes, and dislikes. 
- We use the 'pandas' library to compute the correlation matrix and plot a heatmap to visualize the results.

### Step 7: Channels with Trending Videos

- In this step, we find out which channels have the largest number of trending videos. 
- We use the 'pandas' library to group the data by the channel and compute the total number of trending videos in each channel. Then we plot a bar graph to visualize the results.

### Step 8: Relation between Punctuation, Video Tags, Views, Comments, Likes, and Dislikes

- In this step, we find out whether punctuations in the title and tags have any relation with views, likes, dislikes, and comments. 
- We use the 'pandas' library to compute the correlation matrix and plot a heatmap to visualize the results.

## Libraries Used

The following libraries are used in this project:

- NumPy: used for numerical operations on data
- Pandas: used for data manipulation and analysis
- Seaborn: used for statistical data visualization
- Matplotlib: used for plotting graphs and charts
- NLTK: used for natural language processing tasks
- TextBlob: used for sentiment analysis
- WordCloud: used for generating word clouds
- Emoji: used for processing emojis in the comments
- Plotly: used for interactive data visualization

## Environment

This project uses the Conda environment for managing the dependencies. The environment file is uploaded to Github, which contains all the required libraries and their versions. You can use the following command to create the environment:

`
conda env create -f environment.yml
`

## References

- NumPy: https://numpy.org/doc/stable/
- Pandas: https://pandas.pydata.org/docs/
- Seaborn: https://seaborn.pydata.org/
- Matplotlib: https://matplotlib.org/stable/contents.html
- NLTK: https://www.nltk.org/
- TextBlob: https://textblob.readthedocs.io/en/dev/
- Wordcloud: https://amueller.github.io/word_cloud/
- Emoji: https://pypi.org/project/emoji/
- Plotly: https://plotly.com/python/
- RegexpTokenizer: https://www.nltk.org/api/nltk.tokenize.html#module-nltk.tokenize.regexp
- Stopwords: https://www.nltk.org/book/ch02.html#stopwords-index
- SentimentIntensityAnalyzer: https://www.nltk.org/api/nltk.sentiment.html#module-nltk.sentiment.vader

## Conclusion

In this project, we analyzed the YouTube data using various data analysis techniques. We performed sentiment analysis, word cloud analysis, emoji analysis, and computed the frequency of different features such as likes, dislikes, comments, and views. We also analyzed the correlation between these features and performed channel and category-wise analysis. This project can be extended further by adding more data sources and performing more advanced analysis techniques.