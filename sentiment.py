import tweepy
import pandas as pd
import configparser
import re
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS
import streamlit as st
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
import pytz
import nltk
import string

emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"  # flags (iOS)
                           "]+", flags=re.UNICODE)


def twitter_connection():

    config = configparser.ConfigParser()
    config.read("config.ini")

    api_key = config["twitter"]["api_key"]
    api_key_secret = config["twitter"]["api_key_secret"]
    access_token = config["twitter"]["access_token"]

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    api = tweepy.API(auth)

    return api


api = twitter_connection()


def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('#', '', text)  # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    text = re.sub('https?:\/\/\S+', '', text)
    text = re.sub("\n", "", text)  # Removing hyperlink
    text = re.sub(":", "", text)  # Removing hyperlink
    text = re.sub("_", "", text)  # Removing hyperlink

    text = emoji_pattern.sub(r'', text)

    return text


def clean_text(text):
    # remove puntuation
    text_lc = "".join([word.lower()
                      for word in text if word not in string.punctuation])
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    # remove stopwords and stemming
    text = [ps.stem(word) for word in tokens if word not in stopword]
    return text


def extract_mentions(text):
    text = re.findall("(@[A-Za-z0–9\d\w]+)", text)
    return text


def extract_hashtag(text):
    text = re.findall("(#[A-Za-z0–9\d\w]+)", text)
    return text


def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

# Create a function to get the polarity


def getPolarity(text):
    return TextBlob(text).sentiment.polarity


def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'

# Removing Punctuation


def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text

  # Appliyng tokenization


def tokenization(text):
    text = re.split('\W+', text)
    return text


  # Removing stopwords
stopword = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    return text


 # Appliyng Stemmer
ps = nltk.PorterStemmer()


def stemming(text):
    text = [ps.stem(word) for word in text]
    return text


@st.cache(allow_output_mutation=True)
def preprocessing_data(word_query, number_of_tweets, function_option):

    if function_option == "Search By #Tag and Words":
        posts = tweepy.Cursor(api.search_tweets, q=word_query, count=200,
                              lang="en", tweet_mode="extended").items((number_of_tweets))

    if function_option == "Search By Username":
        posts = tweepy.Cursor(api.user_timeline, screen_name=word_query,
                              count=200, tweet_mode="extended").items((number_of_tweets))

    data = pd.DataFrame(
        [tweet.full_text for tweet in posts], columns=['Tweets'])

    data["mentions"] = data["Tweets"].apply(extract_mentions)
    data["hashtags"] = data["Tweets"].apply(extract_hashtag)

    data['retweets'] = data['Tweets'].str.extract(
        '(RT[\s@[A-Za-z0–9\d\w]+)', expand=False).str.strip()

    data['Tweets'] = data['Tweets'].apply(cleanTxt)

    #data['Tweets'] = data['Tweets'].apply(remove_punct)
    #data['Tweets'] = data['Tweets'].apply(tokenization)
    #data['Tweets'] = data['Tweets'].apply(remove_stopwords)
    #data['Tweets'] = data['Tweets'].apply(stemming)

    data['Subjectivity'] = data['Tweets'].apply(getSubjectivity)
    data['Polarity'] = data['Tweets'].apply(getPolarity)

    data['Analysis'] = data['Polarity'].apply(getAnalysis)

    return data


def download_data(data, label):
    current_time = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    current_time = "{}.{}-{}-{}".format(current_time.date(),
                                        current_time.hour, current_time.minute, current_time.second)
    export_data = st.download_button(
        label="Download {} data as CSV".format(label),
        data=data.to_csv(),
        file_name='{}{}.csv'.format(label, current_time),
        mime='text/csv',

    )
    return export_data


def analyse_mention(data):

    mention = pd.DataFrame(data["mentions"].to_list()).add_prefix("mention_")

    try:
        mention = pd.concat(
            [mention["mention_0"], mention["mention_1"], mention["mention_2"]], ignore_index=True)
    except:
        mention = pd.concat([mention["mention_0"]], ignore_index=True)

    mention = mention.value_counts().head(10)

    return mention


def countvector(data):
    # Applying Countvectorizer
    countVectorizer = CountVectorizer(analyzer=clean_text)
    countVector = countVectorizer.fit_transform(data['Tweets'])
    count_vect_df = pd.DataFrame(
        countVector.toarray(), columns=countVectorizer.get_feature_names())

    count = pd.DataFrame(count_vect_df.sum())
    countdf = count.sort_values(0, ascending=False).head(20)

    return countdf


def analyse_hashtag(data):

    hashtag = pd.DataFrame(data["hashtags"].to_list()).add_prefix("hashtag_")

    try:
        hashtag = pd.concat(
            [hashtag["hashtag_0"], hashtag["hashtag_1"], hashtag["hashtag_2"]], ignore_index=True)
    except:
        hashtag = pd.concat([hashtag["hashtag_0"]], ignore_index=True)

    hashtag = hashtag.value_counts().head(10)

    return hashtag


def graph_sentiment(data):

    analys = data["Analysis"].value_counts().reset_index(
    ).sort_values(by="index", ascending=False)

    return analys


def subjectivity(data):
    subject = data["Subjectivity"].value_counts().reset_index(
    ).sort_values(by="index", ascending=False)

    return subject


def gen_wordcloud(data):
    allWords = ' '.join([twts for twts in data['Tweets']])
    wordCloud = WordCloud(width=500, height=300, random_state=21,
                          max_font_size=110).generate(allWords)
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('WC.jpg')
    img = Image.open("WC.jpg")
    return img


def gen_poswordcloud(data):
    allWords = ' '.join(
        twts for twts in data[data['Analysis'] == "Positive"].Tweets)
    wordCloud = WordCloud(width=500, height=300, random_state=21,
                          max_font_size=110).generate(allWords)
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('WC1.jpg')
    img1 = Image.open("WC1.jpg")
    return img1


def gen_neuwordcloud(data):
    allWords = ' '.join(
        twts for twts in data[data['Analysis'] == "Neutral"].Tweets)
    wordCloud = WordCloud(width=500, height=300, random_state=21,
                          max_font_size=110).generate(allWords)
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('WC2.jpg')
    img2 = Image.open("WC2.jpg")
    return img2


def gen_negwordcloud(data):
    allWords = ' '.join(
        twts for twts in data[data['Analysis'] == "Negative"].Tweets)
    wordCloud = WordCloud(width=500, height=300, random_state=21,
                          max_font_size=110).generate(allWords)
    plt.imshow(wordCloud, interpolation="bilinear")
    plt.axis('off')
    plt.savefig('WC3.jpg')
    img3 = Image.open("WC3.jpg")
    return img3
