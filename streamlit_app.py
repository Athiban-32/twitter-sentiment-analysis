from attr import has
import streamlit as st
import matplotlib.pyplot as plt
from sentiment import preprocessing_data, graph_sentiment, analyse_mention, analyse_hashtag, download_data, gen_wordcloud, gen_poswordcloud, gen_neuwordcloud, gen_negwordcloud, subjectivity, countvector

st.set_page_config(
    page_title="Twitter sentiment analysis",
    page_icon="🕸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "## A Twitter sentiment analysis webappTwitter Sentment Analysis Web App using"
    }
)


st.title("Twitter Sentimental Analysis")

function_option = st.sidebar.selectbox("Select The Funtionality: ", [
                                       "Search By #Tag and Words", "Search By Username"])

if function_option == "Search By #Tag and Words":
    word_query = st.text_input("Enter the Hashtag or any topic")

if function_option == "Search By Username":
    word_query = st.text_input("Enter the Username without @ ")

number_of_tweets = st.slider("How many tweets You want to collect from {}".format(
    word_query), min_value=100, max_value=10000)
st.info("1 Tweets takes approx 0.05 sec so you may have to wait {} minute for {} Tweets".format(
    round((number_of_tweets*0.05/60), 2), number_of_tweets))

if st.button("Analysis Sentiment"):

    data = preprocessing_data(word_query, number_of_tweets, function_option)
    analyse = graph_sentiment(data)
    mention = analyse_mention(data)
    hashtag = analyse_hashtag(data)
    img = gen_wordcloud(data)
    img1 = gen_poswordcloud(data)
    img2 = gen_neuwordcloud(data)
    img3 = gen_negwordcloud(data)
    subject = subjectivity(data)
    countdf = countvector(data)

    st.write(" ")
    st.header("Extracted and Preprocessed Tweets")
    st.write(data)
    download_data(data, label="twitter_sentiment_filtered")
    st.write(" ")

    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("### Exploratory data analysis on the Tweets")

    col1, col2 = st.columns(2)

    with col1:

        st.text("Top 10 Hashtags used in {} tweets".format(number_of_tweets))
        st.bar_chart(hashtag)
    with col2:

        st.text("Most used words in the Tweets")
        st.bar_chart(countdf[1:11])

    col3, col4 = st.columns([1.5, 1])
    with col3:
        st.text("Wordcloud for {} tweets".format(number_of_tweets))
        st.image(img)
    with col4:
        st.text("Top 10 @Mentions in {} tweets".format(number_of_tweets))
        st.bar_chart(mention)
    col5, col6, col7 = st.columns(3)
    with col5:
        st.text("Wordcloud for Positive tweets")
        st.image(img1)
    with col6:
        st.text("Wordcloud for Neutral tweets")
        st.image(img2)
    with col7:
        st.text("Wordcloud for Negative tweets")
        st.image(img3)
    st.subheader("Twitter Sentment Analysis")
    st.bar_chart(analyse)
