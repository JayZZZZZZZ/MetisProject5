
import streamlit as st
st.set_page_config(
    page_title='Airbnb Application',
    page_icon='☮️',
    layout='wide'
)

from IPython.display import Image, HTML
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import re 
import string

import nltk 
from nltk import word_tokenize,sent_tokenize
from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.corpus import wordnet
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import TweetTokenizer

from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')


plt.style.use("seaborn")

#st.header("Streamlit app working ...!")

#map_data = pd.DataFrame(
   # np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
   # columns=['lat', 'lon'])

data = pd.read_csv('/Users/jayz/Documents/GitHubMetis/Final_Project/listings.csv')
data.head()

# Layingout the top section for the app #

#row1_1,row_1_2 = st.beta_columns((5,5))

#with row1_1:
    #st.title ("Airbnb Recommendation System")
    #price_selected = st.slider ("Select Price Range:",0,1000)
    #people_selected = st.slider ("Select Number of People:",0,50)

#with row1_1:
st.title ("Airbnb Recommendation System")
    #price_selected = st.slider ("Select Price Range:",0,1000)

row2_1,row2_2 = st.beta_columns((5,5)) 

with row2_1:
    selected_price_range = st.slider('Price Range Per Night', 0, 999, (0, 999), 1)
with row2_2:
    people_selected = st.slider ("Number of People:",0,50)


#with row1_2:
 #   st.write(
  #      """ ## Using NLP to recommend houses for Users"""
   # )

row3_1,row3_2 = st.beta_columns((9,1))


data['price']=data['price'].replace('[\$,]','',regex=True).astype(float).astype(int)
min_price = int(data['price'].min())
max_price = int(data['price'].max())

data=data[(data.price.between(selected_price_range[0], selected_price_range[1]))]

data2= data[data['accommodates']<= people_selected]



dfO= data2[['name','id','description','space','latitude','longitude','price','accommodates','host_url','picture_url']]

dfO = pd.DataFrame(dfO)

#st.map(dfO)
    
from streamlit_folium import folium_static
import folium


# center on Liberty Bell
m=folium.Map(location= [dfO['latitude'][1],dfO['longitude'][1]],zoom_start = 14)

# add marker for Liberty Bell
#tooltip = "Liberty Bell"

def circle_maker(x):
    folium.Circle(location = [x['latitude'],x['longitude']],popup= '{}\nListing Price: {}'.format(x['name'],x['price'])).add_to(m)

dfO.apply(lambda x: circle_maker(x),axis=1)
# call to render Folium map in Streamlit

with row3_1:
    folium_static(m)

#import streamlit.components.v1 as components

#st.header



listings=dfO
listings['name'] = listings['name'].astype('str')
listings['description'] = listings['description'].astype('str')
listings['space'] = listings['space'].astype('str')
listings['content'] = listings[['name', 'description','space']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
listings['content'].fillna('Null', inplace = True)
#Cleaning Text - round 1#

def clean_text_round1(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text
#Cleaning Text - round 2#

def clean_text_round2(text):
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    return text
# Personalizing Stopwords #
stopword1 = nltk.corpus.stopwords.words('english')
newStopWords = ['im','ive','theyre','hum','washington','seattle','needle','belltown','nan','would','could']
stopword1.extend(newStopWords)


stop = set(stopword1)
punctuation = list(string.punctuation)

stop.update(punctuation)
#exclude_words = set(("not", "no","aren't","couldn't","didn't","doesn't","don't","hasn't","haven't","mightn't","mustn't","needn't","shouldn't","weren't"))

#stop = stop - exclude_words

# Pre-processing & Lemmatizing text #

def get_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
    
# Lemmatizing words that are not stopwords

lemmatizer = WordNetLemmatizer()
def lemmatization1(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stop:
            pos = pos_tag([i.strip()])
            word = lemmatizer.lemmatize(i.strip(),get_pos(pos[0][1]))
            final_text.append(word.lower())
    return " ".join(final_text)

listings['content']=listings['content'].apply(lambda x:clean_text_round1(x))
listings['content']=listings['content'].apply(lambda x:clean_text_round2(x))
listings['content'] = listings['content'].apply(lambda x : lemmatization1(x))


tf = TfidfVectorizer(max_df=0.4,stop_words='english')
tfidf_matrix = tf.fit_transform(listings['content'])

from sklearn.decomposition import NMF
nmf_model=NMF(5)
doc_topic=nmf_model.fit_transform(tfidf_matrix)

topic_word=nmf_model.components_
words = tf.get_feature_names()
t = nmf_model.components_.argsort(axis=1)[:,-1:-10:-1]
topic_words = [[words[e] for e in l] for l in t]

# Family_House
# Tourism_and_Convention
# Peaceful_Vacation
# Young
# Specialty_Needs

doc_topic_nmf = pd.DataFrame(doc_topic.round(5),index = listings['id'],columns = ["component_1","component_2","component_3","component_4","component_5" ])

from sklearn.metrics import pairwise_distances

desc = "Uses an NMF Model to generate recommendations   *. Check out the code [here](URL)!"
st.write(desc)

num = st.number_input('Number of Recommendations', min_value=1, max_value=20, value=5)
t = [st.text_input('Please Describe What You are Looking For In The House','Peaceful Place')]

#num = 5
#t = ['great house close to downtown']

vt = tf.transform(t)
tt = nmf_model.transform(vt)


def topic_result(tt):
    predicted_topics=[np.argsort(each)[::-1][0] for each in tt]
    if predicted_topics==[0]:
        return 'Family House'
    elif predicted_topics==[1]:
        return 'Tourism and Event'
    elif predicted_topics==[2]:
        return 'Peaceful Vacation'
    elif predicted_topics==[3]:
        return 'Young Life Style'
    else:
        return 'Specialty Needs'

similar_indices=pairwise_distances(tt,doc_topic,metric='cosine').argsort()[0][0:num]

similar_id = [(listings['id'].iloc[i]) for i in similar_indices]

similar_name = [(listings['name'].iloc[i]) for i in similar_indices]

similar_desc = [(listings['description'].iloc[i][0:165]) for i in similar_indices]

similar_url = [(listings['host_url'].iloc[i]) for i in similar_indices]

similar_pic = [(listings['picture_url'].iloc[i]) for i in similar_indices]

# def recommend(text,num):
#     a=('Recommending ' + str(num) + ' Airbnb products for ' + str(text))
#     b=('------------------------------------')
#     for i in list(range(0,num)):
#         c=('Recommended: ' + str(similar_name[i]))
#         d=('Description: ' + str(similar_desc[i]) +'...')
#         e=('HostUrl: ' + str(similar_url[i]))
#         f=('Picture: ' + str(similar_pic[i]))
#         return a,b,c,d,e,f

#st.markdown('---')
#with st.spinner("Generating results, please be patient, this can take quite a while. If you adjust anything, you may need to start from scratch."):    



# if st.button('Generate Recommendation'):
#     result = st.markdown(recommend(t,num))
#     st.write(result)

def recommend(text,num):
    #a=str('Recommending ' + str(num) + ' Airbnb products for ' + str(text))
    #b=str('------------------------------------')
    mylistc=[]
    mylistd=[]
    myliste=[]
    mylistf=[]
    for i in range(0,num):
        c=str(similar_name[i])
        d=str(str(similar_desc[i]) +'...')
        e=str(similar_url[i])
        f=str(similar_pic[i])
        mylistc.append(c)
        mylistd.append(d)
        myliste.append(e)
        mylistf.append(f)
    return mylistc,mylistd,myliste,mylistf

#row1_1,row_1_2 = st.beta_columns((5,5))

# if st.button('Generate Recommendation'):
#     result = st.markdown(recommend(t,num))
#     st.markdown(result)

# def write_matrix(outputs):
#     table_md = f'''
#     |Recommended|Description|Url|Picture|
#     |--|--|--|--|
#     |**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|
#     |**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|
#     |**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|
#     |**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|
#     |**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|**{outputs['hi']}**|
#     '''
#     st.markdown(table_md)

l=recommend(t,num)
df =pd.DataFrame(list(zip(l[0],l[1],l[2],l[3])),columns=['Recommended','Description','Link Url','Picture'])
pd.set_option('display.max_colwidth',None)


#revew1#
reviewd=pd.read_csv('/Users/jayz/Documents/GitHubMetis/Final_Project/reviews.csv')
reviewd1=reviewd[reviewd['listing_id']==similar_id[0]]
reviewd1['comments']=reviewd1['comments'].apply(lambda x:clean_text_round1(x))
reviewd1['comments']=reviewd1['comments'].apply(lambda x:clean_text_round2(x))
reviewd1['comments']=reviewd1['comments'].apply(lambda x : lemmatization1(x))
text1 = reviewd1['comments']
wordcloud = WordCloud(width=320, height=150, max_font_size=30, min_font_size=5,background_color = 'white',
    stopwords = stop).generate(str(text1))
fig = plt.figure(
    figsize = (60, 30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.savefig('image1.png',dpi=60)
plt.show()

#review2#
reviewd2=reviewd[reviewd['listing_id']==similar_id[1]]
reviewd2['comments']=reviewd2['comments'].apply(lambda x:clean_text_round1(x))
reviewd2['comments']=reviewd2['comments'].apply(lambda x:clean_text_round2(x))
reviewd2['comments']=reviewd2['comments'].apply(lambda x : lemmatization1(x))
text2 = reviewd2['comments']
wordcloud = WordCloud(width=320, height=150, max_font_size=30, min_font_size=5,background_color = 'white',
    stopwords = stop).generate(str(text2))
fig = plt.figure(
    figsize = (60, 30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.savefig('image2.png',dpi=60)
plt.show()

#review3#
reviewd3=reviewd[reviewd['listing_id']==similar_id[2]]
reviewd3['comments']=reviewd3['comments'].apply(lambda x:clean_text_round1(x))
reviewd3['comments']=reviewd3['comments'].apply(lambda x:clean_text_round2(x))
reviewd3['comments']=reviewd3['comments'].apply(lambda x : lemmatization1(x))
text3 = reviewd3['comments']
wordcloud = WordCloud(width=320, height=150, max_font_size=30, min_font_size=5,background_color = 'white',
    stopwords = stop).generate(str(text3))
fig = plt.figure(
    figsize = (60, 30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.savefig('image3.png',dpi=60)
plt.show()

#review4#
reviewd4=reviewd[reviewd['listing_id']==similar_id[3]]
reviewd4['comments']=reviewd4['comments'].apply(lambda x:clean_text_round1(x))
reviewd4['comments']=reviewd4['comments'].apply(lambda x:clean_text_round2(x))
reviewd4['comments']=reviewd4['comments'].apply(lambda x : lemmatization1(x))
text4 = reviewd4['comments']
wordcloud = WordCloud(width=320, height=150, max_font_size=30, min_font_size=5,background_color = 'white',
    stopwords = stop).generate(str(text4))
fig = plt.figure(
    figsize = (60, 30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.savefig('image4.png',dpi=60)
plt.show()

#review5#
reviewd5=reviewd[reviewd['listing_id']==similar_id[4]]
reviewd5['comments']=reviewd5['comments'].apply(lambda x:clean_text_round1(x))
reviewd5['comments']=reviewd5['comments'].apply(lambda x:clean_text_round2(x))
reviewd5['comments']=reviewd5['comments'].apply(lambda x : lemmatization1(x))
text5 = reviewd5['comments']
wordcloud = WordCloud(width=320, height=150, max_font_size=30, min_font_size=5,background_color = 'white',
    stopwords = stop).generate(str(text5))
fig = plt.figure(
    figsize = (60, 30))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.savefig('image5.png',dpi=60)
plt.show()


from PIL import Image
image1=Image.open('image1.png')
image2=Image.open('image2.png')
image3=Image.open('image3.png')
image4=Image.open('image4.png')
image5=Image.open('image5.png')


if st.button('Generate Recommendation'):

    st.markdown('Your Topic based on Text is: ' + topic_result(tt))
    st.table(df[['Recommended','Description','Link Url']])


    col1, col2, col3, col4, col5 = st.beta_columns(5)

    with col1:
        st.header('Recommended #1')
        st.image(df['Picture'][0])

    with col2:
        st.header('Recommended #2')
        st.image(df['Picture'][1])

    with col3:
        st.header('Recommended #3')
        st.image(df['Picture'][2])

    with col4:
        st.header('Recommended #4')
        st.image(df['Picture'][3])

    with col5:
        st.header('Recommended #5')
        st.image(df['Picture'][4])

    col6, col7, col8, col9, col10 = st.beta_columns(5)

    with col6:
        st.header('Review Cloud #1')
        st.image(image1)

    with col7:
        st.header('Review Cloud #2')
        st.image(image2)

    with col8:
        st.header('Review Cloud #3')
        st.image(image3)

    with col9:
        st.header('Review Cloud #4')
        st.image(image4)

    with col10:
        st.header('Review Cloud #5')
        st.image(image5)