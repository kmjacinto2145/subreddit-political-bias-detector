#library import
import praw
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from flask import Flask, request, render_template, redirect, url_for

#Gets access to the Reddit API
reddit = praw.Reddit(client_id='6W9IFzMWNSjnUg', client_secret='wOIDztiKNDJFB1WdnOQR03_BDXgZlg', user_agent='mick43152')

#Parameters
SCRAPE_POST_NUMBER = 100

def scrape(subreddit, stops):

    posts = []

    #Scrapes posts
    scraped = reddit.subreddit(subreddit).top(limit = SCRAPE_POST_NUMBER)

    print("Scraping r/%s..." % subreddit)
    #Gets post data
    try:
        for post in scraped:
            posts.append(post.title)
    except:
        return False
    posts = pd.Series(posts)

    doc = posts.str.cat(sep = " ") #Gets a large string with all the post titles concatenated

    #Removes stopwords

    doc_tokens = word_tokenize(doc)
    tokens_without_sw = [word for word in doc_tokens if not word in stops]
    separator = ' '
    doc_without_sw = separator.join(tokens_without_sw)

    return doc_without_sw

def doc_similarity(doc1, doc2):
    tfidf = TfidfVectorizer().fit_transform([doc1,doc2])
    pairwise_similarity = tfidf * tfidf.T
    return round(pairwise_similarity.toarray()[1][0],2)

#Starting parameters
political_subreddits = ["Conservative", "Libertarian", "Socialism", "Liberal", "Marxism", "Progressive", "Capitalism", "Anarchism"]
stop = stopwords.words('english')

#Gets dictionary of subreddits and documents
political_subreddit_docs = {}
for subreddit in political_subreddits:
    political_subreddit_docs[subreddit] = scrape(subreddit, stop)

app = Flask(__name__)

@app.route('/', methods = ["GET", "POST"])
def main_menu():
    return render_template("menu.html")

@app.route('/results', methods = ["GET", "POST"])
def results():
    subreddit = request.form['subreddit']
    searched = scrape(subreddit, stop)

    if not searched:
        return redirect(url_for('sub_not_found'))

    similarities = {}
    for sub in political_subreddits:
        similarities[sub] = doc_similarity(searched, political_subreddit_docs[sub])

    similarities = {k: v for k, v in sorted(similarities.items(), key = lambda x: x[1], reverse = True)}

    return render_template("results.html", subreddit = subreddit, similarities = similarities)

@app.route('/sub-not-found', methods = ["GET", "POST"])
def sub_not_found():
    return render_template("sub_not_found.html")

if __name__ == "__main__":
    app.run(debug = False)
