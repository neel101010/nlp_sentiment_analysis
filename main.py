import numpy as np
import nltk
from nltk.corpus import stopwords, twitter_samples
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import string
import regex as re
import tweepy
import settings

# nltk.download('stopwords')
# nltk.download('twitter_samples')


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)
    tweets_clean = []
    for word in tweet_tokens:
        if word not in stopwords_english and  word not in string.punctuation:  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)
    return tweets_clean


def lookup(freqs, word, label):
    n = 0
    pair = (word, label)
    if pair in freqs:
        n = freqs[pair]
    return n


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))


def count_tweets(result, tweets, ys):
    for y, tweet in zip(ys, tweets):
        for word in process_tweet(tweet):
            pair = (word,y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result


def train_naive_bayes(freqs, train_x, train_y):
    loglikelihood = {}
    logprior = 0
    vocab = set([pair[0] for pair in freqs.keys()])
    V = len(vocab)
    N_pos = N_neg = 0
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[pair]
        else:
            N_neg += freqs[pair]
    D = len(train_y)
    D_pos = (len(list(filter(lambda x: x > 0, train_y))))
    D_neg = (len(list(filter(lambda x: x <= 0, train_y))))

    logprior = np.log(D_pos) - np.log(D_neg)

    for word in vocab:
        freq_pos = lookup(freqs,word,1)
        freq_neg = lookup(freqs,word,0)
        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    return logprior, loglikelihood


def naive_bayes_predict(tweet, logprior, loglikelihood):
    word_l = process_tweet(tweet)
    p = 0
    p += logprior

    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p


def test_naive_bayes(test_x, test_y, logprior, loglikelihood):
    accuracy = 0
    y_hats = []
    for tweet in test_x:
        if naive_bayes_predict(tweet, logprior, loglikelihood) > 0:
            y_hat_i = 1
        else:
            y_hat_i = 0
        y_hats.append(y_hat_i)
    error = np.mean(np.absolute(y_hats-test_y))
    accuracy = 1-error
    return accuracy


def process_individual_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    return tweet

# My neel bhagat  main account credentials
consumer_key = settings.consumer_key
consumer_secret = settings.consumer_secret
access_key = settings.access_key
access_secret = settings.access_secret


def get_tweets(username):
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)
    replies = []
    for full_tweets in tweepy.Cursor(api.user_timeline, screen_name=username, timeout=999999).items(10):
        for tweet in tweepy.Cursor(api.search, q='to:' + username, result_type='recent', timeout=999999).items(1000):
            if hasattr(tweet, 'in_reply_to_status_id_str'):
                if tweet.in_reply_to_status_id_str == full_tweets.id_str:
                    replies.append(tweet.text)
            t = process_individual_tweet(full_tweets.text)
            print(t)
            x = naive_bayes_predict(t, logprior, loglikelihood)
            if x > 0:
                print("Tweet : ", t, "Positive Tweet", x)
            else:
                print("Tweet : ", t, "Negative Tweet", x)
            for elements in replies:
                x = naive_bayes_predict(elements, logprior, loglikelihood)
                if x > 0:
                    print("Replies :", elements, "Positive Reply")
                else:
                    print("Replies :", elements, "Negative Reply")
            replies.clear()


if __name__ == '__main__':
    freqs = count_tweets({}, train_x, train_y)
    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
    my_tweet2 = 'Bring our great gaming experience back.Make a game like FIFA 16 stop this shit!!!!'
    p = naive_bayes_predict(my_tweet2, logprior, loglikelihood)
    print(p)
    username = input("Enter user name: ")
    get_tweets(username)
