from Main.processing import *
import tweepy
from Main.settings import *
from Main.testing import *
from Main.run import *


def get_tweets(username, logprior, loglikelihood):
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