from Main.twitter import *
from Main.training import *
from Main.processing import *


if __name__ == '__main__':
    freqs = count_tweets({}, train_x, train_y)
    logprior, loglikelihood = train_naive_bayes(freqs, train_x, train_y)
    my_tweet2 = 'Bring our great gaming experience back.Make a game like FIFA 16 stop this shit!!!!'
    p = naive_bayes_predict(my_tweet2, logprior, loglikelihood)
    print(p)
    username = input("Enter user name: ")
    get_tweets(username, logprior, loglikelihood)
