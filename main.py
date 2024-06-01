from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

flagged_negative_tweets = []
flagged_neutral_tweets = []


labels = ['Negative', 'Neutral', 'Positive']

#samples
# tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"
#tweet = 'Going to commit this crime'
# tweet = 'I killed a man'
# tweet = 'So happy for my life'


# Due to the nature of our model, it won't allow @Cars or any http links. Thus, we 
# must clean up our tweets to only include the @user and http tag. i.e @John = @user and https://drugs.com = http
# The function takes in a tweet as a parameter
def pre_process_data(fn_tweet):
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    
    return tweet_words

# tweet processing area

tweet = ' @John selling cocaine. It is bad for my clients. But I am making money http://drugs.com'

tweet_proc = " ".join(pre_process_data(tweet)) #after the tweet has been cleaned. Tweet processed.


# load model and tokenizer

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta) # the actual model
tokenizer = AutoTokenizer.from_pretrained(roberta) # convert our tweet text into tokens

# sentiment analysis

encoded_tweet = tokenizer(tweet_proc, return_tensors='pt') # pytorch tensor of the tweet processed. The encoded tweet is a dictionary 

# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()  # Get the raw scores (logits)

scores = softmax(scores)  # Apply the softmax function to convert logits to probabilities <Negative, Neutral, Positive>


for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)
