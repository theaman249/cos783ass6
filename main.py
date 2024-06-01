from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

flagged_negative_tweets = []
flagged_neutral_tweets = []

#samples
# tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"
#tweet = 'Going to commit this crime'
# tweet = 'I killed a man'
tweet = 'selling cocaine. It is bad for my clients. But I am making money'
# tweet = 'So happy for my life'

def pre_process_data():
    tweet_words = []
    for word in tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    
    return tweet_words

tweet_proc = " ".join(pre_process_data)

# load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)

labels = ['Negative', 'Neutral', 'Positive']

# sentiment analysis
encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
# output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
output = model(**encoded_tweet)

scores = output[0][0].detach().numpy()
scores = softmax(scores)

for i in range(len(scores)):
    
    l = labels[i]
    s = scores[i]
    print(l,s)
