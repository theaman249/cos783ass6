from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import csv
import tkinter as tk 
from tkinter import filedialog, messagebox
import pandas as pd

flagged_negative_tweets = []
flagged_neutral_tweets = []


labels = ['Negative', 'Neutral', 'Positive']

#samples
# tweet = "@MehranShakarami today's cold @ home 😒 https://mehranshakarami.com"
#tweet = 'Going to commit this crime'
# tweet = 'I killed a man'
# tweet = 'So happy for my life'
# tweet = ' @John selling cocaine. It is bad for my clients. But I am making money http://drugs.com'

# Due to the nature of our model, it won't allow @Cars or any http links. Thus, we 
# must clean up our tweets to only include the @user and http tag. i.e @John = @user and https://drugs.com = http
# The function takes in a tweet as a parameter
def pre_process_data(fn_tweet):
    tweet_words = []
    for word in fn_tweet.split(' '):
        if word.startswith('@') and len(word) > 1:
            word = '@user'
        
        elif word.startswith('http'):
            word = "http"
        tweet_words.append(word)
    
    return tweet_words

# load model and tokenizer

roberta = "cardiffnlp/twitter-roberta-base-sentiment"

model = AutoModelForSequenceClassification.from_pretrained(roberta) # the actual model
tokenizer = AutoTokenizer.from_pretrained(roberta) # convert our tweet text into tokens

# tweet processing area

def process_tweet(file_path):
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
    
    # Iterate over the rows in the CSV file
        for row in reader:
            user = row['user']
            tweet = row['tweet']
            timestamp = row['timestamp']
            # print(f"User: {user}, Tweet: {tweet}, Timestamp: {timestamp}")

            tweet_proc = " ".join(pre_process_data(tweet)) #after the tweet has been cleaned. Tweet processed.

            # sentiment analysis

            encoded_tweet = tokenizer(tweet_proc, return_tensors='pt') # pytorch tensor of the tweet processed. The encoded tweet is a dictionary 

            # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

            output = model(**encoded_tweet)
            scores = output[0][0].detach().numpy()  # Get the raw scores (logits)
            scores = softmax(scores)  # Apply the softmax function to convert logits to probabilities <Negative, Neutral, Positive>

            # Find the label with the highest score
            max_score_index = scores.argmax()
            sentiment = labels[max_score_index]

            # Flag tweets based on sentiment
            if sentiment == 'Negative':
                flagged_negative_tweets.append((user, tweet, timestamp))
            elif sentiment == 'Neutral':
                flagged_neutral_tweets.append((user, tweet, timestamp))

# Print flagged tweets for review
print("Flagged Negative Tweets:")
for tweet in flagged_negative_tweets:
    print(tweet)

print("\n Flagged Neutral Tweets:")
for tweet in flagged_neutral_tweets:
    print(tweet)

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            # Read the CSV file
            # df = pd.read_csv(file_path)
            process_tweet(file_path)
            # Display the content in the Text widget
            text.delete(1.0, tk.END)
            text.insert(tk.END, f"Negative Tweets:\n")
            for tweet in flagged_negative_tweets:
                text.insert(tk.END, f"{tweet}\n")

            text.insert(tk.END, f"\nNeutral Tweets:\n")
            for tweet in flagged_neutral_tweets:
                text.insert(tk.END, f"{tweet}\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read the file: {e}")


# Create the main window
root = tk.Tk()
root.title("Assignment Six")

# Create a button to upload the CSV file
button = tk.Button(root, text="Upload CSV", command=open_file)
button.pack(pady=10)

# Create a Text widget to display the CSV content
text = tk.Text(root, wrap='none')
text.pack(expand=True, fill='both')

# Add a scrollbar for the Text widget
scrollbar_y = tk.Scrollbar(root, orient='vertical', command=text.yview)
scrollbar_y.pack(side='right', fill='y')
text.config(yscrollcommand=scrollbar_y.set)

scrollbar_x = tk.Scrollbar(root, orient='horizontal', command=text.xview)
scrollbar_x.pack(side='bottom', fill='x')
text.config(xscrollcommand=scrollbar_x.set)

# Start the main event loop
root.mainloop()