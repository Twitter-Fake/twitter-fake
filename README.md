### Dataset URL : http://mib.projects.iit.cnr.it/dataset.html

# Methodology & Reasoning For The Approaches
## Data  Pre-processing 

### Code Files : EDA_train_data.ipynb , load_baseline_train_data.py,load_train_data.py,online_features.py
The initial dataset was raw and was organized as a collection from various sources ( TWT,TFP,E13 etc ) , some were human users , some were fake. Their corresponding tweets is also included as per source.
We first converted the data ( load_train_data.py ) into a single data frame, with users and their corresponding tweets (multiple tweets were concatenated in a single string ) . After that we added the label column distinguish a genuine user and fake user. The source from which we read the data gives us the label information ( For example if we read the csv from TWT_Fake_Users then we are reading fake users) 
After we form a single data frame, we store it in a csv file ( user_training_tweet.csv ). Then we clean the tweets by removing stop words ,URL, special characters ( using regex matching and simple filtering ) ( online_features.py ). 
Finally for final testing purposes we have a tweet downloader script ( tweet_downloader.py ) which uses the external Twitter API to get additional genuine tweets in case its required to augment data augmentation.

## Exploratory Data Analysis (EDA_train_data.ipynb )
To understand the users and the nature of tweets we did a EDA on the train data csv file. We first sampled some genuine user tweets and visualized the words with a word cloud, then we sampled some fake users tweets and visualized the same. The language usage and the words used were somewhat strong in the fake tweets. 
Then on the whole dataset we used a simple TF-IDF Representation of the tweets and used K-Means Clustering to group the tweets into specific clusters. We found some pure clusters ( Where all tweets were fake/Genuine ) and the top 10 words among those clusters clearly indicated the difference. 
From the this we concluded that understanding the textual information of the tweet along with the user features can definitely give a better understanding of the current user and can better classify if the user is genuine or fake.

### Additional Features to represent the tweet ( online_features.py )
To extract the additional features from the tweet we attempted the following representations
* Contextual Features ( Tweet Replies, Retweets,etc)
* Sentiment analysis of tweets with the contextual features
* TF-IDF Representation of the tweets( 2000 Words)
* LDA Topic Modelling ( 20 Topics to group the tweets)
* Glove Pre-Trained Encoding ( 200 Dimensions) ( These are word representations, we averaged them out to convert it to sentence representation for a tweet )
* BERT Sentence Encoding (768 Dimensions ) ( Detailed Explanation – BERT.ipynb )

## Models
Using the Different Representations above,  we tried various models . We used the user information ( Without tweets ) with Logistic Regression as a classifier as the baseline model.
We then used several classifiers , trained them with the user data and with the combined user and tweet data ( Using the above representation)


## Result
