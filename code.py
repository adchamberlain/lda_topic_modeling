# Using LDA Topic Modeling and Text Features to Predict "Helpful Votes" in Glassdoor Reviews
# Andrew Chamberlain, Ph.D.
# achamberlain.com
# February 2018

# Background on LDA topic modeling:
# http://rstudio-pubs-static.s3.amazonaws.com/79360_850b2a69980c4488b1db95987a24867a.html
# https://www.analyticsvidhya.com/blog/2016/08/beginners-guide-to-topic-modeling-in-python/
# https://datascience.stackexchange.com/questions/2464/clustering-of-documents-using-the-topics-derived-from-latent-dirichlet-allocatio

# Import packages.
import pymssql
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import yaml
import os
import sys    
import re
import nltk
from nltk.corpus import stopwords
from string import punctuation
from gensim import corpora, models
import gensim
from textstat.textstat import textstat
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import time
import string
from string import digits
from sklearn.model_selection import train_test_split

# Set seed.
np.random.seed(0)



################################################################
#### Import data file 
################################################################

df = pd.read_csv('df.csv')



################################################################
#### Clean text data 
################################################################

# List dataframe columns.
df.columns

# Replace NA values.
df['pros'].fillna(value='na',inplace=True)
df['cons'].fillna(value='na', inplace=True)
df['GOC'].fillna(value='cashier', inplace=True)
df['sector'].fillna(value='Retail', inplace=True)
df['metroID'].fillna(value='na', inplace=True)
df['OverallRating'].fillna(3, inplace=True) # Fill with median ratings value if missing. 
df['lengthOfEmployment'].fillna(0, inplace=True)
df['isCurrentJobFlag'].fillna(0, inplace=True)
df['employeesTotalNum'].fillna(0, inplace=True)
df['pvTotal'].fillna(0, inplace=True)





# Define function to combine text fields and clean out messy characters. 
# CHOOSE TO USE PROS, CONS, OR BOTH IN THIS CODE BLOCK.
def clean_text(df):
    """ text is a column with no special characters except single quotes, lowercased"""
#    df['text'] = df['pros'] # Pros only.
#    df['text'] = df['cons'] # Cons only.
    df['text'] = df['pros'] + ' ' + df['cons']
    df['text'] = df['text'].astype(str)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('401k', 'FourOneK')
    df['text'] = df['text'].str.replace('401(k)', 'FourOneK')
    df['text'] = df['text'].str.replace('401 k', 'FourOneK')
    df['text'] = df['text'].str.replace('401-k', 'FourOneK')
    df['text'] = df['text'].str.replace('%401%k', 'FourOneK')
    df['text'] = df['text'].str.replace('\'', '')
    df['text'] = df['text'].str.replace('%\'%', '')
    df['text'] = df['text'].str.replace('-', '')
    df['text'] = df['text'].str.replace('\n', ' ')
    df['text'] = df['text'].str.replace('\r\n', ' ')
    df['text'] = df['text'].str.replace('\\', ' ')
    df['text'] = df['text'].str.replace('\r', ' ')
    df['text'] = df['text'].str.replace('*', ' ')
    df['text'] = df['text'].str.replace('•', ' ')
    df['text'] = df['text'].str.replace('  ', ' ')
    df['text'] = df['text'].str.replace('  ', ' ')
    df['text'] = df['text'].str.strip()
    return df

# Run clean_text function.
df = clean_text(df)

# Remove all numerical characters. 
remove_digits = str.maketrans('', '', digits)  

strip_nums = str.maketrans('', '', digits)  
df['text'] = df['text'].apply(lambda x: x.translate(remove_digits))

# Print pro/con text of one review.
#df.loc[6,'pros']
#df.loc[6,'cons']
#df.loc[7,'text']

# Create features for grade level, reading ease, word counts, sentence count, and paragraphs. 
# Source: https://pypi.python.org/pypi/textstat/
# Note: \r = paragraph break. \n = white space. 
df['read_ease_grade'] = df['text'].apply(lambda x: textstat.flesch_kincaid_grade(x))
df['sentence_count'] = df['text'].apply(lambda x: textstat.sentence_count(x))
df['word_count'] = df['text'].apply(lambda x: textstat.lexicon_count(x))
df['word_count_squared'] = (df['word_count'])**2
df['paragraph'] = df['pros'].apply(lambda x: x.count('\r')) + df['cons'].apply(lambda x: x.count('\r'))
df['text_ratio'] = (df['textLengthPro'] - df['textLengthCon']) / ((df['textLengthPro'] + df['textLengthCon']))


################################################################
#### Stop words, tokenize, stemming.  
################################################################

#### Tokenize text. 
tokenizer = RegexpTokenizer(r'\w+')
df['tokens'] = df['text'].apply(lambda x: tokenizer.tokenize(x))

#### Stem text.
wordnet_lemmatizer = WordNetLemmatizer()
df['stemmed'] = df['tokens'].apply(lambda x: [wordnet_lemmatizer.lemmatize(y) for y in x])

#### Remove stop words.
en_stop = get_stop_words('en')

moreStops = ["wa","ha","arent","hadnt","hasnt","hed","hes","heres","hows","im",
              "isnt","its","lets","mustnt","shant","shed","shouldnt","thats","theyd","theyve",
              "wasnt","weve","werent","whats","whens","wheres","whos","whys","wont","wouldnt",
              "youd","youll","youve","havent","the","c","ive","you","i","d","doesnt",
              "na","m","ups","s","t","dont","w","youre", "didnt","a","cant","u",
              "don","re","hq", "co","r","hb","ve","n","g","st","csc", "mondo","ha","b","pc",
              "v", "therefore", "amongst", "job", "work", "get", "will","just","like", "know",
              "even","lot","can","make","working", "place","keep","really","thing","much",
              "many","everyone","go", "one", "rd", "every", "way", "also", "actually", "doe", "ever",
              "x"]

stops = en_stop + moreStops 

# Pull out all stop words. Ready for LDA model.
df['stopped'] = df['stemmed'].apply(lambda x: [i for i in x if not i in stops])

# Create set of bigrams, trigrams. 
df['bigrams'] = df['stopped'].apply(lambda x: ["_".join(w) for w in ngrams(x, 2) ])
df['trigrams'] = df['stopped'].apply(lambda x: ["_".join(w) for w in ngrams(x, 3) ])


################################################################
#### Fit LDA topic model.  
################################################################

# Grab tokens column from df. 
#texts = df['stopped']
#texts = df['bigrams'] # Use bigrams in LDA model.
texts = df['trigrams'] # Use trigrams in LDA model.

# Turn tokenized reviews into an id term dictionary.
dictionary = corpora.Dictionary(texts)

# Save dictionary from fit LDA model to load later.
#dictionary.save('dictionary.saved')

# Load saved dictionary.
#dictionary = corpora.Dictionary.load('dictionary.saved')

# Convert tokenized documents into a document-term matrix.
corpus = [dictionary.doc2bow(text) for text in texts]

# Run LDA model (Runs slowly). 
t1 = time.time() # Timer
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=25, eta='auto', id2word = dictionary, minimum_probability=0.05, passes=3, iterations = 50)
t2 = time.time()
print(t2-t1) # Timer
os.system('say "Your code is done."') # Code done alert.

# Save LDA model for loading later.
#ldamodel.save('lda.saved')

#Load saved LDA model to use for topic generation (fit on N = 100,000 GD reviews from 2012-2016). 
#ldamodel = gensim.models.ldamodel.LdaModel.load('lda.saved')

# Print LDA model results.
dist = ldamodel.print_topics(num_topics=25, num_words=5)
dist = pd.DataFrame(dist)
print(dist)
# dist.to_csv('dist.csv') # Examine word distributions by topic. 

# Get the distribution of topics present in each review. 
doc_topics = ldamodel.get_document_topics(corpus, minimum_probability=0.10) # Report only topics with p >= 0.10

# Push top topics for each review into a dataframe. 
top_topics = []
for i in doc_topics:
    top_topics.append(i)
top_topics = pd.DataFrame(top_topics)
#print(top_topics)

####### CODE NEEDS TO BE DE-BUGGED BELOW -- SOMETIMES TOP TOPIC ISN'T LISTED IN LEFT-MOST COLUMN. NEED TO SORT HORIZONTALLY FIRST, IN DESCENDING ORDER.
# Grab column of only the top topic for each review (in column 1), split into two columns for: topic, probability score.
df_topics = pd.DataFrame()
df_topics['topic'] = top_topics.iloc[:,0] # Grab first column only.
df_topics = df_topics['topic'].apply(pd.Series) # Split topic and prob. score into 2 columns.
df_topics.columns = ['topic', 'score']

# Merge in top topics to reviews data frame. 
df = pd.merge(df,df_topics, how='left', left_index=True, right_index=True)
df['topic'].fillna(0, inplace=True)
df['topic'] = df['topic'].apply(lambda x: int(x))



################################################################
#### Create supervised review topics (non-LDA topics).
################################################################

# Create clean text column to search on.
#### Define function to strip out punctuation.
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

df['text_clean'] = df['text'].apply(lambda x: strip_punctuation(x))

# Add white space to beginning and end of text
df['text_clean'] = df['text_clean'].apply(lambda x: " " + x + " ")

# Define 7 dictionaries of search terms to dummify. 
dict_pay = ['pay','compensation','wage','wages', 'paycheck', 'money', 'dollars']
dict_ben = ['benefit','benefits','health','FourOneK', 'vacation', 'retirement', 'time off', 'insurance']
dict_dyn = ['innovation','technology','entrepreneurial', 'dynamic', 'fast paced', 'innovative']
dict_op = ['opportunity', 'growth', 'learning', 'training', 'development', 'classes']
dict_sec = ['security', 'fired', 'layoff', 'laid off', 'downsize', 'firing', 'layoffs']
dict_man = ['manager', 'leadership', 'CEO', 'executives', 'management']
dict_wl = ['work-life', 'work life', 'flexible', 'work from home', 'remote', 'worklife']

# Create list of lists.
terms = [dict_pay, dict_ben, dict_dyn, dict_op, dict_sec, dict_man, dict_wl]

# Create dummy indicators for presence of key words in reviews.
for i in terms:
    for s in i:
        df[s] = df['text_clean'].apply(lambda x: 1 if (" " + s + " ") in x else 0)

# Sum dummies into 7 categories of topics. 
# These are 0/1 indicators for presence of supervised topics in each review.
# To create indicators for number of times topics occur, use "sum" function rather than "max". 
df['dict_pay'] = df[dict_pay].max(axis = 1)
df['dict_ben'] = df[dict_ben].max(axis = 1)
df['dict_dyn'] = df[dict_dyn].max(axis = 1)
df['dict_op'] = df[dict_op].max(axis = 1)
df['dict_sec'] = df[dict_sec].max(axis = 1)
df['dict_man'] = df[dict_man].max(axis = 1)
df['dict_wl'] = df[dict_wl].max(axis = 1)

# Clean up and drop dummies created for individual dictionary terms.
for i in terms:
    for s in i:
        del df[s]

# Create sum variable to indicate number of supervised topics present. 
df['dict_sum'] = df[['dict_pay','dict_ben','dict_dyn','dict_op','dict_sec','dict_man','dict_wl']].sum(axis=1)



################################################################
#### Regression model: CountHelpful = f(review features) + epsilon.
################################################################

# Set data types. 
df['countHelpful'] = df['countHelpful'].astype(int)
#df['OverallRating'] = df['OverallRating'].astype(int) # For regression as integer.
df['OverallRating'] = df['OverallRating'].astype('category') # For regression as FE factor. 
df['lengthOfEmployment'] = df['lengthOfEmployment'].astype(int)
df['isCurrentJobFlag'] = df['isCurrentJobFlag'].astype(int)
df['read_ease_grade'] = df['read_ease_grade'].astype(float) 
#df['read_ease_pros'] = df['read_ease_pros'].astype(float) 
#df['read_ease_cons'] = df['read_ease_cons'].astype(float) 
df['word_count'] = df['word_count'].astype(int) 
df['word_count_squared'] = df['word_count_squared'].astype(int) 
#df['word_count_pros'] = df['word_count_pros'].astype(int) 
#df['word_count_pros_squared'] = df['word_count_pros_squared'].astype(int) 
#df['word_count_cons'] = df['word_count_cons'].astype(int) 
#df['word_count_cons_squared'] = df['word_count_cons_squared'].astype(int) 
df['sentence_count'] = df['sentence_count'].astype(int) 
#df['sentence_count_pros'] = df['sentence_count_pros'].astype(int) 
#df['sentence_count_cons'] = df['sentence_count_cons'].astype(int) 
df['paragraph'] = df['paragraph'].astype(int)
#df['paragraph_pros'] = df['paragraph_pros'].astype(int)
#df['paragraph_cons'] = df['paragraph_cons'].astype(int)
df['dict_pay'] = df['dict_pay'].astype(int) 
df['dict_ben'] = df['dict_ben'].astype(int) 
df['dict_dyn'] = df['dict_dyn'].astype(int) 
df['dict_op'] = df['dict_op'].astype(int) 
df['dict_sec'] = df['dict_sec'].astype(int) 
df['dict_man'] = df['dict_man'].astype(int) 
df['dict_wl'] = df['dict_wl'].astype(int) 
df['dict_sum'] = df['dict_sum'].astype(int)
df['topic'] = df['topic'].astype('category')
df['sector'] = df['sector'].astype('category')
df['GOC'] = df['GOC'].astype('category')
df['stateID'] = df['stateID'].astype('category')
df['employeesTotalNum'] = df['employeesTotalNum'].astype(int)
df['pvTotal'] = df['pvTotal'].astype(int)
df['text_ratio'].astype(float)

# Generate target variable (countHelpful).
y = df['countHelpful']
y = pd.DataFrame(y)
y.columns = ['countHelpful']

# Generate x matrix of feature regressors. 
constant = np.ones(df.shape[0]) # Column of ones for constant term in regression. 
constant = pd.DataFrame(constant)
constant.columns = ['constant']

x = pd.concat([df['OverallRating'],
               df['lengthOfEmployment'], 
               df['isCurrentJobFlag'],
               df['read_ease_grade'],
               #df['read_ease_pros'],
               #df['read_ease_cons'],
               df['word_count'],
               df['word_count_squared'],
               #df['word_count_pros'],
               #df['word_count_pros_squared'],
               #df['word_count_cons'],
               #df['word_count_cons_squared'],
               df['sentence_count'],
               #df['sentence_count_pros'],
               #df['sentence_count_cons'],
               df['paragraph'],
               #df['paragraph_pros'],
               #df['paragraph_cons'],
               df['dict_pay'],
               df['dict_ben'],
               df['dict_dyn'],
               df['dict_op'],
               df['dict_sec'],
               df['dict_man'],
               df['dict_wl'],
               df['dict_sum'],
               df['topic'],
               df['sector'],
               #df['GOC'],
               df['stateID'],
               df['employeesTotalNum'],
               df['pvTotal'],
               df['text_ratio'],
               constant],
               axis=1)

# Create training and test sets. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

# Create single training and test data frames. 
train = pd.concat([y_train,x_train], axis=1)
test = pd.concat([y_test,x_test], axis=1)

# Regression model formula. 
#form = "countHelpful ~ OverallRating + lengthOfEmployment + isCurrentJobFlag + read_ease_grade + word_count + word_count_squared + sentence_count + paragraph + dict_pay + dict_ben + dict_dyn + dict_op + dict_sec + dict_man + dict_wl + dict_sum + employeesTotalNum * pvTotal + text_ratio + C(sector) + C(topic) + C(stateID)" # Overall rating as an ordinal integer.
form = "countHelpful ~ C(OverallRating) + lengthOfEmployment + isCurrentJobFlag + read_ease_grade + word_count + word_count_squared + sentence_count + paragraph + dict_pay + dict_ben + dict_dyn + dict_op + dict_sec + dict_man + dict_wl + dict_sum + employeesTotalNum * pvTotal + text_ratio + C(sector) + C(topic) + C(stateID)" # Overall rating as a categorical FE.


##################################
# Estimate OLS model. ############
##################################
model_ols = smf.ols(form,data = train)
results_ols = model_ols.fit() # Non-robust standard errors. 
#summary_ols = results_ols.summary().as_csv().split("\n")
print(results_ols.summary(yname="countHelpful", alpha = 0.10))

# Create in- and out-of-sample predictions.
fitted_train = pd.DataFrame(results_ols.predict(x_train), columns=['fitted'])
fitted_test = pd.DataFrame(results_ols.predict(x_test), columns=['fitted'])

# Merge fitted values into training and test sets. 
train = pd.merge(train,fitted_train, how='left', left_index=True, right_index=True)
test = pd.merge(test,fitted_test, how='left', left_index=True, right_index=True)

# Calculated model residuals.
train['e'] = train['fitted'] - train['countHelpful'] 
test['e'] = test['fitted'] - test['countHelpful'] 

# Calculate MSE.
train['se'] = train['e']**2
test['se'] = test['e']**2
mse_train = np.mean(train['se'])
mse_test = np.mean(test['se'])
print(mse_train, mse_test)

# Calculate MAE. (mean absolute error)
train['ae'] = np.abs(train['e'])
test['ae'] = np.abs(test['e'])
mae_train = np.mean(train['ae'])
mae_test = np.mean(test['ae'])
print(mae_train, mae_test)

# Calculate Median AE. (median absolute error)
medae_train = np.median(train['ae'])
medae_test = np.median(test['ae'])
print(medae_train, medae_test)

# Sort test dataframe by predicted score. 
sort = test.sort_values('fitted', ascending=False, na_position='last')

# View top and bottom scored reviews. 
sort[:10]
sort[-10:]
df.loc[sort.index[0]]
#df.loc[sort.index[1]]
#df.loc[sort.index[2]]
