import re
from nltk.stem import PorterStemmer
# from nltk.tokenize import word_tokenize
from getVocabList import *

def processEmail(email_contents):
    """
    Preprocesses a the body of an email and returns a list of word indices 

    Preprocesses the body of an email and returns a list of indices of the words contained in the email. 
    """
    # Load Vocabulary
    vocabList = getVocabList()
    # === Preprocess Email ===
    # Lower case
    email_contents = email_contents.lower()
    # Strip all HTML
    # Looks for any expression that starts with < and ends with > 
    # and does not have any < or > in the tag and replace it with a space
    email_contents = re.sub('<[^<>]+>', ' ', email_contents)
    # Handle Numbers
    # Look for one or more characters between 0-9
    email_contents = re.sub(r'[\d]+', 'number', email_contents)
    # Handle URLS
    # Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)
    # Handle Email Addresses
    # Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)
    # Handle $ sign
    email_contents = re.sub('[$]+', 'dollar', email_contents)
    # === Tokenize Email ===
    # Output the email to screen as well
    print('\n==== Processed Email ====\n')
    # Tokenize and also get rid of any punctuation and non alphanumeric characters
    email_contents = re.sub(r'[\W_]+', ' ', email_contents)
    # Stem the word
    ps = PorterStemmer()
    email_contents = [ps.stem(token) for token in email_contents.split(' ')]
    # Look up the word in the dictionary and add to word_indices if found
    word_indices = [vocabList[word] for word in email_contents if word in vocabList]
    # ======
    # Print to screen
    print(' '.join(email_contents))
    # Print footer
    print('\n=========================\n')
    return word_indices
    
    