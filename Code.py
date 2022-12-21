

# ##  Analyze a document
# 
# When you have a long document, you would like to 
# - Quanitfy how `concrete` a sentence is
# - Create a concise summary while preserving it's key information content and overall meaning. Let's implement an `extractive method` based on the concept of TF-IDF. The idea is to identify the key sentences from an article and use them as a summary. 
# 
# 
# 

# ### Step 1: Preprocess the input document 
# 
# Defined a function `proprocess(doc, lemmatized = True, remove_stopword = True, lower_case = True, remove_punctuation = True, pos_tag = False)` 
# - Four input parameters:
#     - `doc`: an input string (e.g. a document)
#     - `lemmatized`: an optional boolean parameter to indicate if tokens are lemmatized. The default value is True (i.e. tokens are lemmatized).
#     - `remove_stopword`: an optional boolean parameter to remove stop words. The default value is True, i.e., remove stop words. 
#     - `remove_punctuation`: optional boolean parameter to remove punctuations. The default values is True, i.e., remove all punctuations.
#     - `lower_case`: optional boolean parameter to convert all tokens to lower case. The default option is True, i.e., lowercase all tokens.
#     - `pos_tag`: optional boolean parameter to add a POS tag for each token. The default option is False, i.e., no POS tagging.  
#     
#        
# - Split the input `doc` into sentences.
# 
# 
# - Tokenize each sentence into unigram tokens and also process the tokens as follows:
#     - If `lemmatized` is True, lemmatize all unigrams. 
#     - If `remove_stopword` is set to True, remove all stop words. 
#     - If `remove_punctuation` is set to True, remove all punctuations. 
#     - If `lower_case` is set to True, convert all tokens to lower case 
#     - If `pos_tag` is set to True, find the POS tag for each token and form a tuple for each token, e.g., ('recently', 'ADV'). Either Penn tags or Universal tags are fine. See mapping of these two tagging systems here: https://universaldependencies.org/tagset-conversion/en-penn-uposf.html
# 
# 
# - Return the original sentence list (`sents`) and also the tokenized (or tagged) sentence list (`tokenized_sents`). 
# 
#    
# 

# In[116]:


import string
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
    
def preprocess(doc, lemmatized=True, pos_tag = False, remove_stopword=True, lower_case = True, remove_punctuation = True):
    
    sents, tokenized_sents = None, None
    final2 = []
    doc2 = doc.replace('\n\n','. ')
    sents = nltk.sent_tokenize(doc2)
    #sents = nltk.regexp_tokenize(doc,'\w+[\s\,\w\-]+\w')
    #sents = re.split('\n\n+|\.',doc)
    word_lemmatizer = WordNetLemmatizer()
    for each_sent in sents:
        loop_sent = []
        tokenized_sents = []
        another2 = []
        tokens = nltk.word_tokenize(each_sent)
                
        #tokens = re.split('\s|\,|\-',each_sent)
        if pos_tag == True:
            tagged_tokens = nltk.pos_tag(tokens)
            tokens = tagged_tokens
        if remove_stopword == True:
            stop_words = stopwords.words('english')
            stop_words += ['The','still','Until','But','two'] 
            for token in tokens:
                if token not in stop_words:
                    tokenized_sents.append(token)
            tokens = tokenized_sents
            
        if remove_punctuation == True:
            tokens = [token.strip(string.punctuation) for token in tokens]
            tokens= [token.strip() for token in tokens if token.strip()!='']
        if lower_case == True:
            for token in tokens:
                another2.append(token.lower())
            tokens = another2
        if lemmatized == True:
            for each_token in tokens:
                loop_sent.append(word_lemmatizer.lemmatize(each_token))
            tokens = loop_sent
        
        final2.append(tokens)
            
    return sents, final2;


# In[117]:


# load test document

text = open("power_of_nlp.txt", "r", encoding='utf-8').read()


# In[118]:


# test with all default options:

sents, tokenized_sents = preprocess(text)
sents[28]
# print first 3 sentences
for i in range(3):
    print(sents[i], "\n",tokenized_sents[i],"\n\n" )


# In[119]:


# process text without remove stopwords, punctuation, lowercase, but with pos tagging

sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)

for i in range(3):
    print(sents[i], "\n",tokenized_sents[i],"\n\n" )


# ### Step 2: Quantify sentence concreteness
# 
# 
# `Concreteness` can increase a message's persuasion. The concreteness can be measured by the use of :
# - `article` (e.g., a, an, and the), 
# - `adpositions` (e.g., in, at, of, on, etc), and
# - `quantifiers`, i.e., adjectives before nouns.
# 
# 
# Defined a function `compute_concreteness(tagged_sent)` as follows:
# - Input argument is `tagged_sent`, a list with (token, pos_tag) tuples as shown above.
# - Find the three types of tokens: `articles`, `adposition`, and `quantifiers`.
# - Compute `concereness` score as:  `(the sum of the counts of the three types of tokens)/(total non-punctuation tokens)`.
# - return the concreteness score, articles, adposition, and quantifiers lists.
# 
# 
# Finds the most concrete and the least concrete sentences from the article. 
# 
# 


# In[120]:


def compute_concreteness(tagged_sent):
    
    concreteness, articles, adpositions,quantifier = None, None, None, None
    count = 0
    articles = ['a','an','the']
    articles_inside = []
    adpositions_inside=[]
    tags=[]
    counter = 0
    total_non_punctuation = 0
    punctuation = string.punctuation
    for x in tagged_sent:
        if x[0] in articles:
            count+=1
            if x[0] not in articles_inside:
                articles_inside.append((x[0],x[1]))
        if x[1] == 'IN':
            count += 1
            if x[0] not in adpositions_inside:
                adpositions_inside.append((x[0],x[1]))
        if x[0] not in punctuation:
            total_non_punctuation += 1
    while counter < len(tagged_sent) and counter+1 < len(tagged_sent):
        if tagged_sent[counter][1] == 'JJ' and tagged_sent[counter+1][1] in ['NN','NNS']:
            tags.append((tagged_sent[counter][0],tagged_sent[counter][1]))
            counter+=1
            count+=1
        else:
            counter+=1
        
        
        
    concreteness = count/total_non_punctuation
    
            
    
    return concreteness, articles_inside, adpositions_inside,tags
    


# In[121]:


# tokenize with pos tag, without change the text much

sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = True, 
                                    remove_stopword=False, remove_punctuation = False, 
                                    lower_case = False)


# In[122]:


len(sents)


# In[123]:


sents[1]
tokenized_sents[1]


# In[124]:


# find concreteness score, articles, adpositions, and quantifiers in a sentence

idx = 1    # sentence id
x = tokenized_sents[idx]
concreteness, articles, adpositions, quantifiers = compute_concreteness(x)

# show sentence
sents[idx]
# show result
concreteness, articles, adpositions, quantifiers


# In[125]:


# Find the most concrete and the least concrete sentences from the article


sentence_id = 0
final_dic = {}
#sents_in_text=[]
#concreteness_all=[]
while sentence_id < len(tokenized_sents):
    concreteness, articles, adpositions, quantifiers = compute_concreteness(tokenized_sents[sentence_id])
    final_dic[sents[sentence_id]]=concreteness
    #concreteness_all.append(concreteness)
    #sents_in_text.append(sents[sentence_id])
    sentence_id += 1
sen_list=list(final_dic.keys())
concrete_score_list = list(final_dic.values())

max_concerete_sentence = max(final_dic.keys(),key=(lambda x: final_dic[x]))
max_loc= concrete_score_list.index(final_dic[max_concerete_sentence])

min_concerete_sentence = min(final_dic.keys(),key=(lambda x: final_dic[x]))
min_loc= concrete_score_list.index(final_dic[min_concerete_sentence])


print (f"The most concerete sentence:  {sen_list[max_loc]}, {final_dic[max_concerete_sentence]:.3f}\n")
print (f"The least concerete sentence:  {sen_list[min_loc]}, {final_dic[min_concerete_sentence]:.3f}")


# ### Step 3: Generate TF-IDF representations for sentences 
# 
# Defined a function `compute_tf_idf(sents, use_idf)` as follows: 
# 
# 
# - Take the following two inputs:
#     - `sents`: tokenized sentences (without pos tagging) returned from Q2.1. These sentences form a corpus for you to calculate `TF-IDF` vectors.
#     - `use_idf`: if this option is true, return smoothed normalized `TF_IDF` vectors for all sentences; otherwise, just return normalized `TF` vector for each sentence.
#     
#     
# # - Returns the `TF-IDF` vectors  if `use_idf` is True.  Return the `TF` vectors if `use_idf` is False.

# In[126]:




def compute_tf_idf(sents, use_idf = True, min_df = 1):
    
    tf_idf = None
    lower_token_sents=[]
    token_list=[]
    for sen in sents:
        for tken in sen:
            token_list.append(tken.lower())
        lower_token_sents.append(token_list)
        token_list=[]
            
    dic = {indx: nltk.FreqDist(token) for indx,token in enumerate(lower_token_sents)}
    dtm = pd.DataFrame.from_dict(dic, orient="index")
    dtm = dtm.fillna(0)
    dtm=dtm.sort_index(axis=0)
    tf = dtm.values
    doc_length = tf.sum(axis=1)
    tf = np.divide(tf,doc_length[:,None])
    if use_idf == False:
        return tf
    else:
        doc_freq = np.where(tf>0,1,0)
        idf =  np.log(np.divide(len(sents)+1, np.sum(doc_freq,axis=0)+1)+1)
        smoothed_tf_idf=normalize(tf*idf)
        return smoothed_tf_idf 


# In[127]:


# test compute_tf_idf function

sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)

# show shape of TF-IDF
tf_idf.shape


# ### Step 5: Identify key sentences as summary 
# 
# The basic idea is that, in a coherence article, all sentences should center around some key ideas. If we can identify a subset of sentences, denoted as $S_{key}$, which precisely capture the key ideas,  then $S_{key}$ can be used as a summary. Moreover, $S_{key}$ should have high similarity to all the other sentences on average, because all sentences are centered around the key ideas contained in $S_{key}$. Therefore, we can identify whether a sentence belongs to $S_{key}$ by its similarity to all the other sentences.
# 
# 
# Defined a function `get_summary(tf_idf, sents, topN = 5)`  as follows:
# 
# - This function takes three inputs:
#     - `tf_idf`: the TF-IDF vectors of all the sentences in a document
#     - `sents`: the original sentences corresponding to the TF-IDF vectors
#     - `topN`: the top N sentences in the generated summary
# 
# - Steps:
#     1. Calculate the cosine similarity for every pair of TF-IDF vectors 
#     1. For each sentence, calculate its average similarity to all the others 
#     1. Select the sentences with the `topN` largest average similarity 
#     1. Print the `topN` sentences index
#     1. Return these sentences as the summary

# In[128]:


def get_summary(tf_idf, sents, topN = 5):
    
    
    similarity=1-pairwise_distances(tf_idf, metric = 'cosine')
    similarity
    avg_similarity=[]
    for x in similarity:
        sum_of_elements = sum(x)
        avg_similarity.append(sum_of_elements/len(x))
    avg_simi_dic = {x:y for x,y in enumerate (avg_similarity)}
    sorted_avg_simi_dic={k:v for k,v in sorted(avg_simi_dic.items(), key=lambda item:item[1])[::-1]}
    
    indx_of_sen = [x for x in sorted_avg_simi_dic.keys()]
   
    num=range(topN)
    
    summary=[]
    for x in num:
        summary.append(sents[indx_of_sen[x]])
        
        
    
    #return similarity
    return summary 


# In[129]:


# put everything together and test with different options

sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = True)
summary  = get_summary(tf_idf, sents, topN = 5)


for sent in summary:
    print(sent,"\n")


# In[130]:


# Please test summary generated under different configurations
sents, tokenized_sents = preprocess(text)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = False)
summary  = get_summary(tf_idf, sents, topN = 5)


for sent in summary:
    print(sent,"\n")


# In[133]:


sents, tokenized_sents = preprocess(text, lemmatized = False, pos_tag = False, 
                                    remove_stopword=True, remove_punctuation = True, 
                                    lower_case = True)
tf_idf = compute_tf_idf(tokenized_sents, use_idf = False)
summary  = get_summary(tf_idf, sents, topN = 5)


for sent in summary:
    print(sent,"\n")


