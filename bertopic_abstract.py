import importlib
import subprocess
try:
    # Check if the module is already installed
    importlib.import_module('torch')
    print("torch is already installed.")
except ImportError:
    # If the module is not installed, try installing it
    subprocess.run(['pip3', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])
    print("torch was installed correctly.")

import torch
print("Torch version:",torch.__version__)
print("Is CUDA enabled?",torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.randn(1).cuda())
else:
    #get torch here: https://pytorch.org/get-started/locally/
    subprocess.run(['pip3', 'uninstall', 'torch'])
    subprocess.run(['pip3', 'install', 'torch', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu118'])


    # Import packages
import os
import io
import sys
import ast
import re
import random
import zipfile
import requests
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# Plotly: Set notebook mode to work offline
import plotly.offline as pyo
import plotly.graph_objs as go
pyo.init_notebook_mode()

# Set data directory
save_path = 'data/'

# Set working directory
if 'notebooks' in os.getcwd():
    os.chdir(os.path.dirname(os.getcwd()))


# Define functions

def simple_bool(message):
    choose = input(message+" (y/n): ").lower()
    your_bool = choose in ["y", "yes","yea","sure"]
    return your_bool

def get_and_extract(file, dir = os.getcwd(), ext = '.zip'):
    url='https://zenodo.org/record/8205724/files/'+file+'.zip?download=1'
    zip_file_name = file+ext
    extracted_folder_name = dir
    # Download the ZIP file
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the ZIP contents
        with io.BytesIO(response.content) as zip_buffer:
            with zipfile.ZipFile(zip_buffer, 'r') as zip_ref:
                zip_ref.extractall(extracted_folder_name)
        print(f"ZIP file '{zip_file_name}' extracted to '{extracted_folder_name}' successfully.")
    else:
        print("Failed to download the ZIP file.")

def get_gitfile(url, flag='', dir = os.getcwd()):
    url = url.replace('blob','raw')
    response = requests.get(url)
    file_name = flag + url.rsplit('/',1)[1]
    file_path = os.path.join(dir, file_name)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded successfully. Saved as {file_name}")
    else:
        print("Unable to download the file.")

def load_preprocessed(doc_name = 'abs_preprocessed.txt'):
    with open(save_path+doc_name, 'r',encoding='utf-8') as file:
        docs_processed = []
        for line in file:
            docs_processed.append(str(line.strip()))

    print("Imported list:", doc_name)
    return docs_processed

#get_gitfile("https://raw.githubusercontent.com/johndef64/pyutilities_datascience/main/general_utilities.py")

def lower(text):
    # Split the text into words
    words = text.split()
    # Lowercase the words that don't have consecutive uppercase letters
    processed_words = [word.lower() if not re.search('[A-Z]{2,}', word) else word for word in words]
    # Join the processed words back into text
    processed_text = ' '.join(processed_words)
    return processed_text


docs =[]
#### Load Abstracts
def load_abstracts_from_csv(doc_name = 'scopus.csv', abs_col= 'Abstract'):
    global docs
    df = pd.read_csv(doc_name, index_col=0)
    docs = df.Abstract.drop_duplicates().to_list()
    print('\nEntry count:',len(df),
          '\nabstract count:', df[abs_col].nunique(),
          '\nEntry without abstract:',len(df)-df.Abstract.nunique())

    # lower text keeping acronyms uppercase

    docs_to_process = docs#[100:1600]#random.sample(docs, 100)

    # Normalize docs
    timea = time.time()
    sampled_docs = docs_to_process
    docs_str = str(sampled_docs)
    docs_lower = lower(docs_str)
    docs_processed = ast.literal_eval(docs_lower)
    print('\nNormalization runtime:',time.time()-timea)

    #print(docs_to_process[1],'\n')
    #print(docs_processed[1])
    return docs_processed


#%%
######### TOPIC MODELING #########

import re
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import MaximalMarginalRelevance
from bertopic.representation import KeyBERTInspired

sentence_transformer = SentenceTransformer("all-mpnet-base-v2")

def setup_model(base_embedder ="allenai-specter",
                n_neighbors  = 15,
                n_components = 5,
                random_state = 1337,
                min_cluster_size = 5):

    global sentence_transformer

    # Step 1 Extract embeddings (SBERT)
    models = ["allenai-specter", # SPECTER is a model trained on scientific citations and can be used to estimate the similarity of two publications. We can use it to find similar papers.
              "all-mpnet-base-v2",  # designed as general purpose model, The all-mpnet-base-v2 model provides the best quality,
              "all-MiniLM-L6-v2" # while all-MiniLM-L6-v2 is 5 times faster and still offers good quality.
              ] # https://www.sbert.net/docs/pretrained_models.html
    #base_embedder = models[0]  # BaseEmbedder
    sentence_transformer = SentenceTransformer(base_embedder) # SentenceTransformer


    # Step 2 - Reduce dimensionality
    # uniform manifold approximation and projection (UMAP) to reduce the dimension of embeddings
    #random_state = 1337 #1000 #1337
    umap_model = UMAP(n_neighbors  = n_neighbors, #num of high dimensional neighbours
                      n_components = n_components, # default:5 #30
                      min_dist     = 0.0,
                      random_state = random_state) # default:None
    # https://stackoverflow.com/questions/71320201/how-to-fix-random-seed-for-bertopic


    # Step 3 - Cluster reduced embeddings
    # HDBSCAN (hierarchical density-based spatial clustering of applications with Noise)  to generate semantically similar document clusters.
    # Since HDBSCAN is a density-based clustering algorithm, the number of clusters is automatically chosen based on the minimum distance to be considered as a neighbor.
    #min_cluster_size = 5 #5 default HDBSCAN()
    hdbscan_model = HDBSCAN(min_cluster_size = min_cluster_size,
                            metric='euclidean',
                            cluster_selection_method='eom',
                            prediction_data=True)

    # Step 4 - Tokenize topics
    vectorizer_model = CountVectorizer(stop_words="english", lowercase=False) # lowercase=False to keep genes uppercase


    # Step 5 - Create topic representation
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True) # False default


    # Step 6 - Fine-tune topic representations with a bertopic.representation model
    # Create your representation model
    representation_model = MaximalMarginalRelevance(diversity = 0.7,  # 0.1 default
                                                    top_n_words = 15) # 10 default
    #representation_model = KeyBERTInspired()


    # Use the representation model in BERTopic on top of the default pipeline

    # All steps together
    topic_model = BERTopic(
        min_topic_size = 10,                         # 10 default
        top_n_words = 15,                            # 10 default
        calculate_probabilities = True,
        embedding_model = sentence_transformer,      # Step 1 - Extract embeddings
        umap_model = umap_model,                     # Step 2 - Reduce dimensionality
        hdbscan_model = hdbscan_model,               # Step 3 - Cluster reduced embeddings
        vectorizer_model = vectorizer_model,         # Step 4 - Tokenize topics
        ctfidf_model = ctfidf_model,                 # Step 5 - Extract topic words
        representation_model= representation_model  # Step 6 - (Optional) Fine-tune topic represenations
    )
    print('Topic Model ready\nUMAP random state:',random_state,'\nBase embedder:',base_embedder)
    return topic_model

#### TRAIN MODEL #####
## If you want to split embedding phase, use it as follows:

def train_model(topic_model, docs_processed, embedding_file = ''):
    # Step 1 Embedding documents with sentence_transformer

    if embedding_file != '':
        embeddings = np.loadtxt(embedding_file)
    else:
        embeddings = sentence_transformer.encode(docs_processed,
                                                 show_progress_bar=True)

    # Train with custom embeddings
    topics, probs = topic_model.fit_transform(docs_processed, embeddings=embeddings)
    #topics = topic_model.fit(docs_processed, embeddings=embeddings)

    return topics, probs, embeddings


def get_topic_info(topic_model):
    df = topic_model.get_topic_info()
    return df

def get_topics(topic_model):
    all_topics = topic_model.get_topics()
    topic_df = pd.DataFrame(all_topics)
    return topic_df

# Probablities
#probs_df = pd.DataFrame(probs)

def get_topic_freq(topic_model):
    topic_freq = topic_model.get_topic_freq()
    print('total',topic_freq.Count.sum())
    return topic_freq

def get_document_info(topic_model, docs_processed):
    doc_info = topic_model.get_document_info(docs_processed)
    print(doc_info.columns)
    return doc_info

'''def visualize_documents(docs_processed, sample=1, embeddings= '', custom_labels=False):
    if embeddings != '':
        map = topic_model.visualize_documents(docs_processed, embeddings=embeddings,sample= sample, custom_labels=custom_labels)
    else:
        map = topic_model.visualize_documents(docs_processed, sample= sample, custom_labels=custom_labels)
    return map'''

import plotly.graph_objects as go

def visualize_documents(topic_model, docs_processed, sample=1, embeddings='', custom_labels=False):
    if embeddings != '':
        map = topic_model.visualize_documents(docs_processed, embeddings=embeddings, sample=sample, custom_labels=custom_labels)
    else:
        map = topic_model.visualize_documents(docs_processed, sample=sample, custom_labels=custom_labels)

    # Create a figure with the provided map
    fig = go.Figure(data=map)

    # Show the figure
    fig.show()

def visualize_distribution(topic_model, probs):
    fig = go.Figure()
    for n in range(3):
        fig = topic_model.visualize_distribution(probs[n],
                                                 min_probability = 0,
                                                 custom_labels = False)
        fig.show()

def visualize_similarty(topic_model, topics, top_n_topics=5):
    fig = go.Figure()
    fig = topic_model.visualize_heatmap(topics=topics,
                                        top_n_topics = top_n_topics,
                                        #n_clusters = 20,
                                        custom_labels=True,
                                        width = 1100,
                                        height = 1100)
    #fig.write_image("fig1.png", engine='kaleido')
    fig.show()

def visualize_hierarchy(topic_model, width=700, height=600):
    fig = go.Figure()
    fig = topic_model.visualize_hierarchy(width=width, height=height) #The topics that were created can be hierarchically reduced.
    fig.show()

def visualize_barchart(topic_model, topics, n_words=20):
    fig = go.Figure()
    topic_model.visualize_barchart(n_words = n_words,
                                   topics = topics,
                                   #top_n_topics=len(topic_info)//4,
                                   )
    fig.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def create_wordcloud(topic_model, topic):
    text = {word: value for word, value in topic_model.get_topic(topic)}
    print(text)
    wc = WordCloud(background_color="white", max_words=1000, width=800, height=400)
    wc.generate_from_frequencies(text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def create_wordcloud_multiple(topic_model, topics, output_path='wordcloud.png', dpi=300, save=True):
    merged_dict = {}
    for i in topics:
        text = {word: value for word, value in topic_model.get_topic(i)}
        merged_dict.update(text)
    #pc.copy(str(merged_dict))

    plt.figure(figsize=(12, 8))

    wc = WordCloud(background_color="white", max_words=1000, width=1000, height=500)
    wc.generate_from_frequencies(merged_dict)

    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if save:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()

def create_wordcloud_from_corpus(corpus, output_path='wordcloud.png', dpi=300, save=True):
    # Combine the text corpus into a single string
    text = ' '.join(corpus)
    # Generate WordCloud from the text
    wc = WordCloud(background_color="white", max_words=1000, width=800, height=400)
    wc.generate(text)
    # Display the WordCloud
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    if save:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.show()


help=r'''
# Basic Usage

import bertopic_abstract as bt

bert_abs = bt.load_abstracts_from_csv(r"C:\Users\Utente\Downloads\scopus_sem+bioint+omics.csv")
topic_model = bt.setup_model(base_embedder ="allenai-specter",
                n_neighbors  = 15,
                n_components = 5,
                random_state = 1337,
                min_cluster_size = 5)
topics, probs, embeddings = bt.train_model(topic_model, bert_abs)

bt.get_topic_info(topic_model)

bt.get_document_info(topic_model, bert_abs)

bt.visualize_documents(topic_model, bert_abs)

#bt.visualize_distribution(topic_model, probs)

bt.visualize_similarty(topic_model,topics, top_n_topics=1)

bt.visualize_hierarchy(topic_model, width=700, height=600)
'''