'''
Engineering Technical Challenge @ Briink
Tree-RAG for Entity Hierarchy

This code applies the retrieval augmentation technique presented in T-RAG: LESSONS FROM
THE LLM TRENCHES, Fatehkia et al. (2024) (https://arxiv.org/pdf/2402.07483.pdf).
'''

# Dependencies
import torch
import torch.nn.functional as F
import pandas as pd
from anytree import Node, RenderTree
from thefuzz import fuzz
from thefuzz import process
import itertools
import warnings

from transformers.utils import logging
from transformers import (
    AutoTokenizer,
    RagRetriever,
    RagTokenForGeneration,
    RagSequenceForGeneration,
)

# Custom Class implementing tree search + entity retrieval
from entitytree import EntityTree

# Code calling the following libraries has been commented out
import spacy
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

# remove warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Building our ESG Knowledge Base

# 1. Loading ESG data

# ESGData contains worldwide ESG data per country. This dataset is currently not leveraged in oour code but could be integrated as well in the Knowledge Base
file1_path = "/Users/clementgillet/Desktop/Technical Presentation @ Briink/ESG_CSV/ESGData.csv"
ESGData = pd.read_csv(file1_path)
ESGData.head(10)

# ESGData contains worldwide ESG data per country. We leverage this dataset for our TREE-RAG implementation
file2_path = "/Users/clementgillet/Desktop/Technical Presentation @ Briink/ESG_CSV/ESGSeries.csv"
ESGSeries = pd.read_csv(file2_path)
ESGSeries.head(10)

# 2. Building the Tree (Sectors > Topics > Subtopics)
# Could not find a logical order, so tree had to be extracted with simple, logical linguistic operations

# Remove NaN values in the topic column and adapt initial df 
ESGSeries.dropna(subset=["Topic"], inplace=True, ignore_index=True)

# Branch on 3 sectors : Environment, Social and Governance
environment = [topic for topic in ESGSeries.Topic if topic.startswith("Environment")]
social = [topic for topic in ESGSeries.Topic if topic.startswith("Social")]
governance = [topic for topic in ESGSeries.Topic if topic.startswith("Governance")]

# Convert to sets to remove duplicates
envTopics = set(environment)
socialTopics = set(social)
govTopics = set(governance)

# Remove sector from string
envTopics = [topic.replace("Environment: ", "") for topic in envTopics]
socialTopics = [topic.replace("Social: ", "") for topic in socialTopics]
govTopics = [topic.replace("Governance: ", "") for topic in govTopics]
allTopics = [envTopics, socialTopics, govTopics]

# Branch on topics (extract subtopics) and create one concatenated string with all subtopics
allSubtopics=[]
for topic in allTopics:
    ls2 = []
    for subtopic in topic:
        ls3 = []
        for idx in range(len(ESGSeries.Topic)):
            if subtopic in ESGSeries.Topic[idx]:
                ls3.append(ESGSeries["Indicator Name"][idx])
        ls2.append(ls3)
    allSubtopics.append(ls2)

# Let's start building a tree with anytree library
root = Node("ESG Data")
Environment = Node("Environment", parent=root)
Social = Node("Social", parent=root)
Governance = Node("Governance", parent=root)

sectors = ["Environment", "Social", "Governance"]
for i, sector in enumerate(allTopics):
    for j, topic in enumerate(sector):
        locals()["_".join(topic.split(" "))] = Node(topic, parent=locals()[sectors[i]])
        for subtopic in allSubtopics[i][j]:
            Node(subtopic, parent=locals()["_".join(sector[j].split(" "))])

# Render final Entity Hierarchy Tree
for pre, fill, node in RenderTree(root):
    print("%s%s" % (pre, node.name))

# 3. Building an indexed Dataset in order to build a retriever
# The following code (commented out) results in 2 files: index.faiss + HuggingFace dataset object saved on disk
'''
loader1 = CSVLoader(file1_path)
loader2 = CSVLoader(file2_path)
data = loader1.load()
metadata = loader2.load()

# Create an instance of the RecursiveCharacterTextSplitter class with specific parameters.
# It splits text into chunks of 1000 characters each with a 150-character overlap.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# 'metadata' holds the text you want to split, split the text into documents using the text splitter.
docs = text_splitter.split_documents(metadata)
titles = ["_" for doc in docs]
texts = [doc.page_content for doc in docs]
dataDict = {"title" : titles, "text" : texts}
ds = Dataset.from_dict(dataDict)

#convert to datasets.Datasets object with "title", "text" and "embeddings"
torch.set_grad_enabled(False)
ctx_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

ds_with_embeddings = ds.map(lambda example: {'embeddings': ctx_encoder(**ctx_tokenizer(example["text"], return_tensors="pt"))[0][0].numpy()})

ds_with_embeddings.save_to_disk('/Users/clementgillet/Desktop/ESGdataset')

#FAISS will allow similarity search at the retriever stage
ds_with_embeddings.add_faiss_index(column='embeddings')
ds_with_embeddings.save_faiss_index('embeddings', '/Users/clementgillet/Desktop/my_index.faiss')
dataset = ds_with_embeddings
'''

# Load previously generated dataset
dataset_path = '/Users/clementgillet/Desktop/ESGdataset' # dataset saved via *dataset.save_to_disk('path/to/dataset')*
index_path = '/Users/clementgillet/Desktop/my_index.faiss'  # faiss index saved dataset.save_faiss_index('embeddings', 'path/to/index')

# Construct a RagRetriever with generator and question_encoder
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path=dataset_path,
    index_path=index_path,
)

# BASELINE RAG

# set up tokenizer and model from pretrained
tokenizer = AutoTokenizer.from_pretrained("facebook/rag-sequence-nq")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

# What follows is the user input being tokenized
query = "What is ESG ?"
inputs = tokenizer(query, return_tensors="pt")
input_ids = inputs["input_ids"]

# Encode
question_hidden_states = model.question_encoder(input_ids)[0]

# Retrieve based on similarity search in our indexed db
docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")

# Calculate the score (%) of retrieved documents
doc_scores = torch.bmm(
    question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)
).squeeze(1)

# Forward retrieved documents to generator
RAGgenerated = model.generate(
    context_input_ids=docs_dict["context_input_ids"],
    context_attention_mask=docs_dict["context_attention_mask"],
    doc_scores=doc_scores,
    num_beams = 3,
)

RAGgenerated_string = tokenizer.batch_decode(RAGgenerated, skip_special_tokens=True)

print("RAG","\nQuestion : ",query, "\nGenerated Answer : ", RAGgenerated_string[0])

# T-RAG Implementation

# Declare instance of our custom class.
entity_tree = EntityTree(root)

# Spacy NER cannot be implemented for the moment, as we would need to train a Spacy NER on the kind of data we're manipulating
'''
nlp = spacy.load("en_core_web_lg")
doc = nlp(query)
mentioned_entities = doc.ents
'''

# Custom way before training a spacy model to detect. It's based in fuzzy matching
# List up all elements of the tree
#itertools is used here to flatten our arrays
allTopics = list(itertools.chain.from_iterable(allTopics))
allSubtopics = list(itertools.chain.from_iterable(allSubtopics))
allSubtopics = list(itertools.chain.from_iterable(allSubtopics))
list_of_entities = sectors + allTopics + allSubtopics

#Find fuzzy matches of the query with entities in our hierarchy tree. Filtering is possible according to a matching treshold
mentioned_entities = [entity[0] for entity in process.extract(query, list_of_entities, scorer=fuzz.ratio) if entity[1] > 40]

# Only if entities are mentioned in the query, will T-RAG operate. Otherwise, a simple RAG takes place...

if len(mentioned_entities) != 0:
    entities_info = entity_tree.retrieve(mentioned_entities)

    input_ids_padded = torch.stack([F.pad(row, (0, 300-len(row))) for row in entities_info["input_ids"]])
    attention_mask_padded = torch.stack([F.pad(row, (0, 300-len(row))) for row in entities_info["attention_mask"]])

    doc_scores = torch.cat((torch.full((len(mentioned_entities),), 100).unsqueeze(0), doc_scores), 1)
    model.config.n_docs = 5 + len(mentioned_entities)
    TRAGgenerated = model.generate(
        context_input_ids = torch.cat((input_ids_padded, docs_dict["context_input_ids"]), 0), # + entity_input_ids
        context_attention_mask = torch.cat((attention_mask_padded, docs_dict["context_attention_mask"]), 0), # + entity_attention_mask
        doc_scores = doc_scores,
        num_beams = 3
    )
else:
    TRAGgenerated = model.generate(
        context_input_ids=docs_dict["context_input_ids"],  # + entity_input_ids
        context_attention_mask=docs_dict["context_attention_mask"],  # + entity_attention_mask
        doc_scores=doc_scores,
        num_beams= 3
    )

TRAGgenerated_string = tokenizer.batch_decode(TRAGgenerated, skip_special_tokens=True)
print("-------------------------------------------------------------------------------------------")
print("T-RAG","\nQuestion : ", query, "\nGenerated Answer : ", TRAGgenerated_string[0])
print("\nThis text was generated with the following entity hierarchy information : \n ", )
if len(mentioned_entities) != 0:
    for elem in entities_info["entity_info"]:
        print(elem)


