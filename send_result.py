# import libraries
import pickle
import pandas as pd
import numpy as np
import hazm
import random
import torch
from tqdm import tqdm
import string
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoConfig, AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel, pipeline, set_seed

# function for preproccessing data
def preprocess_sentences(sentences):
    preprocessed = []
    normalizer = hazm.Normalizer()
    tokenizer = hazm.word_tokenize
    stopwords = hazm.stopwords_list()
    puncs = '.،٪٫!؟'
    for text in tqdm(sentences):
        text = normalizer.normalize(text)
        words = tokenizer(text)
        words = [word for word in words if word not in stopwords and word not in puncs]
        translator = str.maketrans('', '', string.punctuation)
        words = [word.translate(translator) for word in words]
        processed_text = ' '.join(words)
        preprocessed.append(processed_text)
    return preprocessed

# Find similarity Bert
def find_similarity_bert(question):
    user_tokenized_question = bert_tokenizer(question, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = bert_model(**user_tokenized_question)
    user_vectorized_question = model_output.last_hidden_state.mean(dim=1)
    similarities = []
    for question in tqdm(vectorized_bert):
        cosine_similarities = cosine_similarity(user_vectorized_question, question)
        similarities.append(cosine_similarities[0][0])
    similarities = np.array(similarities)

    return similarities

def find_similarity_tfidf(question):
    query_vector = vectorizer_tfidf.transform([question])
    cosine_similarities = np.dot(query_vector, vectorized_tfidf.T).toarray()[0]

    return cosine_similarities

# generate answer gpt-2
def generate_answer_gpt2(question):
    # prompt
    indx = random.randint(0, len(org_questions))
    prompt = f"به سوالات زیر مانند نمونه پاسخ بده:\n سوال: {org_questions[indx+1]}\n جواب: {org_answers[indx+1]}\n سوال: {org_questions[indx+2]}\n جواب: {org_answers[indx+2]}\n سوال : {question}\n جواب: "
    prompt = question
    generated = generator(prompt, max_length=350, num_return_sequences=1, temperature=0.7)
    answer = generated[0]['generated_text'].replace(prompt, '')
    try:
        start_indx = answer.index('سلام عزيزم')
        end_indx = answer[start_indx:].index('.')
        print(start_indx)
        answer = answer[start_indx:].split("\n")[0]
        # answer = answer.split(' ')
        # answer.reverse()
        return answer
    except:
        return answer
    # answer = answer.split(' ')
    # answer.reverse()
    # return ' '.join(answer)

# Read vectorized data Bert
with open('vectorized_bert.pkl', 'rb') as f:
    vectorized_bert = pickle.load(f)

# Read vectorized data tfidf
with open('vectorized_tfidf.pkl', 'rb') as f:
    vectorized_tfidf = pickle.load(f)

# Read vectorizer tfidf
with open('vectorizer.pkl', 'rb') as f:
    vectorizer_tfidf = pickle.load(f)

# Load Bert model
bert_config = AutoConfig.from_pretrained("HooshvareLab/bert-fa-zwnj-base")
bert_tokenizer = AutoTokenizer.from_pretrained("./bert/tokenizer")
bert_model = AutoModel.from_pretrained("./bert/bert_model")

# Load gpt-2 model
model_dir = "gpt2_5epoch"
generator = pipeline('text-generation', model=model_dir)
gpt_tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
gpt_model = GPT2LMHeadModel.from_pretrained(model_dir)
set_seed(42)

# Read user question
data = pd.read_csv('./final_dataset.csv')
org_questions, org_answers = data['question'], data['answer']

print("models loaded successfully...")

def main(input_question, model):
    preprocessed_input_question = preprocess_sentences([input_question])[0]

    if model == "Bert":
        similarities = find_similarity_bert(preprocessed_input_question)
        sorted_indices = np.argsort(similarities)[::-1]
        # print("org_question:")
        # print(org_questions[sorted_indices[0]])

        # print("model answer:")
        # print(org_answers[sorted_indices[0]])

        # return org_answers[sorted_indices[0]]
    elif model == 'tf-idf':
        similarities = find_similarity_tfidf(preprocessed_input_question)
        sorted_indices = np.argsort(similarities)[::-1]
        # print("org_question:")
        # print(org_questions[sorted_indices[0]])

        # print("model answer:")
        # print(org_answers[sorted_indices[0]])\
    if model == "gpt-2":
        answer = generate_answer_gpt2(input_question)
        return answer
        # # tokenizer
        # input_ids = gpt_tokenizer.encode(input_question, return_tensors="pt")
        # output = gpt_model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2)
        # response = gpt_tokenizer.decode(output[0], skip_special_tokens=True)
        # return response

    return org_answers[sorted_indices[0]]

# main("علایم سرطان", "gpt-2")