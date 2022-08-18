import os
import pickle
import re
import string

import streamlit as st

import numpy as np
import pandas as pd

import catboost

import torch
from transformers import AutoTokenizer, AutoModel

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
from nltk import WordNetLemmatizer


def check_nltk():
	for path in nltk.data.path:
		try:
			if len(os.listdir(path)) > 0:
				return
		except:
			continue

	nltk.download("omw-1.4")
	nltk.download("wordnet")


with st.spinner('🌀 Загружаю данные...'):
	check_nltk()
	data = pd.read_csv('data/data.csv')
	embeddings = pd.read_csv('data/features_emb.csv')
	preds = pd.read_csv('data/catboost_preds.csv')
	catboost_bert_model = catboost.CatBoostClassifier(random_state=25).load_model('src/model/catboost.cbm')
	catboost_tf_idf_model = catboost.CatBoostClassifier(random_state=25).load_model('src/model/tf_idf_catboost.cbm')
	bert_tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
	bert_model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased-conversational")
	tf_idf_vocab = pd.read_csv('data/tf_idf_vocab.csv', index_col='Unnamed: 0')


def get_random_message() -> str:
	return data.sample(1)['description'].values[0]


def get_bert_prediction(
	text: str
) -> str:

	res_mapper = {
		0: 'Контактная информация отсутствует',
		1: 'Есть контактная информация'
	}

	tokens = bert_tokenizer.encode(
		text,
		add_special_tokens=True,
		truncation=True,
		max_length=512
	)

	n = 512  # max длина вектора

	padded = torch.LongTensor(
		[
			np.array(tokens + [0] * (n - len(tokens)))
		]
	)

	attention_mask = torch.LongTensor(
		np.where(
			padded != 0, 1, 0
		)
	)

	with torch.no_grad():
		batch_embeddings = bert_model(padded, attention_mask=attention_mask)[0][:, 0, :].numpy()

	return res_mapper.get(int(catboost_bert_model.predict(batch_embeddings)))


def get_tf_idf_pred(text: str) -> str:

	res_mapper = {
		0: 'Контактная информация отсутствует',
		1: 'Есть контактная информация'
	}

	if len(text) == 0:
		return res_mapper.get(0)

	def remove_symbols(data):
		return re.sub('[/*,;-]', '', data)

	def remove_punc(data):
		trans = str.maketrans('', '', string.punctuation)
		return data.translate(trans)

	def white_space(data):
		return ' '.join(data.split())

	def lemmatization(data):
		return ' '.join([WordNetLemmatizer().lemmatize(word) for word in data.split()])

	def complete_noise(data):
		new_data = remove_symbols(data)
		new_data = remove_punc(new_data)
		new_data = white_space(new_data)
		new_data = lemmatization(new_data)
		return new_data

	text = complete_noise(text)
	with open('src/model/tf_idf.pk', 'rb') as fin:
		tf_idf = pickle.load(fin)
	tf_idf.vocabulary_ = tf_idf_vocab.to_dict()['0']
	# tf_idf_new = TfidfVectorizer(ngram_range=(1, 5), vocabulary=tf_idf_vocab.to_dict()['0'])
	# st.write(tf_idf.get_params())
	bag_of_words = tf_idf.transform([text])

	try:
		return res_mapper.get(int(catboost_tf_idf_model.predict(bag_of_words)))
	except:
		return 'В сообщении встречаются слова, отсутствующие в вокабуляре TF-IDF.'


def get_re_pred(text: str) -> str:

	url_pattern = re.compile(r'\b((?:https?://)?(?:(?:www\.)?(?:[\da-z\.-]+)\.(?:[a-z]{2,6})|(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)|(?:(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])))(?::[0-9]{1,4}|[1-5][0-9]{4}|6[0-4][0-9]{3}|65[0-4][0-9]{2}|655[0-2][0-9]|6553[0-5])?(?:/[\w\.-]*)*/?)\b')
	phone_pattern = re.compile(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})')
	if len(re.findall(url_pattern, text)) > 0:
		return 'Есть контактная информация (url)'
	elif len(re.findall(r'[\w\.-]+@[\w\.-]+(\.[\w]+)+', text)) > 0:
		return 'Есть контактная информация (mail)'
	elif len(re.findall(phone_pattern, text)) > 0:
		return 'Есть контактная информация (phone)'
	else:
		return 'Контактная информация отсутствует'

