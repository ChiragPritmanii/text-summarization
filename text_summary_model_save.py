# -*- coding: utf-8 -*-
"""Text Summary Model Save.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vUd6OhiJF7PVlU27VsmyNROrJljJOKsc
"""

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")


model.save_pretrained('./gdrive/MyDrive/bart-large-cnn')
tokenizer.save_pretrained('./gdrive/MyDrive/bart-large-cnn')