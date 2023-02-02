#import dependencies
from flask import Flask, render_template, request
import requests
import torch
import transformers
from transformers import BartForConditionalGeneration, AutoTokenizer

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print ("Device ", torch_device)
torch.set_grad_enabled(False)

print("model initiated")
model = BartForConditionalGeneration.from_pretrained("./bart-large-cnn")
print("model initiated")
tokenizer = AutoTokenizer.from_pretrained("./bart-large-cnn")
print('tokenizer initiated')

@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


@app.route("/summary", methods=['POST'])
def predict():
    if request.method == 'POST':
        client_note =  str(request.form['text'])
        
        inputs = tokenizer([client_note], max_length=768, truncation=True, return_tensors="pt")
        outputs = model.generate(inputs["input_ids"],  num_beams=3, min_length=len((client_note).split(' '))//2, max_length= len((client_note).split(' ')),  do_sample=True)
        client_sum = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        client_sum = "\n".join(sent_tokenize(client_sum))
        
        return render_template('summary.html', notes = client_note, summary = client_sum)
    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)