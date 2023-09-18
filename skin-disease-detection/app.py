from flask import render_template, jsonify, Flask, redirect, url_for, request
import random
import os
import numpy as np
from keras.applications.mobilenet import MobileNet 
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input, decode_predictions
from keras.models import model_from_json
import keras
from keras import backend as K

# Chatbot imports
import json
import pickle
import nltk
import wikipedia

nltk.download('wordnet')
nltk.download('punkt')
import numpy as np
import random
from nltk.stem import WordNetLemmatizer
#####################

lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

# Chatbot loaders
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
###

SKIN_CLASSES = {
  0: 'Actinic Keratoses (Solar Keratoses) or intraepithelial Carcinoma (Bowen’s disease)',
  1: 'Basal Cell Carcinoma',
  2: 'Benign Keratosis',
  3: 'Dermatofibroma',
  4: 'Melanoma',
  5: 'Melanocytic Nevi',
  6: 'Vascular skin lesion'
}

@app.route('/')
def index():
    return render_template('index.html', title='Home')


@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = 'static/data/' + f.filename
        f.save(path)
        j_file = open('modelnew.json', 'r')
        loaded_json_model = j_file.read()
        j_file.close()
        model = model_from_json(loaded_json_model)
        model.load_weights('modelnew.h5')
        
        # Load the uploaded image from the specified path
        img1 = image.load_img(path, target_size=(224, 224))
        img1 = image.img_to_array(img1)  # Convert to numpy array
        img1 = np.expand_dims(img1, axis=0)  # Add batch dimension
        img1 = img1 / 255.0  # Normalize
        
        prediction = model.predict(img1)
        pred = np.argmax(prediction)
        disease = SKIN_CLASSES[pred]
        accuracy = prediction[0][pred]
        K.clear_session()
    return render_template('uploaded.html', title='Success', predictions=disease, acc=accuracy*100, img_file=f.filename)


# Chatbot Route
@app.route("/chatbot")
def home():
  return render_template("chatbot.html")

# Chatbot Functions

def clean_up_sentence(sentence):
  sentence_words = nltk.word_tokenize(sentence)
  sentence_words = [
    lemmatizer.lemmatize(word.lower()) for word in sentence_words
  ]
  return sentence_words


def bow(sentence, words, show_details=True):  # tokenize the pattern
  sentence_words = clean_up_sentence(sentence)
  bag = [0] * len(words)

  for s in sentence_words:
    for i, w in enumerate(words):
      if w == s:
        bag[i] = 1
        if show_details:
          print("found in bag: %s" % w)

  return (np.array(bag))


def predict_class(sentence, model):
  p = bow(sentence, words, show_details=False)
  res = model.predict(np.array([p]))[0]
  ERROR_THRESHOLD = 0.25
  results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
  results.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in results:
    return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
  return return_list


def getResponse(ints, intents_json):
  tag = ints[0]['intent']
  list_of_intents = intents_json['intents']
  for i in list_of_intents:
    if (i['tag'] == tag):
      result = random.choice(i['responses'])
      break
  return result


def chatbot_response(text):
  ints = predict_class(text, model)  #print(ints)
  res = getResponse(ints, intents)

  return res


@app.route("/get")
def get_bot_response():
  user_text = request.args.get('msg')
  bot_text = chatbot_response(user_text)
  if str(bot_text) == "Searching for you":
    try:
      query = user_text.lower().replace("search", "")
      bot_text = wikipedia.summary(query, sentences=2)
    except wikipedia.exceptions.DisambiguationError as e:
      options = e.options[:5]  # limit to 5 options for brevity
      bot_text = f"Which one do you mean? Please choose from the following options:\n{', '.join(options)}"
    except wikipedia.exceptions.PageError:
      bot_text = "Sorry, I couldn't find a Wikipedia page related to your search query."
    except wikipedia.exceptions.WikipediaException:
      bot_text = "Sorry, something went wrong while searching Wikipedia. Please try again later."
  return str(bot_text)
#####################

if __name__ == "__main__":
    app.run(debug=True)
