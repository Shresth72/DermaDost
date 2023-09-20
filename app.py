from flask import render_template, Flask, request
import random
import numpy as np

from tensorflow.keras.applications import EfficientNetB3
import numpy as np
from tensorflow.keras.models import model_from_json,load_model

from keras import backend as K
import cv2
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
# Load your image classification model and setup variables
eff3_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(180, 180, 3))
for layer in eff3_model.layers:
    layer.trainable = False

model_json_file = 'model.json'
model_weights_file = 'model_weights.h5'
with open(model_json_file, "r") as json_file:
    loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_file)

def get_class(image_path):
    try:
        img = cv2.imread(image_path)
    except BaseException:
        return 'false'
    else:
        img = cv2.resize(img, (180, 180))
        img = np.array(img)
        img = img.reshape(1, 180, 180, 3)
        x_t = eff3_model.predict(img)
        x_t = x_t.reshape(1, -1)

        # Predict the result
        result = loaded_model.predict(x_t)
        # Define class names based on your dataset
        class_names = ['Acnes', 'Healthy', 'Vitiligo', 'Fungal Infections',
                       'Melanoma Skin Cancer and Moles', 'Eczema']
        predicted_class_index = np.argmax(result)
        predicted_class_name = class_names[predicted_class_index]
        print(predicted_class_name)
        return predicted_class_name
# Chatbot loaders
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

@app.route('/')
def index():
    return render_template('index.html', title='Home')


@app.route('/uploaded', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        path = 'static/data/' + f.filename
        f.save(path)
        disease = get_class(path)
        K.clear_session()
    return render_template('uploaded.html', title='Success', predictions=disease, acc=100, img_file=f.filename)


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
