import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import json
import random
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()

# --- Emotion Detection Model ---
try:
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    emotion_model = pickle.load(open('emotional_model.pkl', 'rb'))
except Exception as e:
    print(f"Error loading emotion model/vectorizer: {e}")
    exit()

# --- Chatbot Response Model (Keep your existing logic) ---
try:
    model = load_model('model.h5')  # Only for responses, not emotion detection
except:
    print("Warning: model.h5 not loaded. Chatbot responses may not work.")

try:
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
except Exception as e:
    print(f"Error loading intents.json: {e}")
    exit()

try:
    words = pickle.load(open('texts.pkl', 'rb'))
    classes = pickle.load(open('labels.pkl', 'rb'))
except Exception as e:
    print(f"Error loading vocabulary files: {e}")
    exit()

# --- Only these emotions will be tracked ---
EMOTIONS = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

def reset_emotion_file():
    try:
        with open('detected_emotions.txt', 'w', encoding='utf-8') as f:
            for emotion in EMOTIONS:
                f.write(f"{emotion}:0\n")
    except Exception as e:
        print(f"Error resetting emotion file: {e}")

def update_emotion_count(emotion):
    try:
        with open('detected_emotions.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        counts = {e: 0 for e in EMOTIONS}
        for line in lines:
            if ':' in line:
                e, count = line.strip().split(':')
                e = e.strip()
                if e in counts:
                    counts[e] = int(count.strip())
        
        if emotion in counts:
            counts[emotion] += 1
        
        with open('detected_emotions.txt', 'w', encoding='utf-8') as f:
            for e in EMOTIONS:
                f.write(f"{e}:{counts[e]}\n")
    except Exception as e:
        print(f"Error updating emotion counts: {e}")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    try:
        p = bow(sentence, words, show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    except Exception as e:
        print(f"Prediction error: {e}")
        return []

def getResponseAndTip(ints, intents_json):
    try:
        if ints:
            tag = ints[0]['intent']
            for i in intents_json['intents']:
                if i['tag'] == tag:
                    response = random.choice(i['responses'])
                    tip = random.choice(i['tips']) if 'tips' in i else "No tip available."
                    response = response.replace('[Bot]', '').strip()
                    tip = tip.replace('[Tip]', '').strip()
                    return response, tip
        return "Sorry, I didn't understand that.", "No tip available."
    except Exception as e:
        print(f"Response generation error: {e}")
        return "Something went wrong.", "Please try again."

def chatbot_response(msg):
    try:
        # --- Detect emotion using the new model ---
        text_vector = vectorizer.transform([msg])
        detected_emotion = emotion_model.predict(text_vector)[0]
        update_emotion_count(detected_emotion)

        # --- Get chatbot response (keep your existing logic) ---
        ints = predict_class(msg, model)
        response, tip = getResponseAndTip(ints, intents)
        return response, tip
    except Exception as e:
        print(f"Chatbot error: {e}")
        return "I'm having trouble responding right now.", "Please try again later."

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/start_chat")
def start_chat():
    reset_emotion_file()
    return render_template("chat.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if not userText or len(userText.strip()) == 0:
        return "Please type a message.Tip: Start with how you're feeling today."
    response, tip = chatbot_response(userText)
    return f"{response}Tip: {tip}"

if __name__ == "__main__":
    reset_emotion_file()
    app.run(port=5000, debug=False)



#MODIFIED KERAS

# # First: Install TensorFlow (run this in terminal)
# # pip install tensorflow

# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# import pickle
# import numpy as np
# from tensorflow.keras.models import load_model
# import json
# import random
# from flask import Flask, render_template, request, make_response

# lemmatizer = WordNetLemmatizer()

# # Load model with error handling
# try:
#     model = load_model('model.h5')
# except Exception as e:
#     print(f"Error loading model: {e}")
#     exit()

# # Load intents.json with error handling
# try:
#     with open('intents.json', 'r', encoding='utf-8') as f:
#         intents = json.load(f)
# except Exception as e:
#     print(f"Error loading intents.json: {e}")
#     exit()

# # Load vocabulary files with error handling
# try:
#     words = pickle.load(open('texts.pkl', 'rb'))
#     classes = pickle.load(open('labels.pkl', 'rb'))
# except Exception as e:
#     print(f"Error loading vocabulary files: {e}")
#     exit()

# # 10 general emotions to detect and count
# GENERAL_EMOTIONS = [
#     'Sadness', 'Happiness', 'Anger', 'Fear', 'Surprise',
#     'Depression', 'Neutral', 'Anxiety', 'Love', 'Confusion'
# ]

# def reset_emotion_file():
#     try:
#         with open('detected_emotions.txt', 'w', encoding='utf-8') as f:
#             for emotion in GENERAL_EMOTIONS:
#                 f.write(f"{emotion}: 0\n")
#     except Exception as e:
#         print(f"Error resetting emotion file: {e}")

# def update_emotion_count(emotion):
#     try:
#         # Read current counts
#         with open('detected_emotions.txt', 'r', encoding='utf-8') as f:
#             lines = f.readlines()
        
#         counts = {e: 0 for e in GENERAL_EMOTIONS}
#         for line in lines:
#             if ':' in line:
#                 e, count = line.strip().split(':')
#                 e = e.strip()
#                 if e in counts:
#                     counts[e] = int(count.strip())
        
#         # Update count for detected emotion
#         if emotion in counts:
#             counts[emotion] += 1
        
#         # Write back updated counts
#         with open('detected_emotions.txt', 'w', encoding='utf-8') as f:
#             for e in GENERAL_EMOTIONS:
#                 f.write(f"{e}: {counts[e]}\n")
                
#     except Exception as e:
#         print(f"Error updating emotion counts: {e}")

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

# def bow(sentence, words, show_details=True):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for s in sentence_words:
#         for i, w in enumerate(words):
#             if w == s:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence, model):
#     try:
#         p = bow(sentence, words, show_details=False)
#         res = model.predict(np.array([p]))[0]
#         ERROR_THRESHOLD = 0.25
#         results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#         results.sort(key=lambda x: x[1], reverse=True)
#         return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
#     except Exception as e:
#         print(f"Prediction error: {e}")
#         return []

# def getResponseAndTip(ints, intents_json):
#     try:
#         if ints:
#             tag = ints[0]['intent']
#             for i in intents_json['intents']:
#                 if i['tag'] == tag:
#                     response = random.choice(i['responses'])
#                     tip = random.choice(i['tips']) if 'tips' in i else "No tip available."
#                     # Clean response from any placeholder tags
#                     response = response.replace('[Bot]', '').strip()
#                     tip = tip.replace('[Tip]', '').strip()
#                     return response, tip
#         return "Sorry, I didn't understand that.", "No tip available."
#     except Exception as e:
#         print(f"Response generation error: {e}")
#         return "Something went wrong.", "Please try again."

# def chatbot_response(msg):
#     try:
#         ints = predict_class(msg, model)
#         response, tip = getResponseAndTip(ints, intents)
        
#         # Get detected emotion (ensure it's valid)
#         detected_emotion = ints[0]['intent'] if ints else 'Neutral'
#         if detected_emotion not in GENERAL_EMOTIONS:
#             detected_emotion = 'Neutral'
        
#         update_emotion_count(detected_emotion)
#         return response, tip
#     except Exception as e:
#         print(f"Chatbot error: {e}")
#         return "I'm having trouble responding right now.", "Please try again later."

# app = Flask(__name__)
# app.static_folder = 'static'

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     try:
#         userText = request.args.get('msg')
#         if not userText or len(userText.strip()) == 0:
#             return "Please type a message.\n💡 Tip: Start with how you're feeling today."
        
#         response, tip = chatbot_response(userText)
#         return f"{response}\n💡 Tip: {tip}"
    
#     except Exception as e:
#         print(f"API error: {e}")
#         return "An error occurred.\n💡 Tip: Please try again later."

# if __name__ == "__main__":
#     reset_emotion_file()
#     app.run(port=5000, debug=False)  # Set debug=False for production






    
#  BERT CLASSIFICATION MODEL

# import json
# import random
# import torch
# import pickle
# from transformers import BertTokenizer, BertForSequenceClassification, pipeline
# from flask import Flask, render_template, request, make_response
# app = Flask(__name__)

# # List of 10 general emotions to track (modify as needed)
# EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral", "anticipation", "trust", "shame"]

# class MentalHealthChatbot:
#     def __init__(self):
#         # Load BERT model for intent classification
#         self.model = BertForSequenceClassification.from_pretrained('./saved_model')
#         self.tokenizer = BertTokenizer.from_pretrained('./saved_model')
#         with open('label_map.pkl', 'rb') as f:
#             self.label_map = pickle.load(f)
        
#         # Load emotion detection pipeline
#         self.emotion_detector = pipeline(
#             "text-classification", 
#             model="j-hartmann/emotion-english-distilroberta-base",
#             top_k=1
#         )
        
#         # Load intents and responses
#         with open('intents.json', 'r', encoding='utf-8') as f:
#             self.intents = json.load(f)
        
#         # Initialize emotion counts
#         self.emotion_counts = {e: 0 for e in EMOTIONS}
#         self.emotion_file = "detected_emotions.txt"
    
#     def _classify_intent(self, text):
#         encoding = self.tokenizer(
#             text, 
#             truncation=True, 
#             padding='max_length', 
#             max_length=64, 
#             return_tensors="pt"
#         )
#         with torch.no_grad():
#             outputs = self.model(**encoding)
#         logits = outputs.logits
#         pred_idx = torch.argmax(logits, dim=1).item()
#         return [k for k, v in self.label_map.items() if v == pred_idx][0]
    
#     def _detect_emotion(self, text):
#         result = self.emotion_detector(text)[0][0]
#         return result['label'].lower(), result['score']
    
#     def _update_emotion_file(self):
#         with open(self.emotion_file, 'w') as f:
#             for e in sorted(self.emotion_counts.keys()):
#                 f.write(f"{e} : {self.emotion_counts[e]}\n")
    
#     def get_response(self, user_input):
#         # Detect intent and emotion
#         intent = self._classify_intent(user_input)
#         emotion, confidence = self._detect_emotion(user_input)
        
#         # Update emotion counts
#         if emotion in self.emotion_counts:
#             self.emotion_counts[emotion] += 1
#         else:
#             print(f"Warning: Unknown emotion detected: {emotion}")
        
#         # Find matching intent and select response and tip
#         for intent_data in self.intents['intents']:
#             if intent_data['tag'] == intent:
#                 response = random.choice(intent_data['responses'])
#                 tip = random.choice(intent_data['tips'])
#                 break
#         else:
#             response = "I'm still learning. Could you rephrase that?"
#             tip = "Try expressing your thoughts in different words."
        
#         # Prepare output text
#         output = f"{response}\n💡 Tip: - {tip}"
        
#         # Update emotion file
#         self._update_emotion_file()
        
#         # Return output for Node.js
#         return output
    
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/get')
# def get_bot_response():
#     user_text = request.args.get('msg')
#     if not user_text:
#         return "Please say something."
    
#     # Initialize chatbot and get response
#     chatbot = MentalHealthChatbot()
#     response = chatbot.get_response(user_text)
    
#     # Create response object with proper headers
#     resp = make_response(response)
#     resp.headers['Content-Type'] = 'text/plain'
#     return resp

# if __name__ == '__main__':
#     app.run(port=5000, debug=True)



# #GPT FOR CHATBOT

# from flask import Flask, render_template, request
# from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel
# import torch

# app = Flask(__name__, static_folder='static')

# # Emotion detection model
# emotion_detector = pipeline(
#     "text-classification",
#     model="j-hartmann/emotion-english-distilroberta-base",
#     top_k=1
# )

# # GPT model for chat
# gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# gpt_model = GPT2LMHeadModel.from_pretrained('gpt2')

# # Emotion tracking
# EMOTIONS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral", "anticipation", "trust", "shame"]
# emotion_counts = {e: 0 for e in EMOTIONS}
# EMOTION_FILE = "detected_emotions.txt"

# def update_emotion_file():
#     with open(EMOTION_FILE, 'w') as f:
#         for e in sorted(emotion_counts.keys()):
#             f.write(f"{e}: {emotion_counts[e]}\n")

# def detect_emotion(text):
#     result = emotion_detector(text)[0][0]
#     emotion = result['label'].lower()
#     return emotion

# def generate_response_and_tip(user_message):
#     # Prompt GPT to generate both a response and a tip
#     prompt = (
#         f"User: {user_message}\n"
#         "Chatbot: [Response]\n"
#         "💡 Tip: [Tip]"
#     )
#     inputs = gpt_tokenizer(prompt, return_tensors="pt", return_attention_mask=True)
#     outputs = gpt_model.generate(**inputs, max_new_tokens=100)
#     full_output = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
#     # Extract response and tip
#     response_part = full_output.split("Chatbot:")[-1].split("💡 Tip:")[0].strip()
#     tip_part = full_output.split("💡 Tip:")[-1].strip() if "💡 Tip:" in full_output else "No tip generated."
#     return response_part, tip_part

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get")
# def get_bot_response():
#     user_text = request.args.get('msg')
#     if not user_text:
#         return "Please say something.\n💡 Tip: Try sharing your thoughts."
    
#     # Detect emotion and update counts
#     emotion = detect_emotion(user_text)
#     if emotion in emotion_counts:
#         emotion_counts[emotion] += 1
#     else:
#         print(f"Warning: Unknown emotion detected: {emotion}")
#     update_emotion_file()
    
#     # Generate response and tip with GPT
#     response, tip = generate_response_and_tip(user_text)
#     return f"{response}\n💡 Tip: {tip}"

# if __name__ == "__main__":
#     # Initialize emotion file with zeros
#     update_emotion_file()
#     app.run(port=5000)






























