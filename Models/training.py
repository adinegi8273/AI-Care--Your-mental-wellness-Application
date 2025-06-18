import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents with UTF-8 encoding
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Process intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # Add to documents
        documents.append((w, intent['tag']))
        # Add to classes
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and lowercase words
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes
pickle.dump(words, open('texts.pkl', 'wb'))
pickle.dump(classes, open('labels.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

# Create model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train and save model
hist = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('model.h5', hist)
print("Model training complete!")



# import nltk
# nltk.download('wordnet')
# from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# import json
# import pickle
# import numpy as np
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Dropout
# from keras.optimizers import SGD
# import random
# words=[]
# classes = []
# documents = []
# ignore_words = ['?', '!']
# data_file = open('intents.json').read()
# intents = json.loads(data_file)

# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         #tokenize each word
#         w = nltk.word_tokenize(pattern)
#         words.extend(w)
#         #add documents in the corpus
#         documents.append((w, intent['tag']))
#         # add to our classes list
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])
# # lemmatize and lower each word and remove duplicates
# words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# words = sorted(list(set(words)))
# # sort classes
# classes = sorted(list(set(classes)))
# # documents = combination between patterns and intents
# print (len(documents), "documents")
# # classes = intents
# print (len(classes), "classes", classes)
# # words = all words, vocabulary
# print (len(words), "unique lemmatized words", words)
# pickle.dump(words,open('texts.pkl','wb'))
# pickle.dump(classes,open('labels.pkl','wb'))
# # create our training data
# training = []
# # create an empty array for our output
# output_empty = [0] * len(classes)
# # training set, bag of words for each sentence
# for doc in documents:
#     # initialize our bag of words
#     bag = []
#     # list of tokenized words for the pattern
#     pattern_words = doc[0]
#     # lemmatize each word - create base word, in attempt to represent related words
#     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
#     # create our bag of words array with 1, if word match found in current pattern
#     for w in words:
#         bag.append(1) if w in pattern_words else bag.append(0)
    
#     # output is a '0' for each tag and '1' for current tag (for each pattern)
#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
    
#     training.append([bag, output_row])
# # shuffle our features and turn into np.array
# random.shuffle(training)
# train_x, train_y = zip(*training)
# train_x = np.array(train_x)
# train_y = np.array(train_y)
# print("Training data created")
# # Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# # equal to number of intents to predict output intent with softmax
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(train_y[0]), activation='softmax'))
# # Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
# sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# #fitting and saving the model 
# hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
# model.save('model.h5', hist)
# print("model created")


# import json
# import pandas as pd
# import torch
# from sklearn.model_selection import train_test_split
# from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
# from torch.utils.data import Dataset
# import numpy as np

# Load intents.json
# with open('intents.json', 'r', encoding='utf-8') as f:
#     intents = json.load(f)

# # Prepare data: list of (text, label) pairs
# data = []
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         data.append({"text": pattern, "label": intent['tag']})
# df = pd.DataFrame(data)

# # Split into train and validation
# train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# # Create label mapping
# label_map = {tag: idx for idx, tag in enumerate(df['label'].unique())}
# train_labels = [label_map[label] for label in train_df['label']]
# val_labels = [label_map[label] for label in val_df['label']]

# # Tokenize with BERT
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# class IntentDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, label_map):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.label_map = label_map
#     def __len__(self):
#         return len(self.texts)
#     def __getitem__(self, idx):
#         text = str(self.texts.iloc[idx])
#         label = self.labels[idx]
#         encoding = tokenizer(text, truncation=True, padding='max_length', max_length=64, return_tensors="pt")
#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor(label, dtype=torch.long)
#         }

# train_dataset = IntentDataset(train_df['text'], train_labels, tokenizer, label_map)
# val_dataset = IntentDataset(val_df['text'], val_labels, tokenizer, label_map)

# # Load BERT model
# model = BertForSequenceClassification.from_pretrained(
#     'bert-base-uncased',
#     num_labels=len(label_map)
# )

# # Training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=3,
#     per_device_train_batch_size=8,
#     per_device_eval_batch_size=8,
#     warmup_steps=500,
#     weight_decay=0.01,
#     logging_dir='./logs',
#     logging_steps=10,
# )

# # Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=val_dataset,
# )

# # Train
# trainer.train()

# # Save model and tokenizer
# model.save_pretrained('./saved_model')
# tokenizer.save_pretrained('./saved_model')

# # Save label map (optional, for inference)
# import pickle
# with open('label_map.pkl', 'wb') as f:
#     pickle.dump(label_map, f)

# print("BERT model training complete. Saved model, tokenizer, and label map.")


























































# # import nltk
# # nltk.download('wordnet')
# # nltk.download('punkt')
# # from nltk.stem import WordNetLemmatizer
# # import json
# # import pickle
# # import numpy as np
# # from keras.models import Sequential
# # from keras.layers import Dense, Activation, Dropout
# # from keras.optimizers import SGD
# # import random

# # lemmatizer = WordNetLemmatizer()

# # # Load intents.json with UTF-8 encoding
# # with open('intents.json', 'r', encoding='utf-8') as f:
# #     intents = json.load(f)

# # words = []
# # classes = []
# # documents = []
# # ignore_words = ['?', '!']

# # for intent in intents['intents']:
# #     for pattern in intent['patterns']:
# #         # Tokenize each word
# #         w = nltk.word_tokenize(pattern)
# #         words.extend(w)
# #         # Add document to corpus
# #         documents.append((w, intent['tag']))
# #         # Add to classes list
# #         if intent['tag'] not in classes:
# #             classes.append(intent['tag'])

# # # Lemmatize, lowercase, and remove duplicates
# # words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
# # words = sorted(list(set(words)))
# # classes = sorted(list(set(classes)))

# # # Save words and classes
# # pickle.dump(words, open('texts.pkl', 'wb'))
# # pickle.dump(classes, open('labels.pkl', 'wb'))

# # # Create training data
# # training = []
# # output_empty = [0] * len(classes)

# # for doc in documents:
# #     bag = []
# #     pattern_words = doc[0]
# #     pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
# #     for w in words:
# #         bag.append(1) if w in pattern_words else bag.append(0)
# #     output_row = list(output_empty)
# #     output_row[classes.index(doc[1])] = 1
# #     training.append([bag, output_row])

# # random.shuffle(training)
# # train_x, train_y = zip(*training)
# # train_x = np.array(train_x)
# # train_y = np.array(train_y)

# # # Build model
# # model = Sequential()
# # model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.5))
# # model.add(Dense(len(train_y[0]), activation='softmax'))

# # # Compile model
# # sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
# # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # # Train model
# # model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# # # Save model
# # model.save('model.h5')

# # print("Model training complete. Saved model.h5, texts.pkl, and labels.pkl.")



































