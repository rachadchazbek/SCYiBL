from transformers import pipeline
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf

# BERT 
modelName = "distilbert-base-uncased-finetuned-sst-2-english"
model = TFAutoModelForSequenceClassification.from_pretrained(modelName)
tokenizer = AutoTokenizer.from_pretrained(modelName)

rawInputs = ["Nevermind, I don't want to do this", "Meditating is actually helpfull"]
inputs = tokenizer(rawInputs, padding=True, truncation=True, return_tensors="tf")
outputs = model(inputs)
print(outputs.logits.shape)
print(outputs.logits)
predictions = tf.math.softmax(outputs.logits, axis=-1)
print(predictions)

model.config.id2label

# classifier = pipeline("zero-shot-classification")
# classes = ['Cognitive Processing', 'Coping', 'Emotion recognition', 'Empowerment', 'PERMA']
# res = classifier("Nevermind, I don't want to do this", candidate_labels=classes,  model=model, tokenizer=tokenizer)
# print(res)