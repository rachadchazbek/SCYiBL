from operator import mod
from transformers import pipeline, PreTrainedMoldel

classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
# data = 

model = pipeline("zero-shot-classification")
# model(data, candidate_labels=classes)
path = ''
PreTrainedMoldel.save_preTrained(model, 'path')