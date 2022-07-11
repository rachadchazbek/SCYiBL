from operator import mod
from transformers import pipeline

classes = ['CP', 'Constructivism', 'Coping', 'ED', 'ER/SR', 'Empowerment', 'Exposure/Habit', 'PERMA']
# data = 

model = pipeline("zero-shot-classification")
# model(data, candidate_labels=classes)
path = '/Users/rachadchazbek/azimuth/azimuth_shr'
model.save_pretrained(path, ) 