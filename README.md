# SCYiBL-NLP

SCYiBL-NLP is a Natural Language Processing method based on Deep Learning that categorizes sentences into cognitive theories (ex: constructivism, coping).

### To build the Docker image:
`docker build -t scyiblnlp . `

### To run the container
`docker run -d -p 5010:5010 --name scyiblnlp scyiblnlp`


Depending on your machine, this process might take a good amount of time, so be patient.

### To make a prediction: 
## POST - `127.0.0.1:5010` - Makes a prediction
Headers: NA

Content-Type: application/json

With body: 
{ "message": {message} }

## Folders structure
```
  .
  ├── data
      ├── data_huggingface.csv        # dataset
  ├── src                             # Source files
      ├── main.py                     # Flask application + predict function
      ├── sentenceClassifier.py
      ├── sentenceDataset.py
      ├── dataModule.py
  ├── requirements.txt
  ├── Dockerfile                      # Docker file to build the Docker image
  ├── docker-compose.yml
  ├── docker-compose.debug.yml
  ├── README.md

```






## Contributors 
 <a href="https://github.com/rachadchazbek"><img src="https://img.shields.io/github/followers/rachadchazbek?style=social" /></a>
 <a href="https://github.com/theobadra"><img src="https://img.shields.io/github/followers/theobadra?style=social" /></a>
 <a href="https://github.com/benjamindeleener"><img src="https://img.shields.io/github/followers/benjamindeleener?style=social" /></a>




