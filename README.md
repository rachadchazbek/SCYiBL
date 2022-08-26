# SCYiBL-NLP

### To build the Docker image:
`docker build -t scyiblnlp . `

Then proceed by running the Docker image created in The Docker Desktop Dashboard.
Depending on your machine, this process might take a good amount of time, so be patient.

### To make a prediction: 
## POST - `127.0.0.1:5010` - Makes a prediction
Content-Type: application/json

With body: 
{ "message": {message} }

## Folder structure
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




