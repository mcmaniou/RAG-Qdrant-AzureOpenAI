# RAG-Qdrant-AzureOpenAI
A virtual assistant based on the RAG architecture. This project implements the question-answering part of the architecture, utilizing a prepopulated DB. The DB used is Qdrant. And the Azure OpenAI api is used to query an LLM. 

## Get started

### 1. Set up Python dependencies
The project is written in Python version 3.10. To install the requirements please run:
```
pip install -r requirements.txt
```

### 2. Start database
The project uses a Qdrant DB to store the embeddings. The DB is set up in a docker container and port 6334 is open to access it. The `docker-compose.yml` is provided. To start the DB run in your terminal:
```
docker-compose up -d
```

### 3. Fill database
This DB is initially empty. So, if this is the first time we need to add data. 

To fill it with data, please run the python script `store_pdfs.py`. It reads pdf files from a user-provided filepath. 

**The necessary parametrization can be provided using a `conf.yml` file in the same directory as your working environment.** The `conf-example.yml` file is provided as an example.

### 4. Start the virtual assistant
To start the virtual assistant run the `app.py` script. 

As previously, **the necessary parametrization can be provided using a `conf.yml` file in the same directory as your working environment.** The `conf-example.yml` file is provided as an example. 

The app runs in `localhost:5000`. And the UI looks like the following image. You can start interacting with the virtual assistant! The history is also printed in the screen.

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.