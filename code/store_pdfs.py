import os
import yaml
import argparse
from pathlib import Path
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams, Distance

MYDIR = Path(__file__).parent


# read args from conf file
def read_args(conf_file):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    conf = {}

    if os.path.isfile(conf_file):
        with open(conf_file) as f:
            conf.update(yaml.safe_load(f))

    conf_args = vars(args)
    conf_args.update(conf)

    return args


# get textx from pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # read each page
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# get text chunks from raw text
def get_text_chunks(raw_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


# generate embeddings
def generate_embeddings(text, model, client):
    return client.embeddings.create(input=text, model=model)


def main():
    # read arguments
    conf_file = Path(MYDIR, "conf.yml")
    args = read_args(conf_file)

    # get filepaths of pdf files
    try:
        filepaths = os.listdir(args.filepaths['pdfs_folder'])
    except:
        print(f"The 'pdfs_folder' filepath ({args.filepaths['pdfs_folder']}) was not "
              f"readable.")
        return

    filepaths = [args.filepaths['pdfs_folder'] + i for i in filepaths]

    # set up azure-openai client
    azure_client = AzureOpenAI(
        api_version=args.openai_azure['api_version'],
        azure_endpoint=args.openai_azure['azure_endpoint'],
        api_key=args.openai_azure['api_key']
    )

    # set up qdrant client and collection
    qdrant = QdrantClient(host=args.qdrant["host"], port=args.qdrant["port"])
    collection_name = args.qdrant['collection_name']

    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )

    for one_doc in filepaths:

        filename = one_doc.replace(args.filepaths['pdfs_folder'], '')
        try:
            # read file and generate embeddings
            raw_text = get_pdf_text(filepaths)
            text_chunks = get_text_chunks(raw_text)

            embeddings = generate_embeddings(text_chunks,
                                             args.openai_azure['embedding_engine'],
                                             azure_client)

            # transform embeddings to qdrant points and store them
            points = [
                PointStruct(
                    id=idx,
                    vector=data.embedding,
                    payload={'text': text,
                             'file': filename},
                )
                for idx, (data, text) in enumerate(zip(embeddings.data, text_chunks))
            ]

            qdrant.upsert(collection_name, points)
            print(f"Successfully stored {filename} file.")

        except Exception as e:
            print(f"The following error occurred while storing the {filename} file:"
                  f" {e}")
            continue


if __name__ == '__main__':
    main()
