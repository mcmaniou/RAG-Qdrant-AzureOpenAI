import argparse
import os
from pathlib import Path

import yaml
from openai import AzureOpenAI
from qdrant_client import QdrantClient

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


# the virtual assistant functionality, its basic steps are:
# 1. It receives a question as input.
# 2. It searches the database and retrieves relevant content.
# 3. It queries the OpenAI model given the question and the relevant content.
# 4. It returns the response.
def answer_question(question):

    # read arguments
    conf_file = Path(MYDIR, "conf.yml")
    args = read_args(conf_file)

    # set up qdrant client and collection
    qdrant = QdrantClient(host=args.qdrant["host"], port=args.qdrant["port"])
    collection_name = args.qdrant["collection_name"]

    # set up azure-openai client
    azure_client = AzureOpenAI(
        api_version=args.openai_azure["api_version"],
        azure_endpoint=args.openai_azure["azure_endpoint"],
        api_key=args.openai_azure["api_key"],
    )

    # query collection for relevant content
    result = qdrant.search(
        collection_name=collection_name,
        query_vector=azure_client.embeddings.create(
            input=["What is the concave mirror?"],
            model=args.openai_azure["embedding_engine"],
        )
            .data[0]
            .embedding,
        limit=5,
    )

    # format prompt
    context = []
    for one_res in result:
        context.append(one_res.payload['text'])

    context = ",".join(str(element) for element in context)

    prompt = f"Use the following pieces of context to answer the question " \
             f"enclosed within 3 backticks at the end. If you do not know the " \
             f"answer, just say that you do not know given the provided resources, " \
             f"do not try to make up an " \
             f"answer. Please provide an answer which is factually correct and " \
             f"based on the information retrieved from the vector store. Please " \
             f"also mention any quotes supporting the answer if any present in " \
             f"the context supplied within two double quotes. {context} " \
             f"QUESTION:```{question}``` ANSWER: "

    # query OpenAI model
    response = azure_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "assistant",
                "content": prompt,
            },
        ],
        temperature=0
    )

    return response.choices[0].message.content
