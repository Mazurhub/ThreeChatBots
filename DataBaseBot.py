import os
import pandas as pd
import openai
import pinecone
import uuid
import tiktoken
from tqdm.auto import tqdm

openai.api_key = "OPENAI_API_KEY"

pinecone.init(api_key="PINECONE_API_KEY", environment="PINECONE_ENV")
index_name = "benscat"

# Check if the index exists, if not create it
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536, metric='cosine')

index = pinecone.Index(index_name)

# Load data
contents = []
tiktoken_encoding = tiktoken.get_encoding("gpt2")
for file in os.listdir("."):
    if file.endswith(".txt"):
        with open(file, "r") as f:
            file_content = f.read()
        tokens = tiktoken_encoding.encode(file_content)
        total_tokens = len(tokens)
        contents.append((file, file_content, total_tokens))

df = pd.DataFrame(contents, columns=['filename', 'file_content', 'tokens'])
df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]

# Create embeddings with OpenAI
def create_embeddings(text):
    embedding = openai.Embedding.create(input=text, engine='text-embedding-ada-002')
    return embedding['data'][0]['embedding']


df['embeddings'] = df.file_content.apply(create_embeddings)

# Save embeds to Pinecone
batch_size = 10  # The number of embeds to process at once
for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(len(df), i + batch_size)
    embeddings_batch = df[i:i_end][['id', 'embeddings', 'filename', 'file_content']].to_dict(orient='records')

    # Convert the list of embedding vectors to the appropriate Pinecone format
    embeddings_batch = [
        {
            'id': item['id'],
            'values': item['embeddings'],
            'metadata': {
                'filename': item['filename'],
                'file_content': item['file_content']
            }
        }
        for item in embeddings_batch
    ]

    index.upsert(vectors=embeddings_batch)


# A function for constructing a question and answer
def get_answer(question):
    # Create an embedding for the question using the same OpenAI model
    question_embedding = create_embeddings(question)

    # Ask Pinecone based on the question embedding
    response = index.query(question_embedding, top_k=5, include_metadata=True)

    if response['matches']:
        # Use the first matching document's answer
        context = response['matches'][0]['metadata']['file_content']
    else:
        # If no response from Pinecone, generate a default response
        context = "I don't have an answer to that question."

    return context


# Function for generating responses using the GPT-3 model
def generate_gpt3_response(question, context):
    # Invoke the GPT-3 model to generate the response
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Answer the question: '{question}'\nContext: '{context}'",
        temperature=0.0,
        max_tokens=50
    )

    return response['choices'][0]['text']


# A function to generate responses
def generate_answer(question):
    limit = 3090
    embeddings_model = "text-embedding-ada-002"
    embed_query = openai.Embedding.create(
        input=question,
        engine=embeddings_model
    )
    query_embeds = embed_query['data'][0]['embedding']

    response = index.query(query_embeds, top_k=3, include_metadata=True)
    contexts = [x['metadata']['file_content'] for x in response['matches']]

    prompt_start = "Answer the question based on the context below:\n\n"
    prompt_end = f"\n\nQuery: {question}\nAnswer:"

    prompt = None  # Initialize the prompt before the loop

    for i in range(1, len(contexts)):
        if len("-".join(contexts[:i])) >= limit:
            prompt = (prompt_start + "-".join(contexts[:i - 1]) + prompt_end)
            break

    if prompt is None:
        # If no suitable context is found, use question only
        prompt = f"Query: {question}\nAnswer:"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=350,
        top_p=1
    )

    return response['choices'][0]['text']

# Example of use
question = "Does Ben have a cat?"
answer = get_answer(question)
print(f"Query: {question}\nAnswer: {answer}")