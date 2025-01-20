import sys
import time
import threading
from langchain.schema.runnable import RunnableSequence
from sentence_transformers import SentenceTransformer
from supabase import create_client
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from itertools import cycle  # For the spinner

SUPABASE_URL = "https://djxskdtpvrvprdncsbvt.supabase.co"
SUPABASE_KEY = ""  # Replace with your key

# Initialize Sentence Transformer Model
model = SentenceTransformer('all-mpnet-base-v2')

# Initialize Ollama LLaMA 3.2
llm = OllamaLLM(model="llama3.2", temperature=0.0)

supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Template with Conversation History
template = """
You are a helpful AI assistant. Answer the following question based on the context provided:

- Be friendly.
- Do not make assumptions and answer only as per the context.
- Use the conversation history and then make an appropriate response.
- Apologize if you don't know the answer and advise the user to email help@scrimba.com.

Conversation History:
{history}

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"], template=template)


def get_1536_embedding(sentence):
    """Generate a 1536-dimensional embedding for a single sentence."""
    embedding_768 = model.encode(
        [sentence])[0]  # Get a 768-dimensional embedding
    embedding_1536 = embedding_768.tolist() + embedding_768.tolist()
    return embedding_1536


def search_nearest_vector(query_embedding, match_threshold=0.50, match_count=4):
    """Query Supabase for the nearest vector match."""
    response = supabase_client.rpc(
        'match_documents',
        {
            'query_embedding': query_embedding,
            'filter': {},
            'match_count': match_count
        }
    ).execute()
    if response.data:
        return [
            Document(page_content=result['content'], metadata={
                "id": result['id'], "similarity": result['similarity']
            })
            for result in response.data
        ]
    return []


def generate_response(question, conversation_history):
    """Pipeline to process the input and generate a response."""
    # Step 1: Generate embedding
    query_embedding = get_1536_embedding(question)

    # Step 2: Retrieve relevant documents
    relevant_docs = search_nearest_vector(
        query_embedding, match_threshold=0.50, match_count=4)

    # Step 3: Prepare input for the prompt
    context = "\n".join([doc.page_content for doc in relevant_docs])
    inputs = {
        "history": "\n".join(conversation_history),
        "context": context,
        "question": question,
    }

    # Step 4: Generate the response
    prompt_input = prompt.format(**inputs)
    response = llm.generate([prompt_input])  # Wrap in a list
    generations = response.generations
    return generations[0][0].text  # Extract the response string from the list


def show_spinner(stop_event):
    """Display a loading spinner until stop_event is set."""
    spinner = cycle(["|", "/", "-", "\\"])
    while not stop_event.is_set():
        sys.stdout.write("\rProcessing your request... " + next(spinner))
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 40 + "\r")  # Clear the spinner line


# Interactive CLI
conversation_history = []
while True:
    # Get user input
    question = input("You: ")
    if question.lower() in {"bye", "exit", "quit"}:
        print("Exiting the conversation. Goodbye!")
        break

    # Add question to conversation history
    conversation_history.append(f"User: {question}")

    # Create an event to stop the spinner
    stop_spinner_event = threading.Event()
    spinner_thread = threading.Thread(
        target=show_spinner, args=(stop_spinner_event,))
    spinner_thread.start()

    try:
        # Generate the response
        answer = generate_response(question, conversation_history)
    finally:
        # Stop the spinner
        stop_spinner_event.set()
        spinner_thread.join()

    # Add response to conversation history
    conversation_history.append(f"AI: {answer}")

    # Display the response
    print(f"AI: {answer}")
