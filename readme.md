# AI Assistant with LangChain and Sentence Transformers

This project is an interactive AI assistant built using LangChain, Sentence Transformers, and Supabase for vector search. The assistant provides context-aware responses based on a conversation history and context, leveraging the power of a SentenceTransformer model and the Ollama LLaMA language model.

## Features

- Context-aware conversational AI assistant.
- Sentence embedding generation using Sentence Transformers.
- Vector-based document search using Supabase.
- LLaMA-based natural language processing for generating responses.

## Requirements

- Python 3.8+
- Supabase account and service key.
- The following Python libraries:
  - `langchain`
  - `sentence-transformers`
  - `supabase`
  - `langchain-ollama`

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/wasifsn/LLaMa_chatbot.git
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure your Supabase credentials:
   - Replace `SUPABASE_URL` and `SUPABASE_KEY` in the script with your Supabase project URL and service key.

4. Run the program:
   ```bash
   python main.py
   ```

## Usage

- Start the program and interact with the AI assistant through the command line.
- Type your questions, and the AI will provide responses based on context and conversation history.
- Type `exit`, `bye`, or `quit` to end the session.

## Key Components

### Sentence Embedding

The project uses the `all-mpnet-base-v2` model from Sentence Transformers to generate 768-dimensional sentence embeddings. These embeddings are duplicated to create 1536-dimensional vectors for compatibility with Supabase.

### Vector Search

Relevant documents are retrieved from Supabase using a custom RPC function `match_documents`. The function matches query embeddings with stored vectors based on similarity.

### Response Generation

The assistant uses a PromptTemplate from LangChain to structure its queries and utilizes Ollama LLaMA for generating responses.

### Conversation History

The conversation history is maintained to provide context for more accurate and meaningful responses.

## Environment Variables

Ensure you set the following environment variables:

- `SUPABASE_URL`: Your Supabase project URL.
- `SUPABASE_KEY`: Your Supabase service role key.

## Example Interaction

```text
You: What is the purpose of this project?
AI: This project demonstrates an AI assistant that provides context-aware responses based on conversation history and context. It leverages LangChain, Sentence Transformers, and Supabase.

You: How does it work?
AI: The assistant generates embeddings for user queries, retrieves relevant documents using vector search, and uses a language model to generate context-aware responses.

You: exit
Exiting the conversation. Goodbye!
```

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact

For any queries, please contact wasif4000.wn@mgmail.com
