from langchain.text_splitter import RecursiveCharacterTextSplitter
import json


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Function to split the text into chunks using LangChain


def split_text_into_chunks(text, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]

    )
    documents = text_splitter.create_documents([text])
    chunks = [{"id": idx + 1, "content": doc.page_content,
               "metadata": doc.metadata} for idx, doc in enumerate(documents)]

    # Write chunks to a file as JSON
    with open("chunks.txt", "w") as file:
        file.write(json.dumps(chunks, indent=4))

    return chunks


# Path to your local file
file_path = "W:\coding_projects\AI\scrimba_ai_engineer_track\langchain_app\sample_data.txt"

# Reading and splitting the text
text = read_text_file(file_path)
chunks = split_text_into_chunks(text)

print(chunks)

# # Output the chunks
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i + 1}:\n{chunk}\n{'-' * 50}")
