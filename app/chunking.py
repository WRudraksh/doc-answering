from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap =100,
        length_function = len,
        separators = ["\n\n", "\n", " ", ""] 
    )
    return splitter

def chunk_text(text : str):
    splitter = get_text_splitter()
    chunks =splitter.split_text(text)

    structured_chunks=[]
    for i, chunk in enumerate(chunks):
        structured_chunks.append({
            "chunk_id": i,
            "text": chunk
        })
    return structured_chunks