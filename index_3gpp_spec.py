import json
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
import faiss
import numpy as np
import os
import tqdm
import logging
import time

ERROR_LOG_FILE = "./error_logs.log"
load_dotenv('.env')

api_key = os.getenv('OPEN_AI_API_KEY_30')
client = OpenAI(api_key = api_key)

ERROR_LOG_FILE = "./error_logs.log"


# Logger configuration
ERROR_LOG_FILE = "error_logs.log"
logger = logging.getLogger('my_logger')
file_handler = logging.FileHandler(ERROR_LOG_FILE)
logger.setLevel(logging.DEBUG)  # Set the base logging level
file_handler.setLevel(logging.ERROR)  # Set the handler logging level
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def log_error(func_name, error_message, additional_info=None):
    """
    Log errors in a structured format to a file.
    """
    error_entry = {
        "File Name": "3gpp_spec_index.py",
        "Function": func_name,
        "Error Message": error_message,
        "Timestamp": datetime.now().isoformat(),
        "Additional Info": additional_info or {}
    }
    logger.error(json.dumps(error_entry, indent=4))

def embed_texts(texts: list) -> list:
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        log_error(
            func_name="embed_texts",
            error_message=str(e),
        )
        
        return []

def build_vector_store(specs_json_path: str, index_save_path: str, batch_size: int = 16):
    """
    Builds and saves a FAISS HNSW index for high-accuracy nearest neighbor search.
    """
    try:
        # Load the JSON file
        with open(specs_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        log_error("build_vector_store", "Failed to load JSON file", {"file_path": specs_json_path, "error": str(e)})
        raise

    try:
        specs = data.get("Specifications", [])
        if not specs:
            raise ValueError("No specifications found in the JSON file.")
    except Exception as e:
        log_error("build_vector_store", "No specifications found", {"file_path": specs_json_path, "error": str(e)})
        raise

    try:
        text_reprs = [
            (
                f"Spec No: {spec.get('Spec No', '')}\n"
                f"Title: {spec.get('Title', '')}\n"
                f"Type: {spec.get('Type', '')}\n"
                f"Status: {spec.get('Status', '')}\n"
            )
            for spec in specs
            if spec.get("Spec No") and spec.get("Title")
        ]

        if not text_reprs:
            raise ValueError("No valid text representations found in the specifications.")
    except Exception as e:
        log_error("build_vector_store", "Failed to generate text representations", {"error": str(e)})
        raise

    embeddings = []
    try:
        for i in tqdm(range(0, len(text_reprs), batch_size), desc="Embedding specs"):
            batch = text_reprs[i:i + batch_size]
            try:
                batch_embeddings = embed_texts(batch)
                if not batch_embeddings:
                    log_error("build_vector_store", "Embedding batch returned no results", {"batch_index": i})
                    continue
                embeddings.extend(batch_embeddings)
            except Exception as e:
                log_error("build_vector_store", "Error during embedding generation", {"batch_index": i, "error": str(e)})
    except Exception as e:
        log_error("build_vector_store", "Error during embedding loop", {"error": str(e)})
        raise

    if not embeddings:
        log_error("build_vector_store", "No embeddings generated", {})
        raise ValueError("Embeddings list is empty. No data was processed.")

    try:
        embeddings_np = np.array(embeddings, dtype="float32")
        index = faiss.IndexHNSWFlat(1536, 32)
        index.hnsw.efConstruction = 200  # High accuracy during training
        index.hnsw.efSearch = 64  # High accuracy during querying
        index.add(embeddings_np)
    except Exception as e:
        log_error("build_vector_store", "Failed to create and populate FAISS index", {"error": str(e)})
        raise

    try:
        index_file_path = os.path.join(index_save_path, "index_hnsw.faiss")
        os.makedirs(index_save_path, exist_ok=True)
        faiss.write_index(index, index_file_path)
    except Exception as e:
        log_error("build_vector_store", "Failed to save FAISS index", {"index_file_path": index_file_path, "error": str(e)})
        raise

    try:
        # Save metadata, including text_repr
        metadata = [
            {
                "spec_index": i,
                "spec_data": specs[i],
                "text_repr": text_reprs[i]
            }
            for i in range(len(specs))
        ]
        metadata_file_path = os.path.join(index_save_path, "index_hnsw.meta.json")
        with open(metadata_file_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        log_error("build_vector_store", "Failed to save metadata", {"metadata_file_path": metadata_file_path, "error": str(e)})
        raise

    print(f"FAISS index saved at {index_file_path}.")


def retrieve_specs_hnsw(query: str, index_path: str, meta_path: str, top_k=10) -> list:
    """
    Embeds the user query, loads the FAISS HNSW index, performs a similarity search,
    and returns the top_k relevant specs with their metadata.
    """
    # Embed the query
    query_vector = embed_texts([query])  # Pass query as a list
    if not query_vector:
        raise ValueError("Failed to embed the query.")
    query_vector_np = np.array(query_vector, dtype="float32")

    # Load the HNSW index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata_list = json.load(f)

    # Perform the similarity search
    distances, indices = index.search(query_vector_np, top_k)

    # Retrieve the best matches
    best_specs = []
    for dist, idx in zip(distances[0], indices[0]):
        spec_meta = metadata_list[idx]
        best_specs.append({
            "score": float(dist),  # Smaller = more similar (L2 distance)
            "spec_data": spec_meta["spec_data"],
            "text_repr": spec_meta["text_repr"]
        })

    return best_specs



def llm_rank_top_5_specs(candidate_specs: list, user_query: str) -> list:
    """
    Given a small list of up to ~10 candidate specs (with text representations),
    call the LLM to rank them by relevance and output the top 5 in JSON.
    """

    specs_text = []
    for i, spec_item in enumerate(candidate_specs):
        short_text = spec_item["text_repr"][:2000]  # just in case
        specs_text.append(f"{i+1}) {short_text}")

    prompt = f"""
You are a 3GPP specifications expert. A user has the following query:
\"{user_query}\"

We have these candidate specifications (each is a short text snippet):
--------------------------------------------------
{chr(10).join(specs_text)}
--------------------------------------------------

Rank them by relevance to the user’s query and return only the top 5
as valid JSON (an array of objects), where each object has:


 - "Spec No"
 - "Title"
 - "Reason"

Remember, only return JSON with an array of 5 items.
    """

    response = client.chat.completions.create(model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a helpful assistant with deep knowledge of 3GPP specs."
            )
        },
        {"role": "user", "content": prompt},
    ],
    temperature=0.0)

    llm_answer = response.choices[0].message.content

    # Attempt to parse JSON
    try:
        # Remove backticks and clean the JSON
        start_idx = llm_answer.find('[')  # Find the start of the JSON array
        end_idx = llm_answer.rfind(']')  # Find the end of the JSON array
        if start_idx == -1 or end_idx == -1:
            raise ValueError("JSON array not found in the response.")

        cleaned_json = llm_answer[start_idx:end_idx + 1]  # Extract JSON content
        top_5_specs = json.loads(cleaned_json)  # Parse the JSON

    except Exception as e:
        # Raise a detailed error for debugging
        log_error("llm_rank_top_5_specs", str(e) ,f"LLM did not return valid JSON:\n{llm_answer}")
        
    return top_5_specs


def get_top_5_specs(user_query: str, faiss_index_path: str, meta_path:str) -> list:
    """
    1) Retrieves the top 10 specs using an embedding-based similarity search.
    2) (Optional) Calls an LLM to refine them to just the top 5.
    """
    # Step 1: retrieve top 10
    candidate_specs = retrieve_specs_hnsw(user_query, faiss_index_path, meta_path ,top_k=10)

    # Step 2: use LLM to rank the top 10 and produce the final top 5
    final_top_5 = llm_rank_top_5_specs(candidate_specs, user_query)
    return final_top_5
        
 
if __name__ == "__main__":
    start = time.time()
    build_vector_store("Specification_list.json","faiss_index")
    # user_query = "What is Charge Advice Information in 3GPP?"
    # index_path = "faiss_index/index_hnsw.faiss" 
    # meta_path = "faiss_index/index_hnsw.meta.json"
    # top_5 = get_top_5_specs(user_query, index_path,meta_path)
    # print(json.dumps(top_5, indent=2))
    print(time.time()-start)