"""
 This file contains list of all tools that can be used by the agents. 
"""
from typing import Dict, Union
import requests
import json
import re
import os 
from datetime import datetime
from dotenv import load_dotenv
import time
import urllib.parse 
from langchain_community.llms import OpenAI  # Updated import
from index_3gpp_spec import get_top_5_specs
from ftplib import FTP  # Correct import of FTP class
import zipfile
from collections import defaultdict
import traceback  # For detailed error logging

load_dotenv('.env')

def list_directories(ftp_client, base_path):
    """
    Lists directories in the given base_path on the FTP server.
    """
    try:
        ftp_client.cwd(base_path)
        directories = ftp_client.nlst()
        print(f"Available directories in {base_path}: {directories}")
        return directories
    except Exception as e:
        print(f"Error listing directories in {base_path}: {e}")
        traceback.print_exc()
        return []

def get_3gpp_docs(query: str) -> Union[Dict, str]:
    """
    Use this tool to dynamically fetch 3GPP Technical Specifications (TS) and Technical Reports (TR) from 3GPP FTP server based on the query provided.
    The docs will be saved in the temp_rag_space for indexing and further use.
    Each specification will be saved as a separate file.
    Args:
        query (str): The query related to 3GPP specifications
    Returns:
        Union[Dict, str]: Metadata of fetched TS or TR or an error message if the fetch fails.
    """
    BASE_ADDRESS = 'www.3gpp.org'
    BASE_PATH = '/Specs/archive'
    OUTPUT_DIR = 'temp_rag_space'

    index_path = "faiss_index/index_hnsw.faiss" 
    meta_path = "faiss_index/index_hnsw.meta.json"
    
    try:
        top_5 = get_top_5_specs(query, index_path, meta_path)
    except Exception as e:
        print("Error fetching top 5 specs:", e)
        traceback.print_exc()
        return {"status": "error", "message": "Failed to retrieve top 5 specifications."}
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Top 5 Specs:", top_5)
    print("Data Type of top5:", type(top_5))
    
    # Extract the "Spec No" entry from each dictionary in the list
    try:
        spec_no_list = [spec["Spec No"] for spec in top_5]
    except KeyError as e:
        print(f"Missing key in top5 specs: {e}")
        traceback.print_exc()
        return {"status": "error", "message": "Top 5 specifications data is malformed."}
    
    print("Spec No List:", spec_no_list)
    
    for idx, spec_no in enumerate(spec_no_list, start=1):
        print(f"\nProcessing Spec {idx}: {spec_no}")
        try:
            series, doc_number = spec_no.split('.')
            print("Split Spec No:", series, doc_number)
            
            # Initialize FTP client for each spec to ensure fresh connection
            with FTP(BASE_ADDRESS) as ftp_client:
                print("Connecting to FTP server...")
                ftp_client.login()  # Anonymous login
                print("Logged in successfully.")
                
                # Verify if the expected series directory exists
                available_series = list_directories(ftp_client, BASE_PATH)
                expected_series_dir = f"{series}_series"
                
                if expected_series_dir not in available_series:
                    print(f"Expected series directory '{expected_series_dir}' not found. Available directories: {available_series}")
                    continue  # Skip to the next spec
                
                ftp_directory = f"{BASE_PATH}/{series}_series/{series}.{doc_number}"
                print(f"Changing directory to: {ftp_directory}")
                ftp_client.cwd(ftp_directory)
                print("Directory changed successfully.")
                
                filenames = ftp_client.nlst()
                print(f"Filenames in directory: {filenames}")
                filenames.sort()
                print(f"Sorted Filenames: {filenames}")
                
                # Find the latest major version
                latest_major_version = None
                for filename in filenames:
                    if "-" in filename and filename.endswith(".zip"):
                        parts = filename.split('-')
                        if len(parts) < 2:
                            continue
                        major_version = parts[1][0]
                        if not latest_major_version or major_version > latest_major_version.split('-')[1][0]:
                            latest_major_version = filename
                
                print("Latest Major Version Found:", latest_major_version)
                if not latest_major_version:
                    print(f"No valid files found for spec {spec_no}. Skipping...")
                    continue  # Skip to the next spec
                
                # Download the latest major version
                zip_filename = latest_major_version
                output_path = os.path.join(OUTPUT_DIR, zip_filename)
                print(f"Downloading {zip_filename} to {output_path}...")
                
                with open(output_path, "wb") as fp:
                    ftp_client.retrbinary(f"RETR {zip_filename}", fp.write)
                    print(f"{zip_filename} downloaded successfully.")
                
                # Unzip the file using zipfile module
                # unzip_dir = os.path.join(OUTPUT_DIR, zip_filename.replace(".zip", ""))
                # os.makedirs(unzip_dir, exist_ok=True)
                # print(f"Extracting {zip_filename} to {unzip_dir}...")
                
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(OUTPUT_DIR)
                print(f"{zip_filename} extracted successfully.")
                
                # Delete the zip file after extraction
                os.remove(output_path)
                print(f"{zip_filename} removed after extraction.")
                
        except Exception as e:
            print(f"Error processing spec {spec_no}: {e}")
            traceback.print_exc()
            # Continue with the next spec without terminating the entire process
            continue
    
    return {"status": "success", "message": "Specifications fetched and processed successfully."}

# Example usage
if __name__ == '__main__':
    result = get_3gpp_docs("Evolved Packet System (EPS)")
    print("\nResult:", result)
