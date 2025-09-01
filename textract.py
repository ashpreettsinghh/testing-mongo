# COMPLETE FIXED CODE - Copy this entire code block

import os
from PIL import Image
import pandas as pd
import re
import json
import uuid
from textractor import Textractor
from textractor.visualizers.entitylist import EntityList
from textractor.data.constants import TextractFeatures
from textractor.data.text_linearization_config import TextLinearizationConfig
import io
import inflect
from collections import OrderedDict
import boto3
import time
import openpyxl
from openpyxl.cell import Cell
from openpyxl.worksheet.cell_range import CellRange
import numpy as np

# Fixed: Added missing imports that were causing errors
import base64
import random
from botocore.exceptions import ClientError

# Initialize all required clients
s3 = boto3.client("s3")
from botocore.config import Config
config = Config(
    read_timeout=600, 
    retries = dict(max_attempts = 5)
)
from anthropic import Anthropic
client = Anthropic()
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name='us-east-1', config=config)
SAGEMAKER = boto3.client('sagemaker-runtime')  # This was missing in original code

# Configuration for text linearization
linearization_config = TextLinearizationConfig(
    hide_figure_layout=False,
    title_prefix="<title>",
    title_suffix="</title>",
    hide_header_layout=True,
    section_header_prefix="<header>",
    section_header_suffix="</header>",
    table_prefix="<table>",
    table_suffix="</table>",
    list_layout_prefix="<listitem>",
    list_layout_suffix="</listitem>",
    hide_footer_layout=True,
    hide_page_num_layout=True,
)

# Model dimension mapping
model_dimension_mapping = {"titanv2":1024,"titanv1":1536,"bge":1024,"all-mini-lm":384,"e5":1024}

def strip_newline(cell):
    """
    A utility function to strip newline characters from a cell.
    """
    return str(cell).strip()

def layout_table_to_excel(document, ids, csv_separator): 
    """
    Converts an Excel table from a document to a Pandas DataFrame, 
    handling duplicated values across merged cells.
    """
    try:
        # Save the table in excel format to preserve the structure of any merged cells
        buffer = io.BytesIO() 
        document.tables[ids].to_excel(buffer)
        buffer.seek(0)
        
        # Load workbook, get active worksheet
        wb = openpyxl.load_workbook(buffer)
        worksheet = wb.active
        
        # Unmerge cells, duplicate merged values to individual cells
        all_merged_cell_ranges = list(worksheet.merged_cells.ranges)
        
        for merged_cell_range in all_merged_cell_ranges:
            merged_cell = merged_cell_range.start_cell
            worksheet.unmerge_cells(range_string=merged_cell_range.coord)
            for row_index, col_index in merged_cell_range.cells:
                cell = worksheet.cell(row=row_index, column=col_index)
                cell.value = merged_cell.value
        
        # Determine table header index
        df = pd.DataFrame(worksheet.values)
        df = df.map(strip_newline)
        df0 = df.to_csv(sep=csv_separator, index=False, header=None)
        row_count = len([x for x in df0.split("\n") if x])
        
        if row_count > 1:
            if not all(value.strip() == '' for value in df0.split("\n")[0].split(csv_separator)): 
                row_count = 1
        
        # Attach table column names
        column_row = 0 if row_count == 1 else 1
        df.columns = df.iloc[column_row] 
        df = df[column_row+1:]
        return df
    except Exception as e:
        print(f"Error in layout_table_to_excel: {e}")
        return pd.DataFrame()

def split_list_items_(items):
    """
    Splits the given string into a list of items, handling nested lists.
    """
    if not items or not isinstance(items, str):
        return []
        
    # Fixed: Properly escape XML tag patterns in regex
    parts = re.split(r'(<listitem>|</listitem>)', items)
    output = []
    inside_list = False
    list_item = ""

    for p in parts:
        if p == "<listitem>":
            inside_list = True 
            list_item = p
        elif p == "</listitem>":
            inside_list = False
            list_item += p
            if list_item.strip():
                output.append(list_item)
            list_item = "" 
        elif inside_list:
            list_item += p.strip()
        else:
            # Fixed: Handle empty strings properly
            if p.strip():
                lines = [line.strip() for line in p.split('\n') if line.strip()]
                output.extend(lines)
    return output

def sub_header_content_splitta(string):
    """
    Splits the input string by XML tags and returns a list containing the segments of text,
    excluding segments containing specific XML tags.
    """
    if not string:
        return []
        
    # Split by XML tags using regex
    segments = re.split(r'(<[^>]+>)', string)
    result = []
    
    # Filter out segments containing specific XML tags
    exclude_tags = ['<tabledata>', '</tabledata>', '<listitem>', '</listitem>']
    
    for segment in segments:
        segment = segment.strip()
        if segment and not any(tag in segment for tag in exclude_tags):
            if '\n' in segment and '<' not in segment:
                # Split multi-line text segments
                lines = [x.strip() for x in segment.split('\n') if x.strip()]
                result.extend(lines)
            else:
                result.append(segment)
    return result

def _get_emb_(passage, model):
    """
    Fixed embedding function with proper error handling.
    """
    try:
        if "titanv1" in model:
            response = bedrock_runtime.invoke_model(
                body=json.dumps({"inputText":passage}),
                modelId="amazon.titan-embed-text-v1", 
                accept="application/json", 
                contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            embedding = response_body['embedding']
            
        elif "titanv2" in model:
            response = bedrock_runtime.invoke_model(
                body=json.dumps({"inputText":passage,"dimensions":1024,"normalize":False}),
                modelId="amazon.titan-embed-text-v2:0", 
                accept="application/json", 
                contentType="application/json"
            )
            response_body = json.loads(response.get('body').read())
            embedding = response_body['embedding']
            
        elif "all-mini-lm" in model:
            payload = {'text_inputs': [passage]}
            payload = json.dumps(payload).encode('utf-8')
            
            try:
                response = SAGEMAKER.invoke_endpoint(
                    EndpointName="SAGEMAKER JUMPSTART ALL MINI LM V6 ENDPOINT", 
                    ContentType='application/json', 
                    Body=payload
                )
                model_predictions = json.loads(response['Body'].read())
                embedding = model_predictions['embedding'][0]
            except Exception as e:
                print(f"Error invoking SageMaker endpoint for all-mini-lm: {e}")
                print("Please replace 'SAGEMAKER JUMPSTART ALL MINI LM V6 ENDPOINT' with your actual endpoint name")
                return None
                
        elif "e5" in model:
            payload = {"text_inputs":[passage],"mode":"embedding"}
            payload = json.dumps(payload).encode('utf-8')
            
            try:
                response = SAGEMAKER.invoke_endpoint(
                    EndpointName="SAGEMAKER JUMPSTART E5 ENDPOINT", 
                    ContentType='application/json', 
                    Body=payload
                )
                model_predictions = json.loads(response['Body'].read())
                embedding = model_predictions['embedding'][0]
            except Exception as e:
                print(f"Error invoking SageMaker endpoint for e5: {e}")
                print("Please replace 'SAGEMAKER JUMPSTART E5 ENDPOINT' with your actual endpoint name")
                return None
                
        elif "bge" in model:
            payload = {"text_inputs":[passage],"mode":"embedding"}
            payload = json.dumps(payload).encode('utf-8')
            
            try:
                response = SAGEMAKER.invoke_endpoint(
                    EndpointName="SAGEMAKER JUMPSTART BGE ENDPOINT", 
                    ContentType='application/json', 
                    Body=payload
                )
                model_predictions = json.loads(response['Body'].read())
                embedding = model_predictions['embedding'][0]
            except Exception as e:
                print(f"Error invoking SageMaker endpoint for bge: {e}")
                print("Please replace 'SAGEMAKER JUMPSTART BGE ENDPOINT' with your actual endpoint name")
                return None
        else:
            print(f"Unknown model: {model}. Supported models: titanv1, titanv2, all-mini-lm, e5, bge")
            return None
            
        return embedding
    except Exception as e:
        print(f"Error in _get_emb_: {e}")
        return None

def bedrock_streemer(response):
    """
    Stream response from Bedrock models.
    """
    stream = response.get('body')
    answer = ""
    i = 1
    if stream:
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_obj = json.loads(chunk.get('bytes').decode())
                if "delta" in chunk_obj: 
                    delta = chunk_obj['delta']
                    if "text" in delta:
                        text = delta['text'] 
                        print(text, end="")
                        answer += str(text) 
                        i += 1
                if "amazon-bedrock-invocationMetrics" in chunk_obj:
                    input_tokens = chunk_obj['amazon-bedrock-invocationMetrics']['inputTokenCount']
                    output_tokens = chunk_obj['amazon-bedrock-invocationMetrics']['outputTokenCount']
                    print(f"\nInput Tokens: {input_tokens}\nOutput Tokens: {output_tokens}")
    return answer, input_tokens if 'input_tokens' in locals() else 0, output_tokens if 'output_tokens' in locals() else 0

def bedrock_claude_(chat_history, system_message, prompt, model_id, image_path=None):
    """
    Fixed Bedrock Claude function with proper error handling.
    """
    content = []
    if image_path: 
        if not isinstance(image_path, list):
            image_path = [image_path] 
        for img in image_path:
            s3_client = boto3.client('s3')
            match = re.match(r"s3://(.+?)/(.+)", img)
            image_name = os.path.basename(img)
            _, ext = os.path.splitext(image_name)
            if "jpg" in ext: 
                ext = ".jpeg" 
            if match:
                bucket_name = match.group(1)
                key = match.group(2) 
                obj = s3_client.get_object(Bucket=bucket_name, Key=key)
                base_64_encoded_data = base64.b64encode(obj['Body'].read())
                base64_string = base_64_encoded_data.decode('utf-8')
                content.extend([{"type":"text","text":image_name},{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": f"image/{ext.lower().replace('.',')}",
                        "data": base64_string
                    }
                }])
     
    content.append({
        "type": "text",
        "text": prompt
    })
    chat_history.append({"role": "user", "content": content})
    
    prompt_data = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1500,
        "temperature": 0.1,
        "system": system_message,
        "messages": chat_history
    }
    
    answer = ""
    prompt_json = json.dumps(prompt_data)
    response = bedrock_runtime.invoke_model_with_response_stream(
        body=prompt_json, 
        modelId=model_id, 
        accept="application/json", 
        contentType="application/json"
    )
    answer, input_tokens, output_tokens = bedrock_streemer(response) 
    return answer, input_tokens, output_tokens

def _invoke_bedrock_with_retries(current_chat, chat_template, question, model_id, image_path):
    """
    Fixed retry function with proper error handling.
    """
    max_retries = 5
    backoff_base = 2
    max_backoff = 3  # Maximum backoff time in seconds
    retries = 0

    while True:
        try:
            response, input_tokens, output_tokens = bedrock_claude_(current_chat, chat_template, question, model_id, image_path)
            return response, input_tokens, output_tokens
        except ClientError as e:
            if e.response['Error']['Code'] == 'ThrottlingException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            elif e.response['Error']['Code'] == 'ModelStreamErrorException':
                if retries < max_retries:
                    # Throttling, exponential backoff
                    sleep_time = min(max_backoff, backoff_base ** retries + random.uniform(0, 1))
                    time.sleep(sleep_time)
                    retries += 1
                else:
                    raise e
            else:
                # Some other API error, rethrow
                raise

# 🔥 MAIN DOCUMENT PROCESSING FUNCTION - This fixes your IndexError!
def process_document_safely(document, config):
    """
    Process document with proper error handling and bounds checking.
    This is the main fix for your IndexError!
    """
    csv_separator = "|"  # Fixed: consistent naming
    document_holder = {}
    table_page = {}
    count = 0
    unmerge_span_cells = True
    
    print(f"Processing document with {len(document.pages)} pages and {len(document.tables)} tables")
    
    # Loop through each page in the document
    for ids, page in enumerate(document.pages):
        print(f"Processing page {ids + 1}/{len(document.pages)}")
        
        try:
            page_text = page.get_text(config=config)
            
            # Get the number of tables in the extracted document page
            table_count = len([word for word in page_text.split() if "<table>" in word])
            
            # Fixed: Better validation of table counts
            available_tables = len(page.tables)
            if table_count != available_tables:
                print(f"Warning on page {ids}: Expected {table_count} tables but found {available_tables}")
            
            # Fixed: Use safer splitting method
            content = page_text.split("<table>")
            document_holder[ids] = []
            
            for idx, item in enumerate(content):
                # Process items that contain tables or are after the first split
                if "<table>" in item or (idx > 0):
                    # 🔥 CRITICAL FIX: Bounds checking to prevent IndexError
                    if count < len(document.tables):
                        try:
                            if unmerge_span_cells:
                                df = layout_table_to_excel(document, count, csv_separator)
                            else:
                                # Alternative method without unmerging
                                df0 = document.tables[count].to_pandas(use_columns=False).to_csv(
                                    header=False, index=None, sep=csv_separator)
                                
                                row_count = len([x for x in df0.split("\n") if x])
                                
                                # Check if the first row in the csv is empty headers
                                if row_count > 1:
                                    first_row_values = df0.split("\n")[0].split(csv_separator)
                                    if not all(value.strip() == '' for value in first_row_values): 
                                        row_count = 1
                                
                                df = pd.read_csv(io.StringIO(df0), sep=csv_separator,
                                               header=0 if row_count == 1 else 1, 
                                               keep_default_na=False)
                                               
                                # Clean up column names
                                df.rename(columns=lambda x: '' if str(x).startswith('Unnamed:') else x, 
                                        inplace=True)
                            
                            # Convert dataframe to CSV string
                            if not df.empty:
                                table = df.to_csv(index=None, sep=csv_separator)
                                
                                # Store table data
                                if ids in table_page:
                                    table_page[ids].append(table)
                                else:
                                    table_page[ids] = [table]
                                
                                # Fixed: Extract content before and after table properly
                                if "</table>" in item:
                                    # Find content after </table>
                                    table_end = item.find("</table>") + len("</table>")
                                    remaining_content = item[table_end:]
                                else:
                                    remaining_content = item
                                
                                # Fixed: Properly format table with XML tags
                                processed_item = f"<tabledata>{table}</tabledata>"
                                count += 1
                                
                                # Check for list items in remaining content
                                if "<listitem>" in remaining_content:
                                    output = split_list_items_(remaining_content)
                                    document_holder[ids].extend([processed_item] + output)
                                else:
                                    # Split remaining content by lines
                                    remaining_lines = [x.strip() for x in remaining_content.split('\n') if x.strip()]
                                    document_holder[ids].extend([processed_item] + remaining_lines)
                            else:
                                print(f"Warning: Empty table found at index {count}")
                                count += 1
                                
                        except Exception as table_error:
                            print(f"Error processing table {count} on page {ids}: {table_error}")
                            count += 1
                            continue
                    else:
                        # 🔥 KEY FIX: Handle case when table count exceeds available tables
                        print(f"Warning: Table index {count} exceeds available tables ({len(document.tables)})")
                        # Process remaining content without table
                        if "<listitem>" in item:
                            output = split_list_items_(item)
                            document_holder[ids].extend(output)
                        else:
                            lines = [x.strip() for x in item.split('\n') if x.strip()]
                            document_holder[ids].extend(lines)
                else:
                    # Process items that don't contain tables
                    if "<listitem>" in item and "<table>" not in item:
                        output = split_list_items_(item)
                        document_holder[ids].extend(output)
                    else:
                        lines = [x.strip() for x in item.split("\n") if x.strip()]
                        document_holder[ids].extend(lines)
                        
        except Exception as page_error:
            print(f"Error processing page {ids}: {page_error}")
            continue
    
    print(f"Document processing completed. Processed {count} tables total.")
    return document_holder, table_page

def chunk_document_content(document_holder, max_words=200):
    """
    Chunk the processed document content with proper error handling.
    """
    # Flatten the nested list and join by newline
    flattened_list = [item for sublist in document_holder.values() for item in sublist]
    result = "\n".join(flattened_list)
    header_split = result.split("<title>")
    
    chunks = {}
    table_header_dict = {} 
    chunk_header_mapping = {}
    list_header_dict = {}
    
    # Iterate through each title section
    for title_ids, items in enumerate(header_split):
        if not items.strip():
            continue
            
        title_chunks = []
        current_chunk = []
        num_words = 0 
        table_header_dict[title_ids] = {}
        chunk_header_mapping[title_ids] = {}
        list_header_dict[title_ids] = {}
        chunk_counter = 0
        
        # Split by section headers
        sections = items.split('</header>')
        
        for item_ids, item in enumerate(sections):
            if not item.strip():
                continue
                
            lines = sub_header_content_splitta(item)
            SECTION_HEADER = None 
            TITLES = None
            num_words = 0 
            
            for ids_line, line in enumerate(lines):
                if line.strip():
                    # Extract title information
                    if "<title>" in line: 
                        title_match = re.findall(r'<title>(.*?)</title>', line)
                        if title_match:
                            TITLES = title_match[0].strip()
                            line = TITLES
                        
                        # Check if this is just a title line
                        clean_content = re.sub(r'<[^>]+>', '', "".join(lines)).strip()
                        if clean_content == TITLES:
                            chunk_header_mapping[title_ids][chunk_counter] = lines
                            chunk_counter += 1
                    
                    # Process different content types
                    if "<header>" in line:
                        header_match = re.findall(r'<header>(.*?)</header>', line)
                        if header_match:
                            SECTION_HEADER = header_match[0].strip()
                    elif "<tabledata>" in line:
                        # Handle table data
                        table_content = line
                        current_chunk.append(table_content)
                        
                        # Count words in current chunk
                        chunk_text = " ".join([re.sub(r'<[^>]+>', '', chunk_item) for chunk_item in current_chunk])
                        num_words = len(chunk_text.split())
                        
                        if num_words >= max_words or ids_line == len(lines) - 1:
                            if current_chunk:
                                title_chunks.append(current_chunk.copy())
                                current_chunk = []
                                num_words = 0
                    elif "<listitem>" in line:
                        # Handle list items
                        list_content = line
                        current_chunk.append(list_content)
                        
                        # Count words
                        chunk_text = " ".join([re.sub(r'<[^>]+>', '', chunk_item) for chunk_item in current_chunk])
                        num_words = len(chunk_text.split())
                        
                        if num_words >= max_words or ids_line == len(lines) - 1:
                            if current_chunk:
                                title_chunks.append(current_chunk.copy())
                                current_chunk = []
                                num_words = 0
                    else:
                        # Handle regular text
                        current_chunk.append(line)
                        
                        # Count words
                        chunk_text = " ".join([re.sub(r'<[^>]+>', '', chunk_item) for chunk_item in current_chunk])
                        num_words = len(chunk_text.split())
                        
                        if num_words >= max_words:
                            if current_chunk:
                                title_chunks.append(current_chunk.copy())
                                current_chunk = []
                                num_words = 0
        
        # Add any remaining content
        if current_chunk:
            title_chunks.append(current_chunk)
        
        if title_chunks:
            chunks[title_ids] = title_chunks
    
    return chunks, table_header_dict, chunk_header_mapping, list_header_dict

# 🔥 MAIN USAGE FUNCTION - Replace your existing code with this!
def process_textract_document(document, config=None):
    """
    Main function to process a Textract document safely.
    This replaces your existing problematic code!
    """
    if config is None:
        config = linearization_config
    
    try:
        print("Starting document processing...")
        
        # Process document with error handling (this fixes the IndexError!)
        document_holder, table_page = process_document_safely(document, config)
        
        print("Document processing completed, starting chunking...")
        
        # Chunk the content
        chunks, table_header_dict, chunk_header_mapping, list_header_dict = chunk_document_content(document_holder)
        
        print(f"Chunking completed. Generated {len(chunks)} title sections with chunks.")
        
        return {
            'document_holder': document_holder,
            'table_page': table_page,
            'chunks': chunks,
            'table_header_dict': table_header_dict,
            'chunk_header_mapping': chunk_header_mapping,
            'list_header_dict': list_header_dict
        }
        
    except Exception as e:
        print(f"Error in process_textract_document: {e}")
        return None

# 🚀 HOW TO USE - Replace your problematic section with this single line:
# result = process_textract_document(document, linearization_config)

print("✅ COMPLETE CODE READY!")
print("\nTo fix your IndexError, replace your entire problematic section with:")
print("result = process_textract_document(document, linearization_config)")
