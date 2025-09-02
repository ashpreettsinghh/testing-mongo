max_words = 200
chunks = {}
table_header_dict = {} 
chunk_header_mapping = {}
list_header_dict = {}

# iterate through each title section
for title_ids, items in enumerate(header_split):
    title_chunks = []
    current_chunk = []
    num_words = 0   
    table_header_dict[title_ids] = {}
    chunk_header_mapping[title_ids] = {}
    list_header_dict[title_ids] = {}
    chunk_counter = 0
    first_header_portion = True
    
    # Split by title prefix to get individual title sections
    for item_ids, item in enumerate(items.split("<<>")):
        lines = sub_header_content_splitta(item)             
        SECTION_HEADER = None 
        TITLES = None
        num_words = 0  
        
        for ids_line, line in enumerate(lines):
            if not line.strip():
                continue
                
            # FIXED: Correct title extraction based on TextLinearizationConfig
            # Your config uses title_prefix="<<>" (no closing >>)
            if "<<>" in line:
                # Extract everything after <<>
                title_match = re.search(r'<<>(.*)', line)
                if title_match:
                    TITLES = title_match.group(1).strip()
                    line = TITLES 
                    if re.sub(r'<[^>]+>', '', "".join(lines)).strip() == TITLES:
                        chunk_header_mapping[title_ids][chunk_counter] = lines
                        chunk_counter += 1
                # No else needed - TITLES will remain None if no match
            
            # FIXED: Correct section header extraction
            # Your config uses section_header_prefix="<header><<header>>"
            if "<header><<header>>" in line:
                section_header_match = re.search(r'<header><<header>>(.*?)</header></header>', line)
                if section_header_match:
                    SECTION_HEADER = section_header_match.group(1).strip()
                    line = SECTION_HEADER    
                    first_header_portion = True
                # No else needed - SECTION_HEADER will remain None if no match
            
            # Calculate word count for current line
            word_count = len(re.findall(r'\w+', line))
            next_num_words = num_words + word_count

            # Process regular text (not tables or lists)
            if "<table>" not in line and "<list>" not in line and "<<list>><list>" not in line:
                # If adding this line would exceed max_words and we have content
                if next_num_words > max_words and current_chunk and "".join(current_chunk).strip() not in [SECTION_HEADER, TITLES]:
                    # Add section header if applicable
                    if SECTION_HEADER and not first_header_portion:
                        current_chunk.insert(0, SECTION_HEADER.strip())
                    
                    title_chunks.append(current_chunk)
                    chunk_header_mapping[title_ids][chunk_counter] = lines
                    current_chunk = []
                    num_words = 0 
                    chunk_counter += 1
                
                current_chunk.append(line)    
                num_words += word_count

            # Process tables
            if "<table>" in line:
                # Get table header which is usually line before table
                line_index = lines.index(line)
                header = ""
                if line_index != 0 and "<table>" not in lines[line_index-1] and "<list>" not in lines[line_index-1]:
                    header = lines[line_index-1].replace("<header>", "").replace("</header>", "").strip()
                
                # Extract table content between <table> and </table>
                table_match = re.search(r'<table>(.*?)</table>', line, re.DOTALL)
                table = table_match.group(1).strip() if table_match else ""
                
                if not table:
                    continue
                    
                # Convert table to DataFrame
                try:
                    df = pd.read_csv(io.StringIO(table), sep=csv_seperator, keep_default_na=False, header=None)
                    if len(df) > 0:
                        df.columns = df.iloc[0]
                        df = df[1:]
                        df.rename(columns=lambda x: '' if str(x).startswith('Unnamed:') else x, inplace=True)
                        
                        curr_chunk = [df.columns.tolist()]
                        words = len(re.findall(r'\w+', str(current_chunk) + " " + str(curr_chunk)))
                        
                        # Process each row of the table
                        for row in df.itertuples(index=False):
                            curr_chunk.append(row)
                            words += len(re.findall(r'\w+', str(row)))
                            
                            if words > max_words:
                                # Store table header info
                                if chunk_counter in table_header_dict[title_ids]:
                                    table_header_dict[title_ids][chunk_counter].extend([header, table])
                                else:
                                    table_header_dict[title_ids][chunk_counter] = [header, table]
                                
                                # Format the chunk as CSV
                                tab_chunk = "\n".join(
                                    [csv_seperator.join(str(x) for x in curr_chunk[0])] + 
                                    [csv_seperator.join(str(x) for x in r) for r in curr_chunk[1:]]
                                )
                                
                                # Add header to table if needed
                                if header:
                                    if current_chunk and current_chunk[-1].strip().lower() == header.strip().lower():
                                        current_chunk.pop(-1)
                                    if SECTION_HEADER and SECTION_HEADER.lower().strip() != header.lower().strip() and not first_header_portion:
                                        current_chunk.insert(0, SECTION_HEADER.strip())
                                    current_chunk.extend([
                                        header.strip() + ':' if not header.strip().endswith(':') else header.strip(),
                                        tab_chunk
                                    ])
                                else:
                                    if SECTION_HEADER and not first_header_portion:
                                        current_chunk.insert(0, SECTION_HEADER.strip())
                                    current_chunk.append(tab_chunk)
                                
                                title_chunks.append(current_chunk)
                                chunk_header_mapping[title_ids][chunk_counter] = lines
                                chunk_counter += 1
                                current_chunk = []
                                num_words = 0
                                curr_chunk = [df.columns.tolist()]
                                words = len(re.findall(r'\w+', str(df.columns.tolist())))
                
                        # Handle remaining table rows
                        if len(curr_chunk) > 1:  # More than just headers
                            tab_chunk = "\n".join(
                                [csv_seperator.join(str(x) for x in curr_chunk[0])] + 
                                [csv_seperator.join(str(x) for x in r) for r in curr_chunk[1:]]
                            )
                            
                            if chunk_counter in table_header_dict[title_ids]:
                                table_header_dict[title_ids][chunk_counter].extend([header, table])
                            else:
                                table_header_dict[title_ids][chunk_counter] = [header, table]
                            
                            if header:
                                if current_chunk and current_chunk[-1].strip().lower() == header.strip().lower():
                                    current_chunk.pop(-1)
                                if SECTION_HEADER and SECTION_HEADER.lower().strip() != header.lower().strip() and not first_header_portion:
                                    current_chunk.insert(0, SECTION_HEADER.strip())
                                current_chunk.extend([
                                    header.strip() + ':' if not header.strip().endswith(':') else header.strip(),
                                    tab_chunk
                                ])
                            else:
                                if SECTION_HEADER and not first_header_portion:
                                    current_chunk.insert(0, SECTION_HEADER.strip())
                                current_chunk.append(tab_chunk)
                            
                            title_chunks.append(current_chunk)
                            chunk_header_mapping[title_ids][chunk_counter] = lines
                            chunk_counter += 1
                            current_chunk = []
                            num_words = 0
                except Exception as e:
                    print(f"Error processing table: {e}")
                    continue

            # Process lists
            if "<<list>><list>" in line:
                # Get list header
                line_index = lines.index(line)
                header = ""
                if line_index != 0 and "<table>" not in lines[line_index-1] and "<list>" not in lines[line_index-1]:
                    header = lines[line_index-1].replace("<header>", "").replace("</header>", "").strip()
                
                # Extract list content
                list_match = re.search(r'<<list>><list>(.*?)(?:</list><</list>>|$)', line, re.DOTALL)
                list_content = list_match.group(1).strip() if list_match else ""
                list_lines = [l.strip() for l in list_content.split("\n") if l.strip()]
                
                if not list_lines:
                    continue
                
                curr_chunk = []
                words = len(re.findall(r'\w+', str(current_chunk)))
                
                # Process each list item
                for list_item in list_lines:
                    curr_chunk.append(list_item)
                    words += len(re.findall(r'\w+', list_item))
                    
                    if words >= max_words:
                        # Store list header info
                        if chunk_counter in list_header_dict[title_ids]:
                            list_header_dict[title_ids][chunk_counter].extend([header, list_content])
                        else:
                            list_header_dict[title_ids][chunk_counter] = [header, list_content]
                        
                        list_chunk = "\n".join(curr_chunk)
                        
                        # Add header to list if needed
                        if header:
                            if current_chunk and current_chunk[-1].strip().lower() == header.strip().lower():
                                current_chunk.pop(-1)
                            if SECTION_HEADER and SECTION_HEADER.lower().strip() != header.lower().strip() and not first_header_portion:
                                current_chunk.insert(0, SECTION_HEADER.strip())
                            current_chunk.extend([
                                header.strip() + ':' if not header.strip().endswith(':') else header.strip(),
                                list_chunk
                            ])
                        else:
                            if SECTION_HEADER and not first_header_portion:
                                current_chunk.insert(0, SECTION_HEADER.strip())
                            current_chunk.append(list_chunk)
                        
                        title_chunks.append(current_chunk)
                        chunk_header_mapping[title_ids][chunk_counter] = lines
                        chunk_counter += 1
                        current_chunk = []
                        num_words = 0
                        curr_chunk = []
                
                # Handle remaining list items
                if curr_chunk:
                    list_chunk = "\n".join(curr_chunk)
                    
                    if chunk_counter in list_header_dict[title_ids]:
                        list_header_dict[title_ids][chunk_counter].extend([header, list_content])
                    else:
                        list_header_dict[title_ids][chunk_counter] = [header, list_content]
                    
                    if header:
                        if current_chunk and current_chunk[-1].strip().lower() == header.strip().lower():
                            current_chunk.pop(-1)
                        if SECTION_HEADER and SECTION_HEADER.lower().strip() != header.lower().strip() and not first_header_portion:
                            current_chunk.insert(0, SECTION_HEADER.strip())
                        current_chunk.extend([
                            header.strip() + ':' if not header.strip().endswith(':') else header.strip(),
                            list_chunk
                        ])
                    else:
                        if SECTION_HEADER and not first_header_portion:
                            current_chunk.insert(0, SECTION_HEADER.strip())
                        current_chunk.append(list_chunk)
                    
                    title_chunks.append(current_chunk)
                    chunk_header_mapping[title_ids][chunk_counter] = lines
                    chunk_counter += 1
                    current_chunk = []
                    num_words = 0
        
        # Handle remaining content after processing all lines
        if current_chunk and "".join(current_chunk).strip() not in [SECTION_HEADER, TITLES]:
            if SECTION_HEADER and not first_header_portion:
                current_chunk.insert(0, SECTION_HEADER.strip())
            title_chunks.append(current_chunk)
            chunk_header_mapping[title_ids][chunk_counter] = lines
            current_chunk = []
            chunk_counter += 1
    
    # Handle any remaining content after processing all items
    if current_chunk:
        title_chunks.append(current_chunk) 
        chunk_header_mapping[title_ids][chunk_counter] = lines
    
    chunks[title_ids] = title_chunks
