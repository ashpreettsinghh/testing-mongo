def sub_header_content_splitta(string):
    pattern = re.compile(r'<<[^>]+>>')
    segments = re.split(pattern, string)
    result = []
    
    i = 0
    while i < len(segments):
        segment = segments[i]
        if not segment.strip():
            i += 1
            continue
            
        # Check if this segment starts a title/header tag but doesn't close it
        if (("<title>" in segment or "<header>" in segment) and 
            ("</title>" not in segment or "</header>" not in segment)):
            # Look ahead to find the closing tag
            combined = segment
            j = i + 1
            while j < len(segments):
                combined += segments[j]
                if "</title>" in segments[j] or "</header>" in segments[j]:
                    break
                j += 1
            
            # Process the combined segment
            if "<header>" in combined or "<list>" in combined or "<table>" in combined:
                result.append(combined.strip())
            else:
                result.extend([x.strip() for x in combined.split('\n') if x.strip()])
            
            i = j + 1  # Skip the segments we've already processed
        else:
            # Normal processing
            if "<header>" in segment or "<list>" in segment or "<table>" in segment:
                result.append(segment.strip())
            else:
                result.extend([x.strip() for x in segment.split('\n') if x.strip()])
            i += 1
    
    return result
