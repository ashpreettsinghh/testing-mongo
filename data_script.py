
def sort_form_fields_by_position(form_fields, line_tolerance=0.01):
    """Sort form fields by position - top to bottom, left to right"""
    if not form_fields:
        return []
    
    # First sort all fields by top position
    sorted_fields = sorted(form_fields, 
                          key=lambda field: field["bbox"]["Top"] if field["bbox"] else float('inf'))
    
    sorted_result = []
    current_line = []
    
    for field in sorted_fields:
        if not field["bbox"]:  # Skip fields without bbox
            sorted_result.append(field)
            continue
            
        if not current_line:
            # First field in line
            current_line.append(field)
        else:
            last_field = current_line[-1]
            
            # Check if current field is on same line as last field
            if abs(field["bbox"]["Top"] - last_field["bbox"]["Top"]) <= line_tolerance:
                current_line.append(field)
            else:
                # New line detected - sort current line by Left position and add to result
                current_line.sort(key=lambda f: f["bbox"]["Left"] if f["bbox"] else 0)
                sorted_result.extend(current_line)
                
                # Start new line
                current_line = [field]
    
    # Sort and add the last line
    if current_line:
        current_line.sort(key=lambda f: f["bbox"]["Left"] if f["bbox"] else 0)
        sorted_result.extend(current_line)
    
    return sorted_result
