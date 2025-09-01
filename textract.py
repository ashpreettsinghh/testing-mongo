def split_list_items_(items):
    """
    Fixed function with proper regex escaping for XML tags
    """
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
            output.append(list_item)
            list_item = "" 
        elif inside_list:
            list_item += p.strip()
        else:
            # Fixed: Handle empty strings properly
            if p.strip():
                output.extend([line for line in p.split('\n') if line.strip()])
    return output
