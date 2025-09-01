def merge_multiline_tags(lines):
    merged_lines = []
    buffer = ""
    inside_tag = False

    for line in lines:
        if re.search(r'<[^/]+?>', line):  # opening tag like <TITLE>
            inside_tag = True
            buffer = line.strip()
        elif re.search(r'</[^>]+?>', line):  # closing tag like </TITLE>
            buffer += " " + line.strip()
            merged_lines.append(buffer)
            buffer = ""
            inside_tag = False
        elif inside_tag:
            buffer += " " + line.strip()
        else:
            merged_lines.append(line.strip())

    return merged_lines
