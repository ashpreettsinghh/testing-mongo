def merge_multiline_tags(lines):
    """
    Merge lines where opening and closing XML-like tags (e.g. <title> ... </title>)
    got split across multiple list items. Returns a new list of lines.
    Works generically for tags you use: <title>, <header>, <table>, <list>, etc.
    """
    merged = []
    buffer = None
    open_tag = None
    # pattern to capture opening tag like <title> or <<table>><table> (we normalize common forms)
    open_re = re.compile(r'(<(?:<)?\w+>(?:<\w+>)?)', flags=re.IGNORECASE)
    close_re = re.compile(r'(</(?:\w+)>|</\w+><</\w+>>|</table>|</list>)', flags=re.IGNORECASE)

    for ln in lines:
        ln_stripped = ln.strip()
        if buffer is None:
            # if this line contains an opening tag but not the matching close, start buffer
            # detect typical opening tags you use (covers <title>, <header>, <<list>><list>, <<table>><table>)
            if re.search(r'<(?:title|header|table|list|<titles>|<headers>|<<table>><table>)', ln_stripped, flags=re.IGNORECASE) and not re.search(r'</(?:title|header|table|list)>', ln_stripped, flags=re.IGNORECASE):
                buffer = ln_stripped
                # record a rough open tag name for detection of close
                m = re.search(r'<\s*([a-zA-Z0-9_]+)', ln_stripped)
                open_tag = m.group(1).lower() if m else None
            else:
                merged.append(ln_stripped)
        else:
            # we are inside a tag; append this line
            buffer += " " + ln_stripped
            # detect closing for the open_tag (best-effort)
            if (open_tag and re.search(rf'</\s*{re.escape(open_tag)}\s*>', ln_stripped, flags=re.IGNORECASE)) \
               or re.search(r'</(?:title|header|table|list)>', ln_stripped, flags=re.IGNORECASE):
                merged.append(buffer)
                buffer = None
                open_tag = None

    # if buffer left open, push it (avoid losing content)
    if buffer is not None:
        merged.append(buffer)

    return merged
