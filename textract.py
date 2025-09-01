segment = [x if (x.endswith('\n') and ('<' in x and '>' in x)) else x.strip() for x in segment.split('\n') if x.strip() or x.endswith('\n')]
