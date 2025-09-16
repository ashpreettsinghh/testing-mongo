from trp import Document

def sort_fields_top_left(doc):
    for page in doc.pages:
        sorted_fields = []
        for field in page.form.fields:
            sorted_fields.append(field)

        sorted_fields.sort(key=lambda block: block.geometry.boundingBox.top)

        sorted_content = []
        current_line = []
        line_tolerance = 0.01

        for block in sorted_fields:
            if not current_line:
                current_line.append(block)
            else:
                last_block_in_line = current_line[-1]
                # Check if the block is on the same line
                if abs(block.geometry.boundingBox.top - last_block_in_line.geometry.boundingBox.top) < line_tolerance:
                    current_line.append(block)
                else:
                    # Sort current line by left property
                    current_line.sort(key=lambda b: b.geometry.boundingBox.left)
                    sorted_content.extend(current_line)
                    current_line = [block]
        # Add the last line
        if current_line:
            current_line.sort(key=lambda b: b.geometry.boundingBox.left)
            sorted_content.extend(current_line)

        # Replace the page's form.fields with the sorted fields (if you want in-place update)
        page.form.fields = sorted_content  # This is safe if you just want new field order

    return doc  # Output remains a trp.Document object

# Usage:
# sorted_doc = sort_fields_top_left(doc)

