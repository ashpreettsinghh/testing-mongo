from trp import Document

def extract_keys_with_signatures(textract_response_bytes):
    """
    Extract form key texts that have associated signature blocks.
    Uses TRP v1 to traverse key-value relationships.
    """
    doc = Document(textract_response_bytes)
    results = []

    for page in doc.pages:
        # Iterate over all key-value sets on the page
        for kv in page.form.fields:
            key_block = kv.key
            if not key_block:
                continue

            # Check if any child of the key block is a signature
            signature_child = None
            for rel in key_block.relationships or []:
                if rel.type == "CHILD":
                    for child_id in rel.ids:
                        # Find the child block by ID
                        child_block = page.block_map.get(child_id)
                        if child_block and child_block.block_type == "SIGNATURE":
                            signature_child = child_block
                            break
                if signature_child:
                    break

            if signature_child:
                results.append({
                    "key_text": key_block.text,
                    "signature_id": signature_child.id,
                    "signature_confidence": signature_child.confidence,
                    "page": page.page_number,
                    "signature_bbox": signature_child.geometry.bounding_box,
                })

    return results

# Usage:
# response_bytes = your_textract_client.analyze_document(...).to_json().encode()
# key_signatures = extract_keys_with_signatures(response_bytes)
# for entry in key_signatures:
#     print(entry)
