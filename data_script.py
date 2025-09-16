doc = trp.Document(response)
form_fields = []

# Build a map of all blocks for relationship traversal
blocks_map = {}
for page in doc.pages:
    for block in page._blocks:
        blocks_map[block.id] = block

def find_parent_key_value(field_key_block, blocks_map):
    """Find parent key-value pair if available"""
    current_id = field_key_block.id
    
    for block_id, block in blocks_map.items():
        if hasattr(block, 'relationships') and block.relationships:
            for relationship in block.relationships:
                if (relationship.type == 'CHILD' and 
                    hasattr(relationship, 'ids') and 
                    current_id in relationship.ids):
                    
                    if (hasattr(block, 'block_type') and 
                        block.block_type == 'KEY_VALUE_SET' and
                        hasattr(block, 'entity_types') and
                        block.entity_types and
                        'KEY' in block.entity_types):
                        
                        parent_key_text = getattr(block, 'text', None)
                        parent_value_text = None
                        
                        if hasattr(block, 'relationships') and block.relationships:
                            for rel in block.relationships:
                                if rel.type == 'VALUE' and hasattr(rel, 'ids'):
                                    for value_id in rel.ids:
                                        if value_id in blocks_map:
                                            value_block = blocks_map[value_id]
                                            parent_value_text = getattr(value_block, 'text', None)
                                            break
                        
                        return parent_key_text, parent_value_text
    
    return None, None

for page in doc.pages:
    for field in page.form.fields:
        if field.key and field.value:
            key = field.key.text.strip() if field.key.text else None
            value = field.value.text.strip() if field.value.text else None
            
            # Find parent key/value
            parent_key, parent_value = find_parent_key_value(field.key, blocks_map)
            
            field_data = {
                "parentKey": parent_key,
                "parentValue": parent_value,
                "keyId": field.key.id,
                "key": key,
                "value": value,
                "bbox": field.key.geometry.boundingBox.__dict__ if field.key.geometry else None
            }
            form_fields.append(field_data)

if form_fields:
    inferencing_output = OCRInferencingModel(
        file_name=f"{ocr_input.file_name}_page{i}",
        tables=None, 
        form_results=form_fields,
        response=bytes_io
    )
