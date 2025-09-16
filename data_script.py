doc = trp.Document(response)
form_fields = []

# Build a map of all blocks for relationship traversal
blocks_map = {}
for page in doc.pages:
    # Access blocks correctly for TRP v1
    if hasattr(page, '_blocks'):
        for block in page._blocks:
            if hasattr(block, 'id'):
                blocks_map[block.id] = block

def find_parent_key_value(field_key_block, blocks_map):
    """Find parent key-value pair if available"""
    if not hasattr(field_key_block, 'id'):
        return None, None
        
    current_id = field_key_block.id
    
    for block_id, block in blocks_map.items():
        # Check if block has relationships
        if not (hasattr(block, 'relationships') and block.relationships):
            continue
            
        for relationship in block.relationships:
            # Check if this relationship contains our current block as child
            if not (hasattr(relationship, 'type') and relationship.type == 'CHILD'):
                continue
                
            if not (hasattr(relationship, 'ids') and current_id in relationship.ids):
                continue
            
            # Check if parent block is a KEY_VALUE_SET with KEY entity type
            if not (hasattr(block, 'block_type') and block.block_type == 'KEY_VALUE_SET'):
                continue
                
            if not (hasattr(block, 'entity_types') and block.entity_types and 'KEY' in block.entity_types):
                continue
            
            # Get parent key text
            parent_key_text = getattr(block, 'text', None)
            parent_value_text = None
            
            # Find corresponding value for this parent key
            if hasattr(block, 'relationships') and block.relationships:
                for rel in block.relationships:
                    if (hasattr(rel, 'type') and rel.type == 'VALUE' and 
                        hasattr(rel, 'ids')):
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
            
            # Find parent key/value - only if field.key has proper attributes
            parent_key, parent_value = find_parent_key_value(field.key, blocks_map)
            
            # Handle bounding box safely
            bbox = None
            if (hasattr(field.key, 'geometry') and 
                field.key.geometry and 
                hasattr(field.key.geometry, 'boundingBox')):
                try:
                    bbox = field.key.geometry.boundingBox.__dict__
                except:
                    # Fallback if __dict__ doesn't work
                    bbox = {
                        'width': getattr(field.key.geometry.boundingBox, 'width', None),
                        'height': getattr(field.key.geometry.boundingBox, 'height', None),
                        'left': getattr(field.key.geometry.boundingBox, 'left', None),
                        'top': getattr(field.key.geometry.boundingBox, 'top', None)
                    }
            
            field_data = {
                "parentKey": parent_key,
                "parentValue": parent_value,
                "keyId": getattr(field.key, 'id', None),
                "key": key,
                "value": value,
                "bbox": bbox
            }
            form_fields.append(field_data)
