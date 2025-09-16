# Build a map of all blocks for relationship traversal
blocks_map = {}
for page in doc.pages:
    if hasattr(page, '_blocks'):
        for block in page._blocks:
            # Check if block is a dictionary and has 'Id' key
            if isinstance(block, dict) and 'Id' in block:
                blocks_map[block['Id']] = block

def find_parent_key_value(field_key_block, blocks_map):
    """Find parent key-value pair if available"""
    # Get the ID from the field key block (might be attribute or dict key)
    current_id = None
    if hasattr(field_key_block, 'id'):
        current_id = field_key_block.id
    elif isinstance(field_key_block, dict) and 'Id' in field_key_block:
        current_id = field_key_block['Id']
    
    if not current_id:
        return None, None
    
    for block_id, block in blocks_map.items():
        # Check if block has relationships (as dictionary)
        if not ('Relationships' in block and block['Relationships']):
            continue
            
        for relationship in block['Relationships']:
            # Check if this relationship contains our current block as child
            if not (relationship.get('Type') == 'CHILD'):
                continue
                
            if not ('Ids' in relationship and current_id in relationship['Ids']):
                continue
            
            # Check if parent block is a KEY_VALUE_SET with KEY entity type
            if not (block.get('BlockType') == 'KEY_VALUE_SET'):
                continue
                
            if not ('EntityTypes' in block and block['EntityTypes'] and 'KEY' in block['EntityTypes']):
                continue
            
            # Get parent key text
            parent_key_text = block.get('Text', None)
            parent_value_text = None
            
            # Find corresponding value for this parent key
            if 'Relationships' in block and block['Relationships']:
                for rel in block['Relationships']:
                    if rel.get('Type') == 'VALUE' and 'Ids' in rel:
                        for value_id in rel['Ids']:
                            if value_id in blocks_map:
                                value_block = blocks_map[value_id]
                                parent_value_text = value_block.get('Text', None)
                                break
            
            return parent_key_text, parent_value_text
    
    return None, None
