if field.value and hasattr(field.value, 'block'):
        relationships = getattr(field.value.block, 'Relationships', [])
        for relationship in relationships:
            if relationship.get('Type') == 'CHILD':
                for child_id in relationship.get('Ids', []):
                    child_block = next(
                        (blk for blk in textract_response.get('Blocks', []) if blk.get('Id') == child_id),
                        None
                    )
                    if child_block and child_block.get('BlockType') == 'SIGNATURE':
                        is_signature = True
                        break
            if is_signature:
                break
