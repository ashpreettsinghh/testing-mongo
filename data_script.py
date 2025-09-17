def extract_all_signatures(textract_response, doc=None):
import logging


try:
    if doc is None:
        import trp
        doc = trp.Document(textract_response)
    
    form_signatures = []
    standalone_signatures = []
    all_signature_blocks = []
    
    signature_blocks = [
        block for block in textract_response.get('Blocks', [])
        if block.get('BlockType') == 'SIGNATURE'
    ]
    
    all_signature_blocks.extend(signature_blocks)
    form_signature_ids = set()
    
    try:
        for page in doc.pages:
            for field in page.form.fields:
                if field.key:
                    try:
                        is_signature = False
                        signature_block = None
                        
                        if field.value and hasattr(field.value, 'block'):
                            relationships = field.value.block.get('Relationships', [])
                            
                            for relationship in relationships:
                                if relationship.get('Type') == 'CHILD':
                                    for child_id in relationship.get('Ids', []):
                                        child_block = next(
                                            (block for block in textract_response.get('Blocks', [])
                                             if block.get('Id') == child_id), None
                                        )
                                        
                                        if (child_block and 
                                            child_block.get('BlockType') == 'SIGNATURE'):
                                            is_signature = True
                                            signature_block = child_block
                                            form_signature_ids.add(child_id)
                                            break
                            
                            if is_signature:
                                form_signatures.append({
                                    'key': field.key.text.strip() if field.key.text else '',
                                    'signature_block': signature_block,
                                    'field': field
                                })
                                
                    except Exception as field_error:
                        logging.warning(f"Error processing form field: {field_error}")
                        continue
                        
    except Exception as form_error:
        logging.warning(f"Error processing form fields: {form_error}")
    
    for sig_block in signature_blocks:
        if sig_block.get('Id') not in form_signature_ids:
            standalone_signatures.append(sig_block)
    
    return {
        'form_signatures': form_signatures,
        'standalone_signatures': standalone_signatures,
        'all_signature_blocks': all_signature_blocks,
        'total_signatures': len(all_signature_blocks),
        'form_signature_count': len(form_signatures),
        'standalone_signature_count': len(standalone_signatures)
    }
    
except Exception as e:
    logging.error(f"Error in extract_all_signatures: {e}")
    return {
        'form_signatures': [],
        'standalone_signatures': [],
        'all_signature_blocks': [],
        'total_signatures': 0,
        'form_signature_count': 0,
        'standalone_signature_count': 0,
        'error': str(e)
    }
```

def is_field_signature(field, textract_response):
try:
if not field.value or not hasattr(field.value, ‘block’):
return False

```
    relationships = field.value.block.get('Relationships', [])
    
    for relationship in relationships:
        if relationship.get('Type') == 'CHILD':
            for child_id in relationship.get('Ids', []):
                child_block = next(
                    (block for block in textract_response.get('Blocks', [])
                     if block.get('Id') == child_id), None
                )
                
                if (child_block and 
                    child_block.get('BlockType') == 'SIGNATURE'):
                    return True
    
    return False
    
except Exception as e:
    logging.warning(f"Error checking field signature: {e}")
    return False
