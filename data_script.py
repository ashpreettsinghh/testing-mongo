def extract_all_signatures(textract_response, doc=None):
“””
Extract all signatures from Textract response - both form-based and standalone signatures.

```
Args:
    textract_response: Raw Textract response (dict or boto3 response)
    doc: Optional TRP Document object (will create if not provided)

Returns:
    dict: {
        'form_signatures': [{'key': str, 'signature_block': dict, 'field': TRP_Field}],
        'standalone_signatures': [signature_block_dict],
        'all_signature_blocks': [all_signature_blocks]
    }
"""
import logging

try:
    # Initialize TRP document if not provided
    if doc is None:
        import trp
        doc = trp.Document(textract_response)
    
    form_signatures = []
    standalone_signatures = []
    all_signature_blocks = []
    
    # Get all signature blocks from raw response first
    signature_blocks = [
        block for block in textract_response.get('Blocks', [])
        if block.get('BlockType') == 'SIGNATURE'
    ]
    
    all_signature_blocks.extend(signature_blocks)
    form_signature_ids = set()  # Track signatures that are part of forms
    
    # 1. Extract form-based signatures (signatures that are values in key-value pairs)
    try:
        for page in doc.pages:
            for field in page.form.fields:
                if field.key:  # Only process fields with keys
                    try:
                        is_signature = False
                        signature_block = None
                        
                        # Check if field has a value with signature
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
                                            form_signature_ids.add(child_id)  # Track this signature
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
    
    # 2. Extract standalone signatures (signatures not part of any form)
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
“””
Simplified method to check if a specific form field contains a signature.
Use this in your existing loop for the isSignature flag.

```
Args:
    field: TRP Field object
    textract_response: Raw Textract response

Returns:
    bool: True if field contains signature, False otherwise
"""
try:
    if not field.value or not hasattr(field.value, 'block'):
        return False
    
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
```
