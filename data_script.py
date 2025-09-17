def get_all_signature_blocks_with_keys(textract_response):
“””
Find all signature blocks that are associated with any key in form key-value pairs.

```
Args:
    textract_response: AWS Textract response (dict or boto3 response)

Returns:
    list: List of dictionaries containing signature blocks and their associated keys
          Format: [{'key': key_text, 'signature_block': signature_block}, ...]
"""
import trp  # AWS Textract Response Parser

# Parse the Textract response using TRP
doc = trp.Document(textract_response)

signature_results = []

# Iterate through all pages in the document
for page in doc.pages:
    # Get all form fields (key-value pairs) from the page
    for field in page.form.fields:
        # Only process fields that have both key and value
        if field.key and field.value:
            
            # Check if the value contains signature-related content
            value_block = field.value
            
            # Look for child blocks that might be signatures
            for relationship in value_block.block.get('Relationships', []):
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        # Find the child block in the response
                        child_block = next(
                            (block for block in textract_response['Blocks'] 
                             if block['Id'] == child_id), None
                        )
                        
                        # Check if this block is identified as a signature
                        if (child_block and 
                            child_block.get('BlockType') == 'SIGNATURE'):
                            
                            signature_results.append({
                                'key': field.key.text,
                                'signature_block': child_block
                            })

return signature_results
```
