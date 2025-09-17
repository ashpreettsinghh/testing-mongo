def get_signature_block_by_key(textract_response, target_key):
“””
Find signature block associated with a specific key using Textract’s native relationships.

```
Args:
    textract_response: AWS Textract response (dict or boto3 response)
    target_key: The key text to search for (e.g., "Signature", "Sign Here")

Returns:
    dict: Block containing the signature if found, None otherwise
"""
import trp  # AWS Textract Response Parser

# Parse the Textract response using TRP
doc = trp.Document(textract_response)

# Iterate through all pages in the document
for page in doc.pages:
    # Get all form fields (key-value pairs) from the page
    for field in page.form.fields:
        # Check if the key matches our target (case-insensitive partial match)
        if field.key and target_key.lower() in field.key.text.lower():
            # Check if the value contains signature-related content
            if field.value:
                # Textract identifies signatures in forms - check for signature blocks
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
                                return child_block
                
                # If no signature block found in children, return the value block
                # (it might contain signature information)
                return value_block.block

return None
```
