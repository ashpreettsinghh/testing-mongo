import json
import re
import math
from typing import List, Dict, Tuple, Optional
from textractparser import trp

class SignatureMapper:
“””
A class to map signature-related text to actual signature blocks in Textract output.
“””

```
def __init__(self, signature_patterns: Optional[List[str]] = None):
    """
    Initialize the SignatureMapper with regex patterns for signature detection.
    
    Args:
        signature_patterns: List of regex patterns to identify signature-related text
    """
    # Default signature patterns - can be customized
    self.signature_patterns = signature_patterns or [
        r'\b[Ss]ignature\b',
        r'\b[Ss]ign\b',
        r'\b[Ss]igned\b',
        r'\b[Ss]ig\b',
        r'\bX\b',  # Common signature placeholder
        r'Date.*[Ss]igned',
        r'[Ss]ignature.*[Dd]ate',
    ]
    
    # Compile regex patterns for efficiency
    self.compiled_patterns = [re.compile(pattern) for pattern in self.signature_patterns]

def calculate_distance(self, point1: Dict, point2: Dict) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: Dictionary with 'x' and 'y' coordinates
        point2: Dictionary with 'x' and 'y' coordinates
        
    Returns:
        Euclidean distance between the points
    """
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

def get_text_center(self, geometry: Dict) -> Dict[str, float]:
    """
    Get the center point of a text block's bounding box.
    
    Args:
        geometry: Textract geometry object
        
    Returns:
        Dictionary with center coordinates
    """
    bbox = geometry['BoundingBox']
    return {
        'x': bbox['Left'] + bbox['Width'] / 2,
        'y': bbox['Top'] + bbox['Height'] / 2
    }

def find_signature_text(self, document: trp.Document) -> List[Dict]:
    """
    Find all text blocks that match signature patterns.
    
    Args:
        document: Parsed Textract document
        
    Returns:
        List of dictionaries containing signature text information
    """
    signature_texts = []
    
    for page in document.pages:
        for line in page.lines:
            text = line.text.strip()
            
            # Check if text matches any signature pattern
            for i, pattern in enumerate(self.compiled_patterns):
                if pattern.search(text):
                    signature_texts.append({
                        'text': text,
                        'pattern_matched': self.signature_patterns[i],
                        'geometry': line.geometry,
                        'center': self.get_text_center(line.geometry),
                        'page_number': page.id,
                        'block_id': line.id
                    })
                    break  # Stop after first match to avoid duplicates
    
    return signature_texts

def find_signature_blocks(self, document: trp.Document) -> List[Dict]:
    """
    Find all signature blocks in the document.
    
    Args:
        document: Parsed Textract document
        
    Returns:
        List of dictionaries containing signature block information
    """
    signature_blocks = []
    
    for page in document.pages:
        # Look for signature blocks in form fields and other elements
        for block in page.blocks:
            if hasattr(block, 'block_type') and block.block_type == 'SIGNATURE':
                signature_blocks.append({
                    'block_id': block.id,
                    'geometry': block.geometry,
                    'center': self.get_text_center(block.geometry),
                    'page_number': page.id,
                    'confidence': getattr(block, 'confidence', None)
                })
    
    return signature_blocks

def find_nearest_signature(self, text_info: Dict, signature_blocks: List[Dict], 
                         max_distance: float = 0.3) -> Optional[Dict]:
    """
    Find the nearest signature block to a given text.
    
    Args:
        text_info: Information about the signature text
        signature_blocks: List of available signature blocks
        max_distance: Maximum distance to consider for matching
        
    Returns:
        Nearest signature block or None if none found within max_distance
    """
    if not signature_blocks:
        return None
    
    # Filter signature blocks on the same page
    same_page_blocks = [
        block for block in signature_blocks 
        if block['page_number'] == text_info['page_number']
    ]
    
    if not same_page_blocks:
        # If no blocks on same page, consider all blocks but with higher distance penalty
        same_page_blocks = signature_blocks
    
    # Calculate distances and find nearest
    nearest_block = None
    min_distance = float('inf')
    
    for block in same_page_blocks:
        distance = self.calculate_distance(text_info['center'], block['center'])
        
        # Add penalty if not on same page
        if block['page_number'] != text_info['page_number']:
            distance *= 2
        
        if distance < min_distance and distance <= max_distance:
            min_distance = distance
            nearest_block = block
    
    if nearest_block:
        nearest_block['distance'] = min_distance
    
    return nearest_block

def map_signatures(self, textract_json: Dict, max_distance: float = 0.3) -> Dict:
    """
    Main method to map signature texts to signature blocks.
    
    Args:
        textract_json: Raw Textract JSON response
        max_distance: Maximum distance for signature matching
        
    Returns:
        Dictionary containing mapping results
    """
    try:
        # Parse the Textract JSON using trp
        document = trp.Document(textract_json)
        
        # Find signature-related text and signature blocks
        signature_texts = self.find_signature_text(document)
        signature_blocks = self.find_signature_blocks(document)
        
        # Map signature texts to signature blocks
        mappings = []
        unmapped_texts = []
        used_blocks = set()
        
        # Sort signature texts by confidence or position for consistent processing
        signature_texts.sort(key=lambda x: (x['page_number'], x['center']['y'], x['center']['x']))
        
        for text_info in signature_texts:
            # Find available signature blocks (not already used)
            available_blocks = [
                block for block in signature_blocks 
                if block['block_id'] not in used_blocks
            ]
            
            nearest_block = self.find_nearest_signature(text_info, available_blocks, max_distance)
            
            if nearest_block:
                mappings.append({
                    'signature_text': text_info['text'],
                    'pattern_matched': text_info['pattern_matched'],
                    'text_block_id': text_info['block_id'],
                    'signature_block_id': nearest_block['block_id'],
                    'distance': nearest_block['distance'],
                    'page_number': text_info['page_number'],
                    'text_center': text_info['center'],
                    'signature_center': nearest_block['center']
                })
                used_blocks.add(nearest_block['block_id'])
            else:
                unmapped_texts.append({
                    'signature_text': text_info['text'],
                    'pattern_matched': text_info['pattern_matched'],
                    'text_block_id': text_info['block_id'],
                    'page_number': text_info['page_number'],
                    'reason': 'No signature block found within acceptable distance'
                })
        
        return {
            'total_signature_texts_found': len(signature_texts),
            'total_signature_blocks_found': len(signature_blocks),
            'successful_mappings': len(mappings),
            'unmapped_texts': len(unmapped_texts),
            'mappings': mappings,
            'unmapped': unmapped_texts,
            'unused_signature_blocks': [
                block for block in signature_blocks 
                if block['block_id'] not in used_blocks
            ]
        }
        
    except Exception as e:
        return {
            'error': f"Error processing Textract output: {str(e)}",
            'total_signature_texts_found': 0,
            'total_signature_blocks_found': 0,
            'successful_mappings': 0,
            'unmapped_texts': 0,
            'mappings': [],
            'unmapped': []
        }
```

def process_textract_signatures(json_file_path: str,
custom_patterns: Optional[List[str]] = None,
max_distance: float = 0.3) -> Dict:
“””
Convenience function to process a Textract JSON file.

```
Args:
    json_file_path: Path to the Textract JSON output file
    custom_patterns: Optional custom regex patterns for signature detection
    max_distance: Maximum distance for signature matching
    
Returns:
    Mapping results
"""
try:
    with open(json_file_path, 'r') as file:
        textract_json = json.load(file)
    
    mapper = SignatureMapper(custom_patterns)
    return mapper.map_signatures(textract_json, max_distance)

except FileNotFoundError:
    return {'error': f"File not found: {json_file_path}"}
except json.JSONDecodeError:
    return {'error': f"Invalid JSON in file: {json_file_path}"}
except Exception as e:
    return {'error': f"Error processing file: {str(e)}"}
```

# Example usage

if **name** == “**main**”:
# Example 1: Process a JSON file
result = process_textract_signatures(‘textract_output.json’)

```
# Print results
print("=== Signature Mapping Results ===")
print(f"Total signature texts found: {result.get('total_signature_texts_found', 0)}")
print(f"Total signature blocks found: {result.get('total_signature_blocks_found', 0)}")
print(f"Successful mappings: {result.get('successful_mappings', 0)}")
print(f"Unmapped texts: {result.get('unmapped_texts', 0)}")

if 'mappings' in result:
    print("\n=== Successful Mappings ===")
    for i, mapping in enumerate(result['mappings'], 1):
        print(f"{i}. Text: '{mapping['signature_text']}'")
        print(f"   Pattern: {mapping['pattern_matched']}")
        print(f"   Distance: {mapping['distance']:.4f}")
        print(f"   Page: {mapping['page_number']}")
        print()

if 'unmapped' in result and result['unmapped']:
    print("=== Could Not Find Signature For Text ===")
    for unmapped in result['unmapped']:
        print(f"- Text: '{unmapped['signature_text']}'")
        print(f"  Pattern: {unmapped['pattern_matched']}")
        print(f"  Reason: {unmapped['reason']}")
        print()

# Example 2: Using custom patterns
custom_patterns = [
    r'\b[Ss]ignature\b',
    r'\b[Aa]uthorized\s+[Ss]ignature\b',
    r'\b[Dd]igital\s+[Ss]ignature\b'
]

mapper = SignatureMapper(custom_patterns)
# result = mapper.map_signatures(textract_json_data)

print("Script completed successfully!")
```
