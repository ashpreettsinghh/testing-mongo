import json
import re
import math
from typing import List, Dict, Tuple, Optional
from textractparser import trp

class SignatureMapper:
“””
A class to map signature-related text to actual signature blocks in Textract output.
Uses advanced field organization and line-based processing similar to your existing logic.
“””

```
def __init__(self, signature_patterns: Optional[List[str]] = None, line_tolerance: float = 0.01):
    """
    Initialize the SignatureMapper with regex patterns for signature detection.
    
    Args:
        signature_patterns: List of regex patterns to identify signature-related text
        line_tolerance: Tolerance for determining if blocks are on the same line
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
        r'[Ss]ign\s+[Hh]ere',
        r'[Aa]uthorized\s+[Ss]ignature',
    ]
    
    # Compile regex patterns for efficiency
    self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.signature_patterns]
    self.line_tolerance = line_tolerance

def make_organized_content(self, json_data: Dict) -> List[List[Dict]]:
    """
    Organize form fields into lines based on your existing logic.
    This mirrors the make_text_files function from your code.
    
    Args:
        json_data: Raw Textract JSON response
        
    Returns:
        List of lines, where each line is a list of field blocks
    """
    doc = trp.Document(json_data)
    all_organized_content = []
    
    for page in doc.pages:
        # Get all form fields and sort by top position
        sorted_fields = []
        if hasattr(page, 'form') and page.form and hasattr(page.form, 'fields'):
            for field in page.form.fields:
                sorted_fields.append({
                    'field': field,
                    'geometry': field.geometry if hasattr(field, 'geometry') else None,
                    'key': field.key.text if field.key else '',
                    'value': field.value.text if field.value else '',
                    'key_id': field.key.id if field.key else '',
                    'value_id': field.value.id if field.value else '',
                })
        
        # Sort by top position (Y coordinate)
        sorted_fields.sort(key=lambda block: block['geometry'].bounding_box.top if block['geometry'] else 0)
        
        # Group fields into lines
        sorted_content = []
        current_line = []
        
        for block_info in sorted_fields:
            if not current_line:
                current_line.append(block_info)
            else:
                last_block_in_line = current_line[-1]
                
                # Check if the current block is on the same line as the last block
                if (block_info['geometry'] and last_block_in_line['geometry'] and
                    abs(block_info['geometry'].bounding_box.top - 
                        last_block_in_line['geometry'].bounding_box.top) < self.line_tolerance):
                    current_line.append(block_info)
                else:
                    # New line detected, sort the previous line by left position
                    current_line.sort(key=lambda b: b['geometry'].bounding_box.left if b['geometry'] else 0)
                    sorted_content.extend(current_line)
                    
                    # Start a new line with the current block
                    current_line = [block_info]
        
        # Don't forget to sort and add the last line
        if current_line:
            current_line.sort(key=lambda b: b['geometry'].bounding_box.left if b['geometry'] else 0)
            sorted_content.extend(current_line)
        
        all_organized_content.extend(sorted_content)
    
    return all_organized_content

def calculate_distance(self, point1: Dict, point2: Dict) -> float:
    """
    Calculate Euclidean distance between two points.
    """
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

def get_geometry_center(self, geometry) -> Dict[str, float]:
    """
    Get the center point of a geometry's bounding box.
    """
    if hasattr(geometry, 'bounding_box'):
        bbox = geometry.bounding_box
    elif isinstance(geometry, dict) and 'BoundingBox' in geometry:
        bbox = geometry['BoundingBox']
    else:
        return {'x': 0, 'y': 0}
    
    return {
        'x': bbox.left + bbox.width / 2 if hasattr(bbox, 'left') else bbox['Left'] + bbox['Width'] / 2,
        'y': bbox.top + bbox.height / 2 if hasattr(bbox, 'top') else bbox['Top'] + bbox['Height'] / 2
    }

def find_signature_text_in_organized_content(self, organized_content: List[Dict]) -> List[Dict]:
    """
    Find signature-related text in the organized content.
    
    Args:
        organized_content: Organized form fields
        
    Returns:
        List of signature text information
    """
    signature_texts = []
    
    for i, field_info in enumerate(organized_content):
        # Check both key and value for signature patterns
        texts_to_check = []
        
        if field_info['key']:
            texts_to_check.append({
                'text': field_info['key'],
                'type': 'key',
                'id': field_info['key_id'],
                'geometry': field_info['field'].key.geometry if field_info['field'].key else None
            })
        
        if field_info['value']:
            texts_to_check.append({
                'text': field_info['value'],
                'type': 'value', 
                'id': field_info['value_id'],
                'geometry': field_info['field'].value.geometry if field_info['field'].value else None
            })
        
        for text_info in texts_to_check:
            text = text_info['text'].strip()
            
            # Check if text matches any signature pattern
            for pattern_idx, pattern in enumerate(self.compiled_patterns):
                if pattern.search(text):
                    signature_texts.append({
                        'text': text,
                        'pattern_matched': self.signature_patterns[pattern_idx],
                        'field_type': text_info['type'],
                        'field_id': text_info['id'],
                        'geometry': text_info['geometry'],
                        'center': self.get_geometry_center(text_info['geometry']) if text_info['geometry'] else {'x': 0, 'y': 0},
                        'line_index': i,
                        'full_field_info': field_info
                    })
                    break
    
    return signature_texts

def find_signature_blocks_advanced(self, json_data: Dict) -> List[Dict]:
    """
    Find signature blocks using advanced detection methods.
    """
    doc = trp.Document(json_data)
    signature_blocks = []
    
    for page in doc.pages:
        # Method 1: Look for explicit signature blocks
        for block in page.blocks:
            if hasattr(block, 'block_type') and block.block_type == 'SIGNATURE':
                signature_blocks.append({
                    'block_id': block.id,
                    'geometry': block.geometry,
                    'center': self.get_geometry_center(block.geometry),
                    'page_number': page.id,
                    'confidence': getattr(block, 'confidence', None),
                    'detection_method': 'explicit_signature_block'
                })
        
        # Method 2: Look for form fields that might be signature fields
        if hasattr(page, 'form') and page.form and hasattr(page.form, 'fields'):
            for field in page.form.fields:
                # Check if field key suggests it's a signature field
                key_text = field.key.text.lower() if field.key and field.key.text else ''
                value_text = field.value.text.lower() if field.value and field.value.text else ''
                
                is_signature_field = False
                detection_method = ''
                
                # Check if key indicates signature
                if any(pattern.search(key_text) for pattern in self.compiled_patterns):
                    is_signature_field = True
                    detection_method = 'key_pattern_match'
                
                # Check if value is empty (common for signature fields)
                elif 'signature' in key_text and not value_text.strip():
                    is_signature_field = True
                    detection_method = 'empty_signature_field'
                
                # Check for common signature field indicators
                elif any(indicator in key_text for indicator in ['sign', 'x mark', 'initial', 'auth']):
                    is_signature_field = True
                    detection_method = 'signature_indicator'
                
                if is_signature_field:
                    # Use the value geometry if available, otherwise key geometry
                    geometry = field.value.geometry if field.value else field.key.geometry
                    if geometry:
                        signature_blocks.append({
                            'block_id': field.value.id if field.value else field.key.id,
                            'geometry': geometry,
                            'center': self.get_geometry_center(geometry),
                            'page_number': page.id,
                            'confidence': None,
                            'detection_method': detection_method,
                            'field_key': key_text,
                            'field_value': value_text
                        })
    
    return signature_blocks

def find_nearest_signature_advanced(self, text_info: Dict, signature_blocks: List[Dict], 
                                  max_distance: float = 0.15) -> Optional[Dict]:
    """
    Find the nearest signature block using advanced matching logic.
    """
    if not signature_blocks:
        return None
    
    # Calculate distances and potential matches
    candidates = []
    
    for block in signature_blocks:
        distance = self.calculate_distance(text_info['center'], block['center'])
        
        # Scoring system for better matching
        score = distance
        
        # Bonus for being on the same logical line (lower score is better)
        if abs(text_info['center']['y'] - block['center']['y']) < self.line_tolerance:
            score *= 0.5  # Significant bonus for same line
        
        # Bonus for being to the right of the text (common signature placement)
        if block['center']['x'] > text_info['center']['x']:
            score *= 0.8
        
        # Penalty for being too far away
        if distance > max_distance:
            score *= 2
        
        candidates.append({
            'block': block,
            'distance': distance,
            'score': score
        })
    
    # Sort by score (lower is better)
    candidates.sort(key=lambda x: x['score'])
    
    best_candidate = candidates[0] if candidates else None
    
    if best_candidate and best_candidate['distance'] <= max_distance:
        result = best_candidate['block'].copy()
        result['distance'] = best_candidate['distance']
        result['match_score'] = best_candidate['score']
        return result
    
    return None

def map_signatures_advanced(self, textract_json: Dict, max_distance: float = 0.15) -> Dict:
    """
    Advanced signature mapping using organized content and sophisticated matching.
    """
    try:
        # Organize content using your logic
        organized_content = self.make_organized_content(textract_json)
        
        # Find signature texts in organized content
        signature_texts = self.find_signature_text_in_organized_content(organized_content)
        
        # Find signature blocks using advanced detection
        signature_blocks = self.find_signature_blocks_advanced(textract_json)
        
        # Map signature texts to signature blocks
        mappings = []
        unmapped_texts = []
        used_blocks = set()
        
        # Sort signature texts by line position for consistent processing
        signature_texts.sort(key=lambda x: (x['line_index'], x['center']['x']))
        
        for text_info in signature_texts:
            # Find available signature blocks
            available_blocks = [
                block for block in signature_blocks 
                if block['block_id'] not in used_blocks
            ]
            
            nearest_block = self.find_nearest_signature_advanced(text_info, available_blocks, max_distance)
            
            if nearest_block:
                mappings.append({
                    'signature_text': text_info['text'],
                    'pattern_matched': text_info['pattern_matched'],
                    'field_type': text_info['field_type'],
                    'text_field_id': text_info['field_id'],
                    'signature_block_id': nearest_block['block_id'],
                    'distance': nearest_block['distance'],
                    'match_score': nearest_block['match_score'],
                    'line_index': text_info['line_index'],
                    'detection_method': nearest_block['detection_method'],
                    'text_center': text_info['center'],
                    'signature_center': nearest_block['center']
                })
                used_blocks.add(nearest_block['block_id'])
            else:
                unmapped_texts.append({
                    'signature_text': text_info['text'],
                    'pattern_matched': text_info['pattern_matched'],
                    'field_type': text_info['field_type'],
                    'text_field_id': text_info['field_id'],
                    'line_index': text_info['line_index'],
                    'reason': 'Could not find signature block within acceptable distance'
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
            ],
            'organized_fields_count': len(organized_content)
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

def process_textract_signatures_advanced(json_file_path: str,
custom_patterns: Optional[List[str]] = None,
max_distance: float = 0.15,
line_tolerance: float = 0.01) -> Dict:
“””
Advanced convenience function to process a Textract JSON file.
“””
try:
with open(json_file_path, ‘r’) as file:
textract_json = json.load(file)

```
    mapper = SignatureMapper(custom_patterns, line_tolerance)
    return mapper.map_signatures_advanced(textract_json, max_distance)

except FileNotFoundError:
    return {'error': f"File not found: {json_file_path}"}
except json.JSONDecodeError:
    return {'error': f"Invalid JSON in file: {json_file_path}"}
except Exception as e:
    return {'error': f"Error processing file: {str(e)}"}
```

# Example usage

if **name** == “**main**”:
# Example with advanced processing
result = process_textract_signatures_advanced(‘textract_output.json’)

```
# Print detailed results
print("=== Advanced Signature Mapping Results ===")
print(f"Total organized fields: {result.get('organized_fields_count', 0)}")
print(f"Total signature texts found: {result.get('total_signature_texts_found', 0)}")
print(f"Total signature blocks found: {result.get('total_signature_blocks_found', 0)}")
print(f"Successful mappings: {result.get('successful_mappings', 0)}")
print(f"Unmapped texts: {result.get('unmapped_texts', 0)}")

if 'mappings' in result and result['mappings']:
    print("\n=== Successful Mappings ===")
    for i, mapping in enumerate(result['mappings'], 1):
        print(f"{i}. Text: '{mapping['signature_text']}'")
        print(f"   Field Type: {mapping['field_type']}")
        print(f"   Pattern: {mapping['pattern_matched']}")
        print(f"   Distance: {mapping['distance']:.4f}")
        print(f"   Match Score: {mapping['match_score']:.4f}")
        print(f"   Detection Method: {mapping['detection_method']}")
        print(f"   Line Index: {mapping['line_index']}")
        print()

if 'unmapped' in result and result['unmapped']:
    print("=== Could Not Find Signature For Text ===")
    for unmapped in result['unmapped']:
        print(f"- Text: '{unmapped['signature_text']}'")
        print(f"  Field Type: {unmapped['field_type']}")
        print(f"  Pattern: {unmapped['pattern_matched']}")
        print(f"  Line Index: {unmapped['line_index']}")
        print(f"  Reason: {unmapped['reason']}")
        print()

# Example with custom settings
custom_patterns = [
    r'\b[Ss]ignature\b',
    r'\b[Aa]uthorized\s+[Ss]ignature\b',
    r'\b[Dd]igital\s+[Ss]ignature\b',
    r'\b[Ss]ign\s+[Hh]ere\b'
]

print("=== Processing with Custom Patterns ===")
custom_result = process_textract_signatures_advanced(
    'textract_output.json',
    custom_patterns=custom_patterns,
    max_distance=0.2,
    line_tolerance=0.015
)

print(f"Custom processing found {custom_result.get('successful_mappings', 0)} mappings")
print("Script completed successfully!")
```
