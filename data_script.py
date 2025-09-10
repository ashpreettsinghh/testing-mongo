import json
import re
import math
from typing import List, Dict, Tuple, Optional
from textractparser import trp

class SignatureMapper:
“””
A class to map signature-related text to actual signature blocks in Textract output.
Uses advanced field organization and handles competitive mapping scenarios.
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
    Organize form fields and text blocks into lines based on geometric position.
    
    Args:
        json_data: Raw Textract JSON response
        
    Returns:
        List of lines, where each line is a list of blocks (form fields + text blocks)
    """
    doc = trp.Document(json_data)
    all_organized_content = []
    
    for page in doc.pages:
        all_blocks = []
        
        # Add form fields
        if hasattr(page, 'form') and page.form and hasattr(page.form, 'fields'):
            for field in page.form.fields:
                # Add key block
                if field.key:
                    all_blocks.append({
                        'type': 'form_key',
                        'text': field.key.text or '',
                        'geometry': field.key.geometry,
                        'block_id': field.key.id,
                        'field_ref': field
                    })
                
                # Add value block  
                if field.value:
                    all_blocks.append({
                        'type': 'form_value',
                        'text': field.value.text or '',
                        'geometry': field.value.geometry,
                        'block_id': field.value.id,
                        'field_ref': field
                    })
        
        # Add text lines (which might contain signature-related text)
        for line in page.lines:
            all_blocks.append({
                'type': 'text_line',
                'text': line.text or '',
                'geometry': line.geometry,
                'block_id': line.id,
                'line_ref': line
            })
        
        # Sort all blocks by top position (Y coordinate)
        all_blocks.sort(key=lambda block: block['geometry'].bounding_box.top if block['geometry'] else 0)
        
        # Group blocks into lines based on Y position
        sorted_content = []
        current_line = []
        
        for block_info in all_blocks:
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
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)

def get_geometry_center(self, geometry) -> Dict[str, float]:
    """Get the center point of a geometry's bounding box."""
    if hasattr(geometry, 'bounding_box'):
        bbox = geometry.bounding_box
        return {
            'x': bbox.left + bbox.width / 2,
            'y': bbox.top + bbox.height / 2
        }
    elif isinstance(geometry, dict) and 'BoundingBox' in geometry:
        bbox = geometry['BoundingBox']
        return {
            'x': bbox['Left'] + bbox['Width'] / 2,
            'y': bbox['Top'] + bbox['Height'] / 2
        }
    else:
        return {'x': 0, 'y': 0}

def find_signature_text_in_organized_content(self, organized_content: List[Dict]) -> List[Dict]:
    """Find signature-related text in the organized content."""
    signature_texts = []
    
    for i, block_info in enumerate(organized_content):
        text = block_info['text'].strip()
        
        # Check if text matches any signature pattern
        for pattern_idx, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                signature_texts.append({
                    'text': text,
                    'pattern_matched': self.signature_patterns[pattern_idx],
                    'block_type': block_info['type'],
                    'block_id': block_info['block_id'],
                    'geometry': block_info['geometry'],
                    'center': self.get_geometry_center(block_info['geometry']),
                    'line_index': i,
                    'full_block_info': block_info
                })
                break  # Stop after first match to avoid duplicates
    
    return signature_texts

def find_signature_blocks_from_raw_blocks(self, json_data: Dict) -> List[Dict]:
    """
    Find signature blocks by looking at raw Textract blocks.
    This looks for actual SIGNATURE type blocks, not form fields.
    """
    signature_blocks = []
    
    # Parse raw blocks from JSON
    if 'Blocks' in json_data:
        for block in json_data['Blocks']:
            # Look for signature blocks
            if block.get('BlockType') == 'SIGNATURE':
                signature_blocks.append({
                    'block_id': block.get('Id'),
                    'geometry': block.get('Geometry', {}),
                    'center': self.get_geometry_center(block.get('Geometry', {})),
                    'confidence': block.get('Confidence'),
                    'detection_method': 'textract_signature_block'
                })
            
            # Also look for blocks that might be signature areas (empty form fields, etc.)
            elif (block.get('BlockType') == 'LINE' and 
                  not block.get('Text', '').strip() and
                  block.get('Geometry', {}).get('BoundingBox', {}).get('Width', 0) > 0.1):
                # This might be an empty signature line
                signature_blocks.append({
                    'block_id': block.get('Id'),
                    'geometry': block.get('Geometry', {}),
                    'center': self.get_geometry_center(block.get('Geometry', {})),
                    'confidence': block.get('Confidence'),
                    'detection_method': 'empty_line_signature_candidate'
                })
    
    # Also use TRP to find any additional signature blocks
    try:
        doc = trp.Document(json_data)
        for page in doc.pages:
            for block in page.blocks:
                if hasattr(block, 'block_type') and block.block_type == 'SIGNATURE':
                    # Avoid duplicates
                    if not any(sb['block_id'] == block.id for sb in signature_blocks):
                        signature_blocks.append({
                            'block_id': block.id,
                            'geometry': block.geometry,
                            'center': self.get_geometry_center(block.geometry),
                            'confidence': getattr(block, 'confidence', None),
                            'detection_method': 'trp_signature_block'
                        })
    except Exception as e:
        print(f"Warning: Could not parse with TRP: {e}")
    
    return signature_blocks

def create_competitive_mappings(self, signature_texts: List[Dict], 
                              signature_blocks: List[Dict], 
                              max_distance: float = 0.15) -> Dict:
    """
    Create mappings using competitive assignment to handle cases where
    multiple signature texts compete for the same signature block.
    """
    # Calculate all possible text-to-block distances
    candidates = []
    
    for text_idx, text_info in enumerate(signature_texts):
        for block_idx, block_info in enumerate(signature_blocks):
            distance = self.calculate_distance(text_info['center'], block_info['center'])
            
            if distance <= max_distance:
                # Create a scoring system
                score = distance
                
                # Bonus for being on the same logical line
                if abs(text_info['center']['y'] - block_info['center']['y']) < self.line_tolerance:
                    score *= 0.5
                
                # Bonus for signature being to the right of text
                if block_info['center']['x'] > text_info['center']['x']:
                    score *= 0.8
                
                candidates.append({
                    'text_idx': text_idx,
                    'block_idx': block_idx,
                    'text_info': text_info,
                    'block_info': block_info,
                    'distance': distance,
                    'score': score
                })
    
    # Sort candidates by score (lower is better)
    candidates.sort(key=lambda x: x['score'])
    
    # Perform greedy assignment - assign best matches first
    used_texts = set()
    used_blocks = set()
    mappings = []
    
    for candidate in candidates:
        text_idx = candidate['text_idx']
        block_idx = candidate['block_idx']
        
        # Skip if either text or block is already assigned
        if text_idx in used_texts or block_idx in used_blocks:
            continue
        
        # Create mapping
        mappings.append({
            'signature_text': candidate['text_info']['text'],
            'pattern_matched': candidate['text_info']['pattern_matched'],
            'block_type': candidate['text_info']['block_type'],
            'text_block_id': candidate['text_info']['block_id'],
            'signature_block_id': candidate['block_info']['block_id'],
            'distance': candidate['distance'],
            'match_score': candidate['score'],
            'line_index': candidate['text_info']['line_index'],
            'detection_method': candidate['block_info']['detection_method'],
            'text_center': candidate['text_info']['center'],
            'signature_center': candidate['block_info']['center']
        })
        
        # Mark as used
        used_texts.add(text_idx)
        used_blocks.add(block_idx)
    
    # Find unmapped texts
    unmapped_texts = []
    for i, text_info in enumerate(signature_texts):
        if i not in used_texts:
            unmapped_texts.append({
                'signature_text': text_info['text'],
                'pattern_matched': text_info['pattern_matched'],
                'block_type': text_info['block_type'],
                'text_block_id': text_info['block_id'],
                'line_index': text_info['line_index'],
                'reason': 'Could not find signature block within acceptable distance or block was assigned to closer text'
            })
    
    return {
        'mappings': mappings,
        'unmapped_texts': unmapped_texts,
        'used_block_indices': used_blocks,
        'competition_candidates': len(candidates)
    }

def map_signatures_advanced(self, textract_json: Dict, max_distance: float = 0.15) -> Dict:
    """
    Advanced signature mapping with competitive assignment.
    """
    try:
        # Organize all content (forms + text)
        organized_content = self.make_organized_content(textract_json)
        
        # Find signature texts in organized content
        signature_texts = self.find_signature_text_in_organized_content(organized_content)
        
        # Find signature blocks from raw Textract blocks
        signature_blocks = self.find_signature_blocks_from_raw_blocks(textract_json)
        
        # Use competitive mapping
        mapping_result = self.create_competitive_mappings(
            signature_texts, signature_blocks, max_distance
        )
        
        return {
            'total_signature_texts_found': len(signature_texts),
            'total_signature_blocks_found': len(signature_blocks),
            'successful_mappings': len(mapping_result['mappings']),
            'unmapped_texts': len(mapping_result['unmapped_texts']),
            'competition_candidates': mapping_result['competition_candidates'],
            'mappings': mapping_result['mappings'],
            'unmapped': mapping_result['unmapped_texts'],
            'unused_signature_blocks': [
                block for i, block in enumerate(signature_blocks) 
                if i not in mapping_result['used_block_indices']
            ],
            'organized_content_count': len(organized_content)
        }
        
    except Exception as e:
        import traceback
        return {
            'error': f"Error processing Textract output: {str(e)}",
            'traceback': traceback.format_exc(),
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
# Example with competitive mapping
result = process_textract_signatures_advanced(‘textract_output.json’)

```
# Print detailed results
print("=== Advanced Signature Mapping Results ===")
print(f"Total organized content blocks: {result.get('organized_content_count', 0)}")
print(f"Total signature texts found: {result.get('total_signature_texts_found', 0)}")
print(f"Total signature blocks found: {result.get('total_signature_blocks_found', 0)}")
print(f"Competition candidates evaluated: {result.get('competition_candidates', 0)}")
print(f"Successful mappings: {result.get('successful_mappings', 0)}")
print(f"Unmapped texts: {result.get('unmapped_texts', 0)}")

if 'mappings' in result and result['mappings']:
    print("\n=== Successful Mappings ===")
    for i, mapping in enumerate(result['mappings'], 1):
        print(f"{i}. Text: '{mapping['signature_text']}'")
        print(f"   Block Type: {mapping['block_type']}")
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
        print(f"  Block Type: {unmapped['block_type']}")
        print(f"  Pattern: {unmapped['pattern_matched']}")
        print(f"  Line Index: {unmapped['line_index']}")
        print(f"  Reason: {unmapped['reason']}")
        print()

if 'unused_signature_blocks' in result and result['unused_signature_blocks']:
    print(f"\n=== Unused Signature Blocks: {len(result['unused_signature_blocks'])} ===")
    for block in result['unused_signature_blocks']:
        print(f"- Block ID: {block['block_id']}")
        print(f"  Detection Method: {block['detection_method']}")
        print(f"  Center: ({block['center']['x']:.3f}, {block['center']['y']:.3f})")
        print()

print("Script completed successfully!")
```
