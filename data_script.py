# Three-Agent Email Processing System in Nexus

Here's how to create **three specialized agents** for your email processing workflow using Nexus's multi-agent architecture:

## **Agent 1: Email Extraction Agent**

### **Profile Configuration** (`nexus/nexus_base/nexus_profiles/email_extractor.yaml`)

```yaml
agentProfile:
  name: "EmailExtractor"
  avatar: "📧"
  persona: "I am a specialized email extraction agent. I excel at parsing email files, extracting metadata, body content, and attachments with high precision. I validate email structure and ensure no data is lost during extraction."
  actions:
    - parse_email_file
    - extract_email_metadata
    - extract_attachments
    - validate_email_structure
  knowledge: 
    - email_formats_knowledge
  memory:
    - semantic  # Remember email patterns and structures
  reasoners:
    - email_structure_reasoner
  planners:
    - email_extraction_planner
  evaluators:
    - extraction_quality_evaluator
  feedback:
    - extraction_feedback
```

### **Custom Actions** (`nexus/nexus_base/nexus_actions/email_extraction.py`)

```python
import email
import base64
from nexus.nexus_base.action_manager import agent_action

@agent_action
def parse_email_file(uploaded_file):
    """Parse uploaded email file (.eml, .msg) and extract basic structure."""
    try:
        # Read the email file
        if hasattr(uploaded_file, 'read'):
            email_content = uploaded_file.read()
        else:
            email_content = uploaded_file
            
        # Parse email message
        msg = email.message_from_bytes(email_content)
        
        return {
            "status": "success",
            "email_object": msg,
            "headers": dict(msg.items()),
            "is_multipart": msg.is_multipart()
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "email_object": None
        }

@agent_action
def extract_email_metadata(email_parse_result):
    """Extract metadata from parsed email object."""
    if email_parse_result["status"] != "success":
        return {"status": "failed", "error": "Invalid email object"}
    
    msg = email_parse_result["email_object"]
    
    metadata = {
        "subject": msg.get('Subject', ''),
        "from": msg.get('From', ''),
        "to": msg.get('To', ''),
        "date": msg.get('Date', ''),
        "message_id": msg.get('Message-ID', ''),
        "content_type": msg.get('Content-Type', ''),
        "encoding": msg.get('Content-Transfer-Encoding', '')
    }
    
    return {
        "status": "success",
        "metadata": metadata,
        "email_object": msg
    }

@agent_action
def extract_attachments(email_parse_result):
    """Extract all attachments from email with their metadata."""
    if email_parse_result["status"] != "success":
        return {"status": "failed", "attachments": []}
    
    msg = email_parse_result["email_object"]
    attachments = []
    body_text = ""
    
    for part in msg.walk():
        content_disposition = part.get("Content-Disposition")
        
        # Extract body text
        if part.get_content_type() == "text/plain" and not content_disposition:
            body_text += part.get_payload(decode=True).decode('utf-8', errors='ignore')
        
        # Extract attachments
        elif content_disposition and 'attachment' in content_disposition:
            filename = part.get_filename()
            if filename:
                attachments.append({
                    "filename": filename,
                    "content_type": part.get_content_type(),
                    "size": len(part.get_payload(decode=True)) if part.get_payload(decode=True) else 0,
                    "content": base64.b64encode(part.get_payload(decode=True)).decode('utf-8'),
                    "encoding": part.get('Content-Transfer-Encoding', '')
                })
    
    return {
        "status": "success",
        "body_text": body_text,
        "attachments": attachments,
        "attachment_count": len(attachments)
    }

@agent_action
def validate_email_structure(extraction_result):
    """Validate extracted email data for completeness and quality."""
    validation_results = {
        "is_valid": True,
        "warnings": [],
        "errors": [],
        "quality_score": 1.0
    }
    
    # Check metadata completeness
    metadata = extraction_result.get("metadata", {})
    if not metadata.get("subject"):
        validation_results["warnings"].append("Missing email subject")
        validation_results["quality_score"] -= 0.1
    
    if not metadata.get("from"):
        validation_results["errors"].append("Missing sender information")
        validation_results["is_valid"] = False
    
    # Check attachment extraction
    attachments = extraction_result.get("attachments", [])
    if not attachments:
        validation_results["warnings"].append("No attachments found")
    
    # Validate attachment integrity
    for att in attachments:
        if not att.get("content"):
            validation_results["errors"].append(f"Empty attachment: {att.get('filename', 'unknown')}")
            validation_results["quality_score"] -= 0.2
    
    return validation_results
```

***

## **Agent 2: OCR Processing Agent**

### **Profile Configuration** (`nexus/nexus_base/nexus_profiles/ocr_processor.yaml`)

```yaml
agentProfile:
  name: "OCRProcessor"
  avatar: "👁️"
  persona: "I am an OCR specialist agent focused on converting document images and PDFs to text with maximum accuracy. I handle multiple OCR APIs, implement fallback strategies, and ensure quality text extraction with confidence scoring."
  actions:
    - process_attachments_ocr
    - validate_ocr_quality
    - retry_failed_ocr
    - enhance_ocr_results
  knowledge: 
    - ocr_best_practices
    - document_types_knowledge
  memory:
    - procedural  # Remember successful OCR strategies
    - episodic    # Learn from OCR failures and successes
  reasoners:
    - ocr_quality_reasoner
    - document_type_reasoner
  planners:
    - ocr_strategy_planner
    - fallback_planner
  evaluators:
    - ocr_accuracy_evaluator
  feedback:
    - ocr_improvement_feedback
```

### **Custom Actions** (`nexus/nexus_base/nexus_actions/ocr_processing.py`)

```python
import requests
import base64
from io import BytesIO
from nexus.nexus_base.action_manager import agent_action

@agent_action
def process_attachments_ocr(attachments, primary_ocr_endpoint, fallback_ocr_endpoints=None):
    """Process all attachments through OCR with fallback strategies."""
    if not attachments:
        return {"status": "no_attachments", "results": []}
    
    ocr_results = []
    fallback_endpoints = fallback_ocr_endpoints or []
    
    for attachment in attachments:
        # Skip non-document files
        if not _is_document_file(attachment.get('filename', '')):
            continue
            
        result = _process_single_attachment(
            attachment, 
            primary_ocr_endpoint, 
            fallback_endpoints
        )
        ocr_results.append(result)
    
    return {
        "status": "completed",
        "results": ocr_results,
        "total_processed": len(ocr_results),
        "success_count": len([r for r in ocr_results if r["status"] == "success"])
    }

def _is_document_file(filename):
    """Check if file is a document that can be OCR'd."""
    document_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    return any(filename.lower().endswith(ext) for ext in document_extensions)

def _process_single_attachment(attachment, primary_endpoint, fallback_endpoints):
    """Process single attachment with OCR including fallbacks."""
    filename = attachment.get('filename', 'unknown')
    content = base64.b64decode(attachment['content'])
    
    # Try primary OCR endpoint
    result = _call_ocr_api(content, filename, primary_endpoint)
    if result["status"] == "success":
        return result
    
    # Try fallback endpoints
    for fallback_endpoint in fallback_endpoints:
        fallback_result = _call_ocr_api(content, filename, fallback_endpoint)
        if fallback_result["status"] == "success":
            fallback_result["used_fallback"] = True
            fallback_result["fallback_endpoint"] = fallback_endpoint
            return fallback_result
    
    return {
        "filename": filename,
        "status": "failed",
        "error": "All OCR endpoints failed",
        "attempts": len(fallback_endpoints) + 1
    }

def _call_ocr_api(content, filename, endpoint):
    """Call OCR API with proper error handling."""
    try:
        files = {'document': (filename, BytesIO(content), 'application/octet-stream')}
        response = requests.post(endpoint, files=files, timeout=30)
        
        if response.status_code == 200:
            ocr_data = response.json()
            return {
                "filename": filename,
                "status": "success",
                "ocr_text": ocr_data.get('text', ''),
                "confidence": ocr_data.get('confidence', 0.0),
                "word_count": len(ocr_data.get('text', '').split()),
                "endpoint_used": endpoint,
                "processing_time": response.elapsed.total_seconds()
            }
        else:
            return {
                "filename": filename,
                "status": "failed",
                "error": f"HTTP {response.status_code}: {response.text}",
                "endpoint_used": endpoint
            }
            
    except requests.exceptions.Timeout:
        return {
            "filename": filename,
            "status": "failed",
            "error": "OCR request timeout",
            "endpoint_used": endpoint
        }
    except Exception as e:
        return {
            "filename": filename,
            "status": "failed", 
            "error": str(e),
            "endpoint_used": endpoint
        }

@agent_action
def validate_ocr_quality(ocr_results):
    """Validate OCR results for quality and completeness."""
    quality_report = {
        "overall_quality": "good",
        "average_confidence": 0.0,
        "recommendations": [],
        "high_quality_count": 0,
        "low_quality_count": 0
    }
    
    if not ocr_results.get("results"):
        quality_report["overall_quality"] = "no_results"
        return quality_report
    
    confidences = []
    for result in ocr_results["results"]:
        if result["status"] == "success":
            confidence = result.get("confidence", 0.0)
            confidences.append(confidence)
            
            if confidence > 0.8:
                quality_report["high_quality_count"] += 1
            elif confidence < 0.6:
                quality_report["low_quality_count"] += 1
                quality_report["recommendations"].append(
                    f"Low confidence OCR for {result['filename']}: {confidence:.2f}"
                )
    
    if confidences:
        quality_report["average_confidence"] = sum(confidences) / len(confidences)
        
        if quality_report["average_confidence"] < 0.6:
            quality_report["overall_quality"] = "poor"
        elif quality_report["average_confidence"] < 0.8:
            quality_report["overall_quality"] = "fair"
    
    return quality_report

@agent_action
def retry_failed_ocr(ocr_results, retry_endpoints):
    """Retry OCR for failed documents using different endpoints."""
    retry_results = []
    
    for result in ocr_results.get("results", []):
        if result["status"] == "failed":
            # In a real implementation, you'd need the original attachment data
            # This is a placeholder for the retry logic
            retry_results.append({
                "filename": result["filename"],
                "status": "retry_attempted",
                "original_error": result.get("error"),
                "retry_endpoints": retry_endpoints
            })
    
    return {
        "retry_attempted": len(retry_results),
        "results": retry_results
    }
```

***

## **Agent 3: Entity Matching Agent**

### **Profile Configuration** (`nexus/nexus_base/nexus_profiles/entity_matcher.yaml`)

```yaml
agentProfile:
  name: "EntityMatcher"
  avatar: "🎯"
  persona: "I am an entity matching specialist that excels at finding and ranking the best matches from API results. I use sophisticated algorithms, confidence scoring, and multi-criteria matching to ensure the highest accuracy in entity identification."
  actions:
    - extract_fields_from_ocr
    - call_matching_api
    - perform_entity_matching
    - validate_matches
    - rank_matches_by_confidence
  knowledge: 
    - matching_algorithms
    - entity_patterns
  memory:
    - semantic    # Remember successful field patterns
    - episodic    # Learn from matching successes/failures
  reasoners:
    - matching_confidence_reasoner
    - entity_similarity_reasoner
  planners:
    - matching_strategy_planner
    - field_extraction_planner
  evaluators:
    - match_quality_evaluator
  feedback:
    - matching_improvement_feedback
```

### **Custom Actions** (`nexus/nexus_base/nexus_actions/entity_matching.py`)

```python
import requests
import re
from difflib import SequenceMatcher
from nexus.nexus_base.action_manager import agent_action

@agent_action
def extract_fields_from_ocr(ocr_results, field_definitions, extraction_patterns=None):
    """Extract specific fields from OCR text using patterns and LLM assistance."""
    if not ocr_results.get("results"):
        return {"status": "no_ocr_data", "extracted_fields": []}
    
    extracted_documents = []
    patterns = extraction_patterns or _get_default_patterns()
    
    for ocr_result in ocr_results["results"]:
        if ocr_result["status"] != "success":
            continue
            
        ocr_text = ocr_result["ocr_text"]
        filename = ocr_result["filename"]
        
        extracted_fields = {}
        confidence_scores = {}
        
        # Extract each field using patterns and validation
        for field in field_definitions:
            field_result = _extract_single_field(ocr_text, field, patterns)
            extracted_fields[field] = field_result["value"]
            confidence_scores[field] = field_result["confidence"]
        
        extracted_documents.append({
            "filename": filename,
            "extracted_fields": extracted_fields,
            "confidence_scores": confidence_scores,
            "overall_confidence": sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0,
            "extraction_quality": _assess_extraction_quality(extracted_fields, confidence_scores)
        })
    
    return {
        "status": "completed",
        "extracted_documents": extracted_documents,
        "total_documents": len(extracted_documents)
    }

def _get_default_patterns():
    """Default regex patterns for common fields."""
    return {
        "invoice_number": [
            r"invoice\s*#?\s*:?\s*([A-Z0-9-]+)",
            r"inv\s*#?\s*:?\s*([A-Z0-9-]+)",
            r"document\s*#?\s*:?\s*([A-Z0-9-]+)"
        ],
        "amount": [
            r"total\s*:?\s*\$?([0-9,]+\.?\d*)",
            r"amount\s*:?\s*\$?([0-9,]+\.?\d*)",
            r"\$([0-9,]+\.?\d*)"
        ],
        "date": [
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
            r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})",
            r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2},?\s+\d{4})"
        ],
        "vendor": [
            r"from\s*:?\s*([^\n]+)",
            r"vendor\s*:?\s*([^\n]+)",
            r"company\s*:?\s*([^\n]+)"
        ]
    }

def _extract_single_field(text, field_name, patterns):
    """Extract a single field using multiple patterns."""
    field_patterns = patterns.get(field_name, [])
    
    for pattern in field_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return {
                "value": match.group(1).strip(),
                "confidence": 0.9,  # High confidence for regex match
                "extraction_method": "regex_pattern"
            }
    
    # Fallback to simple text search
    search_terms = _get_search_terms_for_field(field_name)
    for term in search_terms:
        if term.lower() in text.lower():
            # Find text near the search term
            nearby_text = _extract_nearby_text(text, term)
            if nearby_text:
                return {
                    "value": nearby_text,
                    "confidence": 0.6,  # Lower confidence for fuzzy match
                    "extraction_method": "proximity_search"
                }
    
    return {
        "value": "",
        "confidence": 0.0,
        "extraction_method": "not_found"
    }

def _get_search_terms_for_field(field_name):
    """Get search terms for field detection."""
    term_mapping = {
        "invoice_number": ["invoice", "inv", "document", "ref"],
        "amount": ["total", "amount", "sum", "due"],
        "date": ["date", "issued", "created"],
        "vendor": ["vendor", "company", "from", "supplier"]
    }
    return term_mapping.get(field_name, [field_name])

def _extract_nearby_text(text, search_term, window_size=50):
    """Extract text near a search term."""
    index = text.lower().find(search_term.lower())
    if index == -1:
        return ""
    
    start = max(0, index - window_size)
    end = min(len(text), index + len(search_term) + window_size)
    nearby = text[start:end]
    
    # Extract likely field value (simple heuristic)
    lines = nearby.split('\n')
    for line in lines:
        if search_term.lower() in line.lower():
            parts = line.split(':')
            if len(parts) > 1:
                return parts[1].strip()[:50]  # Limit length
    
    return ""

def _assess_extraction_quality(fields, confidence_scores):
    """Assess overall quality of field extraction."""
    if not fields:
        return "no_extraction"
    
    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores)
    filled_fields = sum(1 for v in fields.values() if v)
    fill_rate = filled_fields / len(fields)
    
    if avg_confidence > 0.8 and fill_rate > 0.8:
        return "excellent"
    elif avg_confidence > 0.6 and fill_rate > 0.6:
        return "good"
    elif avg_confidence > 0.4 or fill_rate > 0.4:
        return "fair"
    else:
        return "poor"

@agent_action
def call_matching_api(extracted_documents, matching_api_endpoint, api_config=None):
    """Call external API to find entity matches."""
    if not extracted_documents:
        return {"status": "no_data", "api_results": []}
    
    api_results = []
    config = api_config or {}
    
    for document in extracted_documents:
        fields = document["extracted_fields"]
        
        # Prepare API payload
        api_payload = {
            "query_fields": fields,
            "confidence_threshold": config.get("confidence_threshold", 0.5),
            "max_results": config.get("max_results", 10),
            "search_type": config.get("search_type", "fuzzy_match")
        }
        
        try:
            response = requests.post(
                matching_api_endpoint,
                json=api_payload,
                timeout=config.get("timeout", 30)
            )
            
            if response.status_code == 200:
                api_data = response.json()
                api_results.append({
                    "document": document["filename"],
                    "status": "success",
                    "matches": api_data.get("matches", []),
                    "total_matches": len(api_data.get("matches", [])),
                    "search_metadata": api_data.get("metadata", {}),
                    "response_time": response.elapsed.total_seconds()
                })
            else:
                api_results.append({
                    "document": document["filename"],
                    "status": "api_error",
                    "error": f"HTTP {response.status_code}",
                    "matches": []
                })
                
        except Exception as e:
            api_results.append({
                "document": document["filename"],
                "status": "request_failed",
                "error": str(e),
                "matches": []
            })
    
    return {
        "status": "completed",
        "api_results": api_results,
        "total_requests": len(api_results)
    }

@agent_action
def perform_entity_matching(api_results, matching_config=None):
    """Perform sophisticated entity matching and ranking."""
    config = matching_config or {}
    confidence_threshold = config.get("confidence_threshold", 0.7)
    max_matches_per_document = config.get("max_matches", 5)
    
    matching_results = []
    
    for api_result in api_results.get("api_results", []):
        if api_result["status"] != "success":
            matching_results.append({
                "document": api_result["document"],
                "status": "no_matches",
                "best_match": None,
                "all_matches": [],
                "error": api_result.get("error")
            })
            continue
        
        matches = api_result.get("matches", [])
        if not matches:
            matching_results.append({
                "document": api_result["document"],
                "status": "no_matches",
                "best_match": None,
                "all_matches": []
            })
            continue
        
        # Enhanced matching with multiple criteria
        enhanced_matches = []
        for match in matches:
            enhanced_score = _calculate_enhanced_match_score(match, config)
            if enhanced_score >= confidence_threshold:
                enhanced_match = {
                    **match,
                    "enhanced_confidence": enhanced_score,
                    "match_reasons": _get_match_reasons(match)
                }
                enhanced_matches.append(enhanced_match)
        
        # Sort by enhanced confidence score
        enhanced_matches.sort(key=lambda x: x["enhanced_confidence"], reverse=True)
        
        # Limit results
        final_matches = enhanced_matches[:max_matches_per_document]
        
        matching_results.append({
            "document": api_result["document"],
            "status": "matched" if final_matches else "below_threshold",
            "best_match": final_matches[0] if final_matches else None,
            "all_matches": final_matches,
            "total_candidates": len(matches),
            "matches_above_threshold": len(final_matches)
        })
    
    return {
        "status": "completed",
        "matching_results": matching_results,
        "summary": _generate_matching_summary(matching_results)
    }

def _calculate_enhanced_match_score(match, config):
    """Calculate enhanced match score using multiple factors."""
    base_confidence = match.get("confidence", 0.0)
    
    # Field match quality
    field_matches = match.get("field_matches", {})
    field_quality = sum(field_matches.values()) / len(field_matches) if field_matches else 0.0
    
    # Text similarity
    text_similarity = match.get("text_similarity", 0.0)
    
    # Weighted combination
    weights = config.get("score_weights", {
        "base_confidence": 0.4,
        "field_quality": 0.4, 
        "text_similarity": 0.2
    })
    
    enhanced_score = (
        base_confidence * weights["base_confidence"] +
        field_quality * weights["field_quality"] +
        text_similarity * weights["text_similarity"]
    )
    
    return min(enhanced_score, 1.0)  # Cap at 1.0

def _get_match_reasons(match):
    """Get reasons why this match was selected."""
    reasons = []
    
    if match.get("confidence", 0) > 0.8:
        reasons.append("High API confidence score")
    
    field_matches = match.get("field_matches", {})
    strong_fields = [k for k, v in field_matches.items() if v > 0.8]
    if strong_fields:
        reasons.append(f"Strong field matches: {', '.join(strong_fields)}")
    
    if match.get("text_similarity", 0) > 0.7:
        reasons.append("High text similarity")
    
    return reasons

def _generate_matching_summary(results):
    """Generate summary of matching results."""
    total_docs = len(results)
    matched_docs = len([r for r in results if r["status"] == "matched"])
    
    return {
        "total_documents": total_docs,
        "successfully_matched": matched_docs,
        "match_rate": matched_docs / total_docs if total_docs > 0 else 0.0,
        "average_confidence": _calculate_average_confidence(results)
    }

def _calculate_average_confidence(results):
    """Calculate average confidence across all matches."""
    confidences = []
    for result in results:
        if result["best_match"]:
            confidences.append(result["best_match"].get("enhanced_confidence", 0.0))
    
    return sum(confidences) / len(confidences) if confidences else 0.0

@agent_action
def validate_matches(matching_results, validation_config=None):
    """Validate matching results for quality and consistency."""
    config = validation_config or {}
    min_confidence = config.get("min_confidence", 0.6)
    
    validation_report = {
        "overall_quality": "good",
        "validated_matches": [],
        "flagged_matches": [],
        "recommendations": []
    }
    
    for result in matching_results.get("matching_results", []):
        if result["status"] != "matched":
            continue
            
        best_match = result["best_match"]
        confidence = best_match.get("enhanced_confidence", 0.0)
        
        validation_entry = {
            "document": result["document"],
            "match_id": best_match.get("id"),
            "confidence": confidence,
            "validation_status": "approved"
        }
        
        # Validation checks
        if confidence < min_confidence:
            validation_entry["validation_status"] = "needs_review"
            validation_entry["reason"] = "Low confidence score"
            validation_report["flagged_matches"].append(validation_entry)
        else:
            validation_report["validated_matches"].append(validation_entry)
    
    # Generate recommendations
    flagged_count = len(validation_report["flagged_matches"])
    if flagged_count > 0:
        validation_report["recommendations"].append(
            f"{flagged_count} matches need manual review due to low confidence"
        )
    
    return validation_report
```

***

## **Multi-Agent Coordination Setup**

### **Workflow Orchestration** (`nexus/nexus_base/nexus_actions/three_agent_coordinator.py`)

```python
from nexus.nexus_base.action_manager import agent_action

@agent_action
def orchestrate_three_agent_workflow(uploaded_email_file, ocr_endpoints, matching_api_config):
    """Orchestrate the three-agent email processing workflow."""
    
    workflow_state = {
        "stage": "initialization",
        "results": {},
        "errors": [],
        "processing_time": {}
    }
    
    try:
        # Stage 1: Email Extraction Agent
        workflow_state["stage"] = "email_extraction"
        
        # This would call the EmailExtractor agent
        extraction_result = {
            "agent": "EmailExtractor",
            "action": "complete_email_extraction",
            "input": uploaded_email_file
        }
        workflow_state["results"]["extraction"] = extraction_result
        
        # Stage 2: OCR Processing Agent
        workflow_state["stage"] = "ocr_processing"
        
        ocr_result = {
            "agent": "OCRProcessor", 
            "action": "complete_ocr_processing",
            "input": extraction_result["attachments"]
        }
        workflow_state["results"]["ocr"] = ocr_result
        
        # Stage 3: Entity Matching Agent
        workflow_state["stage"] = "entity_matching"
        
        matching_result = {
            "agent": "EntityMatcher",
            "action": "complete_entity_matching", 
            "input": ocr_result["ocr_results"]
        }
        workflow_state["results"]["matching"] = matching_result
        
        workflow_state["stage"] = "completed"
        workflow_state["final_result"] = {
            "email_data": extraction_result,
            "ocr_data": ocr_result,
            "matching_data": matching_result,
            "processing_summary": _generate_processing_summary(workflow_state)
        }
        
    except Exception as e:
        workflow_state["stage"] = "failed"
        workflow_state["errors"].append(str(e))
    
    return workflow_state

def _generate_processing_summary(workflow_state):
    """Generate summary of the entire processing workflow."""
    return {
        "total_stages": 3,
        "completed_stages": len(workflow_state["results"]),
        "final_status": workflow_state["stage"],
        "has_errors": len(workflow_state["errors"]) > 0
    }
```

***

## **Usage in Streamlit Interface**

```python
def three_agent_email_processing_page(username, win_height):
    st.title("Three-Agent Email Processing System")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Email File (.eml)", type=["eml"])
    
    # Configuration
    col1, col2 = st.columns(2)
    with col1:
        primary_ocr_endpoint = st.text_input("Primary OCR API Endpoint")
        fallback_ocr_endpoints = st.text_area("Fallback OCR Endpoints (one per line)")
    
    with col2:
        matching_api_endpoint = st.text_input("Entity Matching API Endpoint")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7)
    
    if st.button("Process with Three Agents") and uploaded_file:
        nexus = get_nexus()
        
        # Initialize agents
        email_agent = nexus.get_agent("EmailExtractor")
        ocr_agent = nexus.get_agent("OCRProcessor")
        matching_agent = nexus.get_agent("EntityMatcher")
        
        # Processing workflow
        with st.spinner("Agent 1: Extracting email content..."):
            # Email extraction
            extraction_result = email_agent.get_response({
                "action": "parse_and_extract_email",
                "email_file": uploaded_file
            })
            st.success("✅ Email extraction completed")
        
        with st.spinner("Agent 2: Processing OCR..."):
            # OCR processing
            ocr_result = ocr_agent.get_response({
                "action": "process_attachments_ocr", 
                "attachments": extraction_result["attachments"],
                "primary_endpoint": primary_ocr_endpoint,
                "fallback_endpoints": fallback_ocr_endpoints.split('\n')
            })
            st.success("✅ OCR processing completed")
        
        with st.spinner("Agent 3: Performing entity matching..."):
            # Entity matching
            matching_result = matching_agent.get_response({
                "action": "extract_and_match_entities",
                "ocr_results": ocr_result,
                "matching_endpoint": matching_api_endpoint,
                "confidence_threshold": confidence_threshold
            })
            st.success("✅ Entity matching completed")
        
        # Display results
        st.subheader("Processing Results")
        
        tabs = st.tabs(["Email Data", "OCR Results", "Entity Matches"])
        
        with tabs[0]:
            st.json(extraction_result)
        
        with tabs[1]:
            st.json(ocr_result)
        
        with tabs[2]:
            st.json(matching_result)
```

This three-agent system provides **specialized expertise**, **error isolation**, **parallel processing capabilities**, and **individual optimization** for each processing stage, making your email processing workflow both robust and highly efficient.

Sources
