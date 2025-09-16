form_fields = []
    
    for page_num, page in enumerate(doc.pages, 1):
        for field_idx, field in enumerate(page.form.fields):
            if field.key and field.value:
                key = field.key.text.strip() if field.key.text else None
                value = field.value.text.strip() if field.value.text else None
                
                if key:  # Only add if key is not empty
                    field_data = {
                        "field_id": f"page_{page_num}_field_{field_idx}",
                        "key": key,
                        "value": value,
                        "page_number": page_num,
                        "field_index": field_idx,
                        "key_bbox": field.key.geometry.boundingBox.__dict__ if field.key.geometry else None,
                        "value_bbox": field.value.geometry.boundingBox.__dict__ if field.value.geometry else None,
                        "key_polygon": [{"x": p.x, "y": p.y} for p in field.key.geometry.polygon] if field.key.geometry else None,
                        "value_polygon": [{"x": p.x, "y": p.y} for p in field.value.geometry.polygon] if field.value.geometry else None,
                        "key_confidence": getattr(field.key, 'confidence', 0.0),
                        "value_confidence": getattr(field.value, 'confidence', 0.0),
                        "created_at": None,  # You can add timestamp if needed
                        "processed": False   # Flag for post-processing tracking
                    }
                    
                    form_fields.append(field_data)
