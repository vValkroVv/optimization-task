def doc_to_target(doc):
    if doc["error_type"] == "0":
        return "No violations"
    return doc["error_type"]
