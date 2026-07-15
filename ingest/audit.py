from __future__ import annotations
import json
import hashlib
import time
from pathlib import Path
from core.config import Settings, get_settings
from core.types import PIISpan

class PIIAuditLog:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.log_path = Path(self.settings.pii_audit_log_path)

    def record(
        self,
        tenant_id: str,
        doc_id: str,
        chunk_id: str,
        text: str,
        spans: list[PIISpan]
    ) -> None:
        if not spans:
            return

        # Ensure parent folder exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = time.time()
        
        with self.log_path.open("a", encoding="utf-8") as f:
            for span in spans:
                record_data = {
                    "tenant_id": tenant_id,
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "type": span.type,
                    "start": span.start,
                    "end": span.end,
                    "timestamp": timestamp,
                }
                
                if self.settings.pii_audit_value_hash:
                    # Salt and hash transiently sliced values
                    raw_val = text[span.start:span.end]
                    salt = self.settings.pii_audit_hash_salt or ""
                    salted = f"{salt}{raw_val}".encode("utf-8")
                    value_hash = hashlib.sha256(salted).hexdigest()[:16]
                    record_data["value_hash"] = value_hash
                    
                f.write(json.dumps(record_data, ensure_ascii=False) + "\n")
