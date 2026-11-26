"""
Phase 5: High-Level KYC Retrieval Service with Field Masking and Authorization.
Translates structured database rows into verified, zero-hallucination context.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "02_kyc_db"))
from db_manager import db_manager

# Roles and Permissions
# 'admin'/'officer': Full unmasked access
# 'public'/'guest': Masked Aadhaar/PAN (e.g. XXXXXXXX1234)
ROLE_PERMISSIONS = {
    "admin": {"unmasked": True, "allowed_queries": ["all"]},
    "officer": {"unmasked": True, "allowed_queries": ["all"]},
    "guest": {"unmasked": False, "allowed_queries": ["name", "age", "parents"]}
}

def mask_sensitive_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Applies strict data masking to sensitive government identifiers."""
    masked = dict(record)
    if "aadhaar_number" in masked and masked["aadhaar_number"]:
        raw_aadhaar = str(masked["aadhaar_number"])
        masked["aadhaar_number"] = "XXXX-XXXX-" + raw_aadhaar[-4:]
    if "pan_number" in masked and masked["pan_number"]:
        raw_pan = str(masked["pan_number"])
        masked["pan_number"] = raw_pan[:2] + "XXXXX" + raw_pan[-1]
    return masked

class KYCService:
    def __init__(self):
        self.db = db_manager

    def lookup_record(
        self,
        identifier: str,
        query_type: str = "auto",
        actor_role: str = "officer"
    ) -> Dict[str, Any]:
        """
        Retrieves authoritative citizen data with record-level security.
        """
        identifier = identifier.strip()
        record: Optional[Dict[str, Any]] = None

        # Auto-detect identifier type if query_type is 'auto'
        if query_type == "auto":
            if identifier.upper().startswith("P") and identifier[1:].isdigit():
                query_type = "person_id"
            elif len(identifier.replace(" ", "")) == 12 and identifier.replace(" ", "").isdigit():
                query_type = "aadhaar"
            elif len(identifier) == 10 and identifier[:5].isalpha() and identifier[5:9].isdigit() and identifier[9].isalpha():
                query_type = "pan"
            else:
                query_type = "name"

        if query_type == "person_id":
            record = self.db.get_citizen_by_id(identifier, actor_role=actor_role)
        elif query_type == "aadhaar":
            record = self.db.get_citizen_by_aadhaar(identifier, actor_role=actor_role)
        elif query_type == "pan":
            record = self.db.get_citizen_by_pan(identifier, actor_role=actor_role)
        elif query_type == "name":
            results = self.db.get_citizen_by_name(identifier, actor_role=actor_role)
            record = results[0] if results else None

        if not record:
            return {
                "found": False,
                "message": f"No citizen record found for identifier '{identifier}' (Searched as {query_type}).",
                "record": None
            }

        # Check Role-Based Access Control
        perms = ROLE_PERMISSIONS.get(actor_role, ROLE_PERMISSIONS["guest"])
        if not perms["unmasked"]:
            final_record = mask_sensitive_fields(record)
        else:
            final_record = record

        return {
            "found": True,
            "query_type": query_type,
            "record": final_record,
            "formatted_summary": (
                f"Citizen Record [ID: {final_record.get('person_id')}]:\n"
                f"• Name: {final_record.get('name')}\n"
                f"• Father's Name: {final_record.get('father_name')}\n"
                f"• Mother's Name: {final_record.get('mother_name')}\n"
                f"• Age: {final_record.get('age')} years\n"
                f"• Aadhaar Number: {final_record.get('aadhaar_number')}\n"
                f"• PAN Number: {final_record.get('pan_number')}"
            )
        }

kyc_service = KYCService()

if __name__ == "__main__":
    test_res = kyc_service.lookup_record("P00001", actor_role="officer")
    print(test_res["formatted_summary"])
    print("\nPublic Role Masked:")
    test_masked = kyc_service.lookup_record("P00001", actor_role="guest")
    print(test_masked["formatted_summary"])

