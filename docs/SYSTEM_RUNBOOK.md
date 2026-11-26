# Production Runbook & System Operations Guide
**Project:** Hybrid RAG + Fine-Tuned Foundation LLM + Exact KYC System
**Target:** Existing_LLM_Project
**Status:** Verified & Operational

---

## 1. Quick Start Commands

### A. Environment Activation
```bash
cd /Users/DF/Documents/llm/Existing_LLM_Project
source /Users/DF/Documents/llm/Custom_LLM_Project/venv/bin/activate
```

### B. Run Automated Verification Tests
```bash
python3 08_tests/run_all_tests.py
```
*(Executes exact KYC SQL lookups, Hybrid RAG RRF search, and SQL injection defense tests).*

### C. Launch Unified FastAPI Gateway (Port 8000)
```bash
python3 07_serving/api_service.py
```
* Interactive Swagger Docs: `http://localhost:8000/docs`
* Health Check: `http://localhost:8000/health`
* Direct KYC Lookup: `http://localhost:8000/v1/kyc/P00001`
* Unified Chat: `POST http://localhost:8000/v1/chat`

### D. Launch Streamlit Interactive UI (Port 8501)
```bash
streamlit run 07_serving/app_ui.py
```
* Automatically opens in browser at `http://localhost:8501`.
* Allows testing as `officer` (unmasked) or `guest` (masked PII).

---

## 2. API Usage Examples

### 1. Exact KYC Lookup Query (SQL Guaranteed)
```bash
curl -X POST http://localhost:8000/v1/chat \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the Aadhaar number and parents of person P00001?",
       "actor_role": "officer"
     }'
```
**Response:**
```json
{
  "route": "EXACT_KYC",
  "response": "The father of Arjun Sharma (P00001) is Rajesh Sharma and the mother is Sunita Sharma. The Aadhaar number is 997228186620.",
  "verified_sql_record": {
    "person_id": "P00001",
    "name": "Arjun Sharma",
    "father_name": "Rajesh Sharma",
    "mother_name": "Sunita Sharma",
    "age": 47,
    "aadhaar_number": "997228186620",
    "pan_number": "RBLXA4104K"
  },
  "citations": [],
  "grounding_used": true
}
```

### 2. Document Policy Query (Hybrid RAG)
```bash
curl -X POST http://localhost:8000/v1/chat \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What foundation models are supported in our architecture?"
     }'
```

### 3. Conversational Query (WhatsApp Tone)
```bash
curl -X POST http://localhost:8000/v1/chat \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Hi! Where should we order vegetarian food for lunch?"
     }'
```

---

## 3. Security & Compliance Architecture

1. **Zero Raw KYC in LLM Weights:**  
   Fine-tuning is conducted strictly on `01_data/clean_conversational_train.jsonl` (WhatsApp chats). Citizen KYC records live solely inside `01_data/kyc_index.sqlite`.
2. **SQL Injection Neutralization:**  
   All database queries use parameterized inputs (`?` placeholders). String interpolation is strictly prohibited.
3. **Role-Based Field Masking:**  
   Non-officer roles (`guest`, `public`) receive masked Aadhaar (`XXXX-XXXX-6620`) and PAN (`RBXXXXXK`).
4. **Audit Logging:**  
   Every KYC access is immutably logged to `kyc_audit_logs` with UTC timestamp, role, searched ID, and outcome.

