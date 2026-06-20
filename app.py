from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import uvicorn
import hashlib
import time
from datetime import datetime
import os
import json

# Import core modules
from ai_core import AIAgent
from config import logger, COLLECTION_PRESCRIPTIONS, COLLECTION_ALLERGIES

# Initialize global AI Agent instance
ai_agent: Optional[AIAgent] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown lifespan events.
    Ensures MongoDB connection is established on boot and closed cleanly on termination.
    """
    global ai_agent
    logger.info("Initializing Healthcare AI Agent server startup lifespan...")
    try:
        ai_agent = AIAgent()
        logger.info("Healthcare AI Agent initialized successfully with MongoDB.")
    except Exception as e:
        logger.error(f"Failed to initialize AIAgent / MongoDB: {str(e)}. Running in local datastore mode.")
        ai_agent = None
    yield
    if ai_agent:
        ai_agent.close()
        logger.info("Healthcare AI Agent server shutdown complete.")

# Initialize FastAPI App with Metadata
app = FastAPI(
    title="Healthcare AI Agent Core API",
    description="REST API interface exposing Clinical Duplicate Detection, Drug Interactions, Allergy Screening, Alternative Recommendations, and Prescribing Pattern Anomalies.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS Middleware
# Allows React/Vue/Vite/HTML frontends running on other local ports to fetch resources
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# CRYPTOGRAPHY HELPER (Python Implementation)
# ============================================
HEX_CHARS = '0123456789abcdef'

def get_shift(doctor_id: str) -> int:
    doctor_id = str(doctor_id)
    total_sum = 0
    for char in doctor_id:
        if char.isdigit():
            total_sum += int(char)
    if total_sum == 0:
        hash_val = 0
        for char in doctor_id:
            hash_val = ord(char) + ((hash_val << 5) - hash_val)
            hash_val = (hash_val & 0xFFFFFFFF)
            if hash_val & 0x80000000:
                hash_val = hash_val - 0x100000000
        total_sum = abs(hash_val)
    return (total_sum % 14) + 1

def shift_hex(hex_str: str, shift: int) -> str:
    sb = []
    for char in hex_str.lower():
        if char in HEX_CHARS:
            idx = HEX_CHARS.index(char)
            new_idx = (idx + shift) % 16
            if new_idx < 0:
                new_idx += 16
            sb.append(HEX_CHARS[new_idx])
        else:
            sb.append(char)
    return "".join(sb)

def sha256_hash(input_str: str) -> str:
    return hashlib.sha256(input_str.encode('utf-8')).hexdigest()

def sign(hash_str: str, doctor_id: str) -> str:
    shift = get_shift(doctor_id)
    return shift_hex(hash_str, shift)

def decrypt(signature: str, doctor_id: str) -> str:
    shift = get_shift(doctor_id)
    return shift_hex(signature, -shift)

def json_stringify_rx(rx: dict) -> str:
    meds_list = []
    for med in rx.get("medicines", []):
        meds_list.append(f'{{"name":"{med.get("name","")}","interval":"{med.get("interval","")}"}}')
    meds_str = "[" + ",".join(meds_list) + "]"
    
    date_val = rx.get("date", "")
    if isinstance(date_val, datetime):
        date_val = date_val.strftime("%Y-%m-%d")
    elif not isinstance(date_val, str):
        date_val = str(date_val)
        
    return (
        f'{{"id":"{rx.get("id","")}",'
        f'"doctorName":"{rx.get("doctorName","")}",'
        f'"hospitalName":"{rx.get("hospitalName","")}",'
        f'"patientName":"{rx.get("patientName","")}",'
        f'"disease":"{rx.get("disease","")}",'
        f'"date":"{date_val}",'
        f'"time":"{rx.get("time","")}",'
        f'"medicines":{meds_str}}}'
    )

# ============================================
# DATABASE PIPELINE & LOCAL DATASORES
# ============================================
DB_FILE = os.path.join(os.path.dirname(__file__), "prescriptions_db.json")

local_consultations = []
local_doctors = []
local_activity_logs = []

def read_local_db() -> list:
    if not os.path.exists(DB_FILE):
        demo_rx = {
            "id": "RX-9921",
            "doctorName": "Marcus Vance",
            "hospitalName": "OmniHealth Clinic",
            "patientName": "Elena Vance",
            "disease": "Asthma Treatment",
            "date": "2026-06-19",
            "time": "14:20",
            "medicines": [
                { "name": "Albuterol 90mcg Inhaler", "interval": "2 puffs every 4 hours" },
                { "name": "Prednisone 10mg", "interval": "Once daily (With Breakfast)" }
            ],
            "signature": "",
            "doctorSignId": "99281",
            "isDispensed": False
        }
        payload_str = json_stringify_rx(demo_rx)
        rx_hash = sha256_hash(payload_str)
        demo_rx["signature"] = sign(rx_hash, "99281")
        write_local_db([demo_rx])
        return [demo_rx]
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def write_local_db(data: list):
    try:
        with open(DB_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to write local JSON DB: {str(e)}")

def serialize_doc(doc: dict) -> Optional[dict]:
    if doc is None:
        return None
    doc_copy = dict(doc)
    if "_id" in doc_copy:
        doc_copy["_id"] = str(doc_copy["_id"])
    for key, value in doc_copy.items():
        if isinstance(value, datetime):
            if value.hour == 0 and value.minute == 0 and value.second == 0:
                doc_copy[key] = value.strftime("%Y-%m-%d")
            else:
                doc_copy[key] = value.isoformat()
        elif isinstance(value, list):
            doc_copy[key] = [serialize_doc(item) if isinstance(item, dict) else item for item in value]
    return doc_copy

def serialize_list(docs: list) -> list:
    return [serialize_doc(doc) for doc in docs]

async def log_activity(event_type: str, patient_name: str, actor_id: str, details: str):
    timestamp_ms = int(time.time() * 1000)
    hash_input = f"{timestamp_ms}-{event_type}-{patient_name}-{actor_id}-{details}"
    log_hash = sha256_hash(hash_input)
    
    log_entry = {
        "timestamp": datetime.utcnow(),
        "eventType": event_type,
        "patientName": patient_name,
        "actorId": actor_id,
        "details": details,
        "hash": log_hash
    }
    
    logger.info(f"[ACTIVITY LOG] [{event_type}] Actor: {actor_id}, Patient: {patient_name} - {details}")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            db["activity_logs"].insert_one(log_entry.copy())
        except Exception as e:
            logger.error(f"Failed to save activity log in MongoDB: {str(e)}")
            local_activity_logs.append(log_entry)
    else:
        local_activity_logs.append(log_entry)

# ============================================
# PYDANTIC SCHEMAS FOR DATA VALIDATION
# ============================================
class AuditRequest(BaseModel):
    patient_id: str = Field(..., description="ID of the patient, e.g., 'Priya_123'", example="Priya_123")
    doctor_id: str = Field(..., description="ID of the prescribing physician, e.g., 'Dr_Kumar_456'", example="Dr_Kumar_456")
    new_medicine: str = Field(..., description="Name of the medicine, e.g., 'Metformin'", example="Metformin")
    new_dosage: str = Field(..., description="Dosage details, e.g., '500mg'", example="500mg")
    disease: Optional[str] = Field(None, description="Diagnosed disease (needed for alternative recommendations)", example="Type 2 Diabetes")

class DuplicateRequest(BaseModel):
    patient_id: str = Field(..., example="Priya_123")
    new_medicine: str = Field(..., example="Metformin")
    new_dosage: str = Field(..., example="500mg")

class InteractionRequest(BaseModel):
    patient_id: str = Field(..., example="Priya_123")
    new_medicine: str = Field(..., example="Metformin")
    new_dosage: str = Field(..., example="500mg")

class AllergyRequest(BaseModel):
    patient_id: str = Field(..., example="Priya_123")
    new_medicine: str = Field(..., example="Penicillin")

class RecommendationRequest(BaseModel):
    patient_id: str = Field(..., example="Priya_123")
    disease: str = Field(..., example="Type 2 Diabetes")
    current_medicine: str = Field(..., example="Metformin")

class PrescriptionMedicine(BaseModel):
    name: str
    interval: str

class PrescriptionInput(BaseModel):
    id: str
    doctorName: str
    hospitalName: str
    patientName: str
    disease: str
    date: str
    time: str
    medicines: List[PrescriptionMedicine]
    doctorSignId: Optional[str] = "889218"

class DispenseInput(BaseModel):
    id: str

class ConsultationInput(BaseModel):
    patientName: str
    patientId: Optional[str] = None

class AcceptRejectInput(BaseModel):
    id: str

class DoctorLoginInput(BaseModel):
    doctorMobile: str

class DoctorRegisterInput(BaseModel):
    name: str
    hospitalName: str
    doctorMobile: str

class VerifyScanInput(BaseModel):
    raw_payload: str
    signature: str
    timestamp: str

# ============================================
# API ENDPOINTS
# ============================================
@app.get("/", tags=["General"])
async def root():
    """Welcome and API metadata status check."""
    return {
        "status": "online",
        "api_name": "Healthcare AI Agent Core Service",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# --- Core Prescription Endpoints ---
@app.get("/api/prescriptions", tags=["Prescription Management"])
async def get_prescriptions(patient: str = "Elena Vance"):
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            query = {"patientName": {"$regex": f"^{patient}$", "$options": "i"}}
            cursor = db[COLLECTION_PRESCRIPTIONS].find(query)
            data = list(cursor)
            return serialize_list(data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        data = read_local_db()
        filtered = [rx for rx in data if rx.get("patientName", "").lower() == patient.lower()]
        return filtered

@app.post("/api/prescriptions", tags=["Prescription Management"])
async def create_prescription(rx: PrescriptionInput):
    rx_dict = rx.dict()
    
    # Calculate signature server-side
    payload_str = json_stringify_rx(rx_dict)
    rx_hash = sha256_hash(payload_str)
    
    doctor_sign_id = rx_dict.get("doctorSignId") or "889218"
    rx_dict["signature"] = sign(rx_hash, doctor_sign_id)
    rx_dict["isDispensed"] = False
    
    logger.info(f"Issuing prescription {rx_dict['id']}. Hash: {rx_hash}. Signed signature: {rx_dict['signature']}")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            rx_mongo = rx_dict.copy()
            if "date" in rx_mongo and isinstance(rx_mongo["date"], str):
                try:
                    rx_mongo["date"] = datetime.fromisoformat(rx_mongo["date"])
                except Exception:
                    pass
            db[COLLECTION_PRESCRIPTIONS].insert_one(rx_mongo)
            
            await log_activity(
                'CREATE_PRESCRIPTION', 
                rx_dict["patientName"], 
                doctor_sign_id, 
                f"Doctor {rx_dict['doctorName']} created prescription {rx_dict['id']} for {rx_dict['disease']}."
            )
            return serialize_doc(rx_mongo)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        data = read_local_db()
        data.append(rx_dict)
        write_local_db(data)
        await log_activity(
            'CREATE_PRESCRIPTION', 
            rx_dict["patientName"], 
            doctor_sign_id, 
            f"Doctor {rx_dict['doctorName']} created prescription {rx_dict['id']} for {rx_dict['disease']}."
        )
        return rx_dict

@app.post("/api/prescriptions/dispense", tags=["Prescription Management"])
async def dispense_prescription(input_data: DispenseInput):
    rx_id = input_data.id
    logger.info(f"Attempting to dispense prescription ID: {rx_id}")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            updated = db[COLLECTION_PRESCRIPTIONS].find_one_and_update(
                {"id": rx_id},
                {"$set": {"isDispensed": True}},
                return_document=True
            )
            if updated:
                await log_activity(
                    'DISPENSE_PRESCRIPTION', 
                    updated.get("patientName", "Unknown"), 
                    'Pharmacy', 
                    f"Pharmacy dispensed medications and burned token for prescription {rx_id}."
                )
                return serialize_doc(updated)
            else:
                raise HTTPException(status_code=404, detail="Prescription not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        data = read_local_db()
        for rx in data:
            if rx.get("id") == rx_id:
                rx["isDispensed"] = True
                write_local_db(data)
                await log_activity(
                    'DISPENSE_PRESCRIPTION', 
                    rx.get("patientName", "Unknown"), 
                    'Pharmacy', 
                    f"Pharmacy dispensed medications and burned token for prescription {rx_id}."
                )
                return rx
        raise HTTPException(status_code=404, detail="Prescription not found")

@app.get("/api/prescriptions/{rx_id}", tags=["Prescription Management"])
async def get_prescription_by_id(rx_id: str):
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            data = db[COLLECTION_PRESCRIPTIONS].find_one({"id": rx_id})
            if data:
                await log_activity(
                    'SCAN_PHARMACY', 
                    data.get("patientName", "Unknown"), 
                    'Pharmacy', 
                    f"Pharmacy scanned and retrieved prescription details for {rx_id}."
                )
                return serialize_doc(data)
            else:
                raise HTTPException(status_code=404, detail="Prescription not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        data = read_local_db()
        rx = next((r for r in data if r.get("id") == rx_id), None)
        if rx:
            await log_activity(
                'SCAN_PHARMACY', 
                rx.get("patientName", "Unknown"), 
                'Pharmacy', 
                f"Pharmacy scanned and retrieved prescription details for {rx_id}."
            )
            return rx
        raise HTTPException(status_code=404, detail="Prescription not found")

@app.post("/api/prescriptions/verify-scan", tags=["Prescription Management"])
async def verify_scan(data: VerifyScanInput):
    try:
        raw_payload = data.raw_payload
        signature = data.signature
        timestamp = data.timestamp
        
        # Calculate local SHA-256 hash of the incoming text data
        local_hash = sha256_hash(raw_payload)
        
        parts = raw_payload.split("|")
        if len(parts) < 8:
            logger.warning(f"Invalid payload format parsed. Splitting by '|' returned only {len(parts)} parts.")
            return {"verified": False, "error": "Invalid payload format"}
            
        rx_id = parts[0]
        doctor_sign_id = parts[8] if len(parts) > 8 else "889218"
        
        logger.info(f"Verifying scan: rx_id={rx_id}, signature={signature}, doctor_sign_id={doctor_sign_id}")
        
        # Retrieve prescription from db
        db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
        db_rx = None
        if db is not None:
            db_rx = db[COLLECTION_PRESCRIPTIONS].find_one({"id": rx_id})
        else:
            local_data = read_local_db()
            db_rx = next((r for r in local_data if r.get("id") == rx_id), None)
            
        if not db_rx:
            logger.warning(f"Prescription with ID {rx_id} not found.")
            return {"verified": False, "error": "Prescription not found"}
            
        # Get signature stored inside MongoDB Atlas / local DB
        stored_signature = db_rx.get("signature")
        
        # Verify that the incoming signature matches the signature stored in MongoDB
        if signature != stored_signature:
            logger.warning(f"Verification fail: incoming signature '{signature}' does not match stored signature '{stored_signature}'")
            return {"verified": False, "error": "Signature mismatch"}
            
        # Verify hash integrity
        decrypted_hash = decrypt(signature, doctor_sign_id)
        if decrypted_hash != local_hash:
            # Let's also check if it matches the hash of json_stringify_rx(db_rx) in case of serialization formatting differences
            db_payload = json_stringify_rx(db_rx)
            db_hash = sha256_hash(db_payload)
            if decrypted_hash != db_hash:
                logger.warning(f"Decrypted hash verification failed: decrypted '{decrypted_hash}' vs local hash '{local_hash}' or db hash '{db_hash}'")
                return {"verified": False, "error": "Integrity check failed: payload modified after signing"}
                
        # Log successful scan
        await log_activity(
            'SCAN_PHARMACY', 
            db_rx.get("patientName", "Unknown"), 
            'Pharmacy', 
            f"Zero-trust cross-verification succeeded for prescription {rx_id}."
        )
        
        return {"verified": True, "prescription": serialize_doc(db_rx)}
    except Exception as e:
        logger.error(f"Error during verification route: {str(e)}")
        return {"verified": False, "error": str(e)}

# --- Consultation Management Endpoints ---
@app.post("/api/consultation/request", tags=["Consultation Management"])
async def create_consultation_request(input_data: ConsultationInput):
    patient_name = input_data.patientName
    patient_id = input_data.patientId or patient_name
    request_id = f"req_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    request_data = {
        "id": request_id,
        "patientName": patient_name,
        "patientId": patient_id,
        "status": "pending",
        "createdAt": datetime.utcnow()
    }
    
    logger.info(f"New consultation request created: {request_id} for patient: {patient_name} (ID: {patient_id})")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            db["consultation_requests"].insert_one(request_data)
            await log_activity(
                'SCAN_CONSULTATION', 
                patient_name, 
                'Doctor', 
                f"Doctor scanned session QR code and initiated connection request {request_id}."
            )
            return serialize_doc(request_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        local_consultations.append(request_data)
        await log_activity(
            'SCAN_CONSULTATION', 
            patient_name, 
            'Doctor', 
            f"Doctor scanned session QR code and initiated connection request {request_id}."
        )
        return serialize_doc(request_data)

@app.get("/api/consultation/status/{req_id}", tags=["Consultation Management"])
async def get_consultation_status(req_id: str):
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            req_doc = db["consultation_requests"].find_one({"id": req_id})
            if req_doc:
                return {"status": req_doc.get("status", "pending")}
            else:
                raise HTTPException(status_code=404, detail="Request not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        req_doc = next((r for r in local_consultations if r.get("id") == req_id), None)
        if req_doc:
            return {"status": req_doc.get("status", "pending")}
        raise HTTPException(status_code=404, detail="Request not found")

@app.get("/api/consultation/pending", tags=["Consultation Management"])
async def get_pending_consultation(patient: str, patientId: Optional[str] = None):
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    
    or_filters = [{"patientName": {"$regex": f"^{patient}$", "$options": "i"}}]
    if patientId:
        or_filters.append({"patientName": {"$regex": f"^{patientId}$", "$options": "i"}})
        or_filters.append({"patientId": {"$regex": f"^{patientId}$", "$options": "i"}})
        
    query = {
        "$or": or_filters,
        "status": "pending"
    }

    if db is not None:
        try:
            pending = db["consultation_requests"].find_one(
                query,
                sort=[("createdAt", -1)]
            )
            return serialize_doc(pending)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        filtered = []
        for r in local_consultations:
            if r.get("status") == "pending":
                c_name = r.get("patientName", "").lower()
                c_id = r.get("patientId", "").lower()
                match_name = (c_name == patient.lower()) or (patientId and c_name == patientId.lower())
                match_id = patientId and (c_id == patientId.lower())
                if match_name or match_id:
                    filtered.append(r)
        if filtered:
            return serialize_doc(filtered[-1])
        return None

@app.post("/api/consultation/accept", tags=["Consultation Management"])
async def accept_consultation(input_data: AcceptRejectInput):
    req_id = input_data.id
    logger.info(f"Accepting consultation request: {req_id}")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            updated = db["consultation_requests"].find_one_and_update(
                {"id": req_id},
                {"$set": {"status": "accepted"}},
                return_document=True
            )
            if updated:
                await log_activity(
                    'ACCEPT_ACCESS', 
                    updated.get("patientName", "Unknown"), 
                    'Patient', 
                    f"Patient accepted doctor connection request {req_id}."
                )
                return serialize_doc(updated)
            else:
                raise HTTPException(status_code=404, detail="Request not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        for r in local_consultations:
            if r.get("id") == req_id:
                r["status"] = "accepted"
                await log_activity(
                    'ACCEPT_ACCESS', 
                    r.get("patientName", "Unknown"), 
                    'Patient', 
                    f"Patient accepted doctor connection request {req_id}."
                )
                return serialize_doc(r)
        raise HTTPException(status_code=404, detail="Request not found")

@app.post("/api/consultation/reject", tags=["Consultation Management"])
async def reject_consultation(input_data: AcceptRejectInput):
    req_id = input_data.id
    logger.info(f"Rejecting consultation request: {req_id}")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            updated = db["consultation_requests"].find_one_and_update(
                {"id": req_id},
                {"$set": {"status": "rejected"}},
                return_document=True
            )
            if updated:
                await log_activity(
                    'REJECT_ACCESS', 
                    updated.get("patientName", "Unknown"), 
                    'Patient', 
                    f"Patient rejected doctor connection request {req_id}."
                )
                return serialize_doc(updated)
            else:
                raise HTTPException(status_code=404, detail="Request not found")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        for r in local_consultations:
            if r.get("id") == req_id:
                r["status"] = "rejected"
                await log_activity(
                    'REJECT_ACCESS', 
                    r.get("patientName", "Unknown"), 
                    'Patient', 
                    f"Patient rejected doctor connection request {req_id}."
                )
                return serialize_doc(r)
        raise HTTPException(status_code=404, detail="Request not found")

# --- Doctor Management Endpoints ---
@app.post("/api/doctor/login", tags=["Doctor Management"])
async def doctor_login(input_data: DoctorLoginInput):
    doctor_mobile = input_data.doctorMobile
    doc_id_str = str(doctor_mobile)
    logger.info(f"Doctor Sign-In Request. ID: {doc_id_str}")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            doc = db["doctor"].find_one({"doctor_id": doc_id_str})
            if doc:
                await log_activity(
                    'DOCTOR_LOGIN', 
                    'N/A', 
                    doc_id_str, 
                    f"Doctor {doc.get('name', 'Unknown')} signed in successfully."
                )
                return serialize_doc(doc)
            else:
                raise HTTPException(
                    status_code=404, 
                    detail="Doctor ID not found. Please register first."
                )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        doc = next((d for d in local_doctors if d.get("doctor_id") == doc_id_str), None)
        if doc:
            await log_activity(
                'DOCTOR_LOGIN', 
                'N/A', 
                doc_id_str, 
                f"Doctor {doc.get('name', 'Unknown')} signed in successfully."
            )
            return doc
        raise HTTPException(
            status_code=404, 
            detail="Doctor ID not found. Please register first."
        )

@app.post("/api/doctor/register", tags=["Doctor Management"])
async def doctor_register(input_data: DoctorRegisterInput):
    name = input_data.name
    hospital_name = input_data.hospitalName
    doctor_mobile = input_data.doctorMobile
    doc_id_str = str(doctor_mobile)
    
    logger.info(f"Doctor Registration Request. ID: {doc_id_str}, Name: {name}")
    
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            doc = db["doctor"].find_one({"doctor_id": doc_id_str})
            if not doc:
                doc = {
                    "doctor_id": doc_id_str,
                    "name": name,
                    "hospitalName": hospital_name,
                    "createdAt": datetime.utcnow()
                }
                db["doctor"].insert_one(doc.copy())
            else:
                db["doctor"].update_one(
                    {"doctor_id": doc_id_str},
                    {"$set": {"name": name, "hospitalName": hospital_name}}
                )
                doc["name"] = name
                doc["hospitalName"] = hospital_name
            
            await log_activity(
                'DOCTOR_REGISTER', 
                'N/A', 
                doc_id_str, 
                f"Doctor {name} registered from {hospital_name}."
            )
            return serialize_doc(doc)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        doc = next((d for d in local_doctors if d.get("doctor_id") == doc_id_str), None)
        if not doc:
            doc = {
                "doctor_id": doc_id_str,
                "name": name,
                "hospitalName": hospital_name,
                "createdAt": datetime.utcnow()
            }
            local_doctors.append(doc)
        else:
            doc["name"] = name
            doc["hospitalName"] = hospital_name
        
        await log_activity(
            'DOCTOR_REGISTER', 
            'N/A', 
            doc_id_str, 
            f"Doctor {name} registered from {hospital_name}."
        )
        return doc

# --- Ledger & Activity Logs ---
@app.get("/api/activity-logs", tags=["Ledger logs"])
async def get_activity_logs():
    db = ai_agent.mongo.db if (ai_agent and ai_agent.mongo) else None
    if db is not None:
        try:
            cursor = db["activity_logs"].find().sort("timestamp", -1)
            return serialize_list(list(cursor))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return list(reversed(local_activity_logs))

# --- Clinical AI Audit Endpoints ---
@app.post("/api/audit", tags=["Prescription Audit"])
async def audit_prescription(req: AuditRequest):
    """
    Perform a complete clinical audit for a new prescription.
    Combines: Duplicate Detection, Drug Interactions, Allergy Checks, Alternatives, and Doctor Pattern Analysis.
    """
    if not ai_agent:
        raise HTTPException(status_code=503, detail="AI Agent logic is currently unavailable.")
    try:
        logger.info(f"API Request: POST /api/audit patient={req.patient_id} med={req.new_medicine}")
        result = ai_agent.analyze_prescription(
            patient_id=req.patient_id,
            doctor_id=req.doctor_id,
            new_medicine=req.new_medicine,
            new_dosage=req.new_dosage,
            disease=req.disease
        )
        return result
    except Exception as e:
        logger.error(f"Error executing complete audit endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/duplicate-check", tags=["Clinical Controls"])
async def duplicate_check(req: DuplicateRequest):
    """Detect if the patient has had a duplicate prescription within the safety window (30 days)."""
    if not ai_agent:
        raise HTTPException(status_code=503, detail="AI Agent logic is currently unavailable.")
    try:
        logger.info(f"API Request: POST /api/duplicate-check patient={req.patient_id} med={req.new_medicine}")
        result = ai_agent.detect_duplicate_medicine(
            patient_id=req.patient_id,
            new_medicine=req.new_medicine,
            new_dosage=req.new_dosage
        )
        return result
    except Exception as e:
        logger.error(f"Error executing duplicate-check endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interaction-check", tags=["Clinical Controls"])
async def interaction_check(req: InteractionRequest):
    """Check drug interactions and duplicates against the patient's active drug list using Cerebras API."""
    if not ai_agent:
        raise HTTPException(status_code=503, detail="AI Agent logic is currently unavailable.")
    try:
        logger.info(f"API Request: POST /api/interaction-check patient={req.patient_id} med={req.new_medicine}")
        result = ai_agent.check_drug_interactions(
            patient_id=req.patient_id,
            new_medicine=req.new_medicine,
            new_dosage=req.new_dosage
        )
        return result
    except Exception as e:
        logger.error(f"Error executing interaction-check endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/allergy-check", tags=["Clinical Controls"])
async def allergy_check(req: AllergyRequest):
    """Check if the prescribed medication triggers any patient allergies logged in MongoDB."""
    if not ai_agent:
        raise HTTPException(status_code=503, detail="AI Agent logic is currently unavailable.")
    try:
        logger.info(f"API Request: POST /api/allergy-check patient={req.patient_id} med={req.new_medicine}")
        result = ai_agent.check_allergy_conflict(
            patient_id=req.patient_id,
            new_medicine=req.new_medicine
        )
        return result
    except Exception as e:
        logger.error(f"Error executing allergy-check endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recommendations", tags=["AI Clinical Guidance"])
async def recommend_alternatives(req: RecommendationRequest):
    """Fetch ranked allergy-safe drug recommendations for a disease using Cerebras AI."""
    if not ai_agent:
        raise HTTPException(status_code=503, detail="AI Agent logic is currently unavailable.")
    try:
        logger.info(f"API Request: POST /api/recommendations disease={req.disease} replace={req.current_medicine}")
        result = ai_agent.recommend_alternative_medicine(
            patient_id=req.patient_id,
            disease=req.disease,
            current_medicine=req.current_medicine
        )
        return result
    except Exception as e:
        logger.error(f"Error executing recommendations endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/pattern-analysis/{doctor_id}", tags=["AI Clinical Guidance"])
async def pattern_analysis(doctor_id: str):
    """Analyze a physician's prescribing history to discover outlier/anomaly prescribing patterns."""
    if not ai_agent:
        raise HTTPException(status_code=503, detail="AI Agent logic is currently unavailable.")
    try:
        logger.info(f"API Request: GET /api/pattern-analysis/{doctor_id}")
        result = ai_agent.analyze_prescription_pattern(doctor_id=doctor_id)
        return result
    except Exception as e:
        logger.error(f"Error executing pattern-analysis endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# WEB PORTAL FRONTEND SERVING
# ============================================
@app.get("/doctor-login", tags=["Web Portal"])
async def serve_doctor_login():
    return FileResponse(os.path.join(os.path.dirname(__file__), "public", "doctor-login.html"))

@app.get("/doctor-prescription", tags=["Web Portal"])
async def serve_doctor_prescription():
    return FileResponse(os.path.join(os.path.dirname(__file__), "public", "doctor-prescription.html"))

@app.get("/pharmacy-portal", tags=["Web Portal"])
async def serve_pharmacy_portal():
    return FileResponse(os.path.join(os.path.dirname(__file__), "public", "pharmacy-portal.html"))

@app.get("/css/style.css", tags=["Web Portal Assets"])
async def serve_style_css():
    return FileResponse(
        os.path.join(os.path.dirname(__file__), "public", "css", "style.css"),
        media_type="text/css"
    )

if __name__ == "__main__":
    logger.info("Starting local development server on port 5000...")
    # Listen on all network interfaces
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
