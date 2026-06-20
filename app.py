from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
import uvicorn

# Import core modules
from ai_core import AIAgent
from config import logger

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
        yield
    finally:
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

if __name__ == "__main__":
    logger.info("Starting local development server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
