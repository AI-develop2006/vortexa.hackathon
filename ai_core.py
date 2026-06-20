import os
import json
from datetime import datetime, timedelta
from pymongo import MongoClient
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from requests import post

# Import settings and config-level logger from config
import config
from config import (
    MONGODB_URI, MONGODB_DATABASE, COLLECTION_PRESCRIPTIONS, COLLECTION_ALLERGIES,
    DUPLICATE_TOLERANCE_DAYS, PRESCRIPTION_HISTORY_DAYS, CEREBRAS_API_KEY, CEREBRAS_API_HEADERS,
    CEREBRAS_API_URL, CEREBRAS_MODEL, MOCK_CEREBRAS, OUTPUT_DIR, OUTPUT_FILE, logger
)

# ============================================
# MONGODB CONNECTION CLASS
# ============================================
class MongoDBConnector:
    """
    MongoDB Connector Class
    Handles all database operations
    """
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(MONGODB_URI, timeoutMS=10000)
            self.db = self.client[MONGODB_DATABASE]
            logger.info(f"Connected to MongoDB: {MONGODB_DATABASE}")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
            raise

    def get_prescriptions_collection(self):
        """Get prescriptions collection"""
        return self.db[COLLECTION_PRESCRIPTIONS]

    def get_allergies_collection(self):
        """Get allergies collection"""
        return self.db[COLLECTION_ALLERGIES]

    def get_patients_collection(self):
        """Get patients collection"""
        return self.db["patients"]

    def close(self):
        """Close MongoDB connection"""
        self.client.close()
        logger.info("MongoDB connection closed")


# ============================================
# ============================================
# CEREBRAS API CLIENT CLASS
# ============================================
class CerebrasAPIClient:
    """
    Cerebras API Client Class
    Handles all API calls to Cerebras.
    Supports local mock mode for instant evaluation without API keys.
    """
    def __init__(self):
        """Initialize Cerebras API client"""
        self.headers = CEREBRAS_API_HEADERS
        self.mock_mode = MOCK_CEREBRAS
        
        # Fallback to mock mode if API key is empty or default placeholder
        if CEREBRAS_API_KEY == "your_api_key_here" or not CEREBRAS_API_KEY:
            logger.warning("CEREBRAS_API_KEY is not configured or is placeholder. Falling back to Mock Mode.")
            self.mock_mode = True
            
        logger.info(f"Cerebras API client initialized. Mock mode: {self.mock_mode}")

    def check_drug_interactions(self, current_medicines: list, new_medicine: str, allergies: list) -> dict:
        """
        Check drug interactions + duplicate detection
        Args:
            current_medicines: List of current medicines (e.g., ["Metformin 500mg", "Aspirin 75mg"])
            new_medicine: New medicine to check (e.g., "Metformin 500mg")
            allergies: List of patient allergies (e.g., ["Penicillin"])
        Returns:
            dict: Cerebras API response with interaction details
        """
        logger.info(f"Checking drug interactions for patient (checking: {new_medicine})")
        
        if self.mock_mode:
            logger.info("Mock Mode active: Generating simulated response for drug interactions.")
            # Simulate drug interaction logic based on name
            new_med_base = new_medicine.split()[0].lower()
            current_med_bases = [m.split()[0].lower() for m in current_medicines]
            
            is_duplicate = new_med_base in current_med_bases
            is_allergy_conflict = any(allergy.lower() == new_med_base for allergy in allergies)
            
            if is_allergy_conflict:
                return {
                    "interaction_risk": "HIGH",
                    "interaction_details": f"Patient profile contains allergy conflict with {new_medicine}.",
                    "duplicate": is_duplicate,
                    "allergy_conflict": True,
                    "alternatives": ["Amoxicillin", "Ciprofloxacin"] if "penicillin" in new_med_base else ["Glipizide", "Sitagliptin"],
                    "recommendation": "DO NOT prescribe. High allergy conflict risk detected.",
                    "severity_score": 8,
                    "confidence": 0.98
                }
            elif is_duplicate:
                return {
                    "interaction_risk": "MEDIUM",
                    "interaction_details": f"Clinical duplicate detected: {new_medicine} matches existing active prescription.",
                    "duplicate": True,
                    "allergy_conflict": False,
                    "alternatives": ["Glipizide", "Sitagliptin"] if "metformin" in new_med_base else [],
                    "recommendation": f"Continue existing prescription of {new_medicine}. Avoid duplication.",
                    "severity_score": 5,
                    "confidence": 0.95
                }
            else:
                # Mock a low-risk interaction between Metformin and Aspirin (common diabetic regimen)
                return {
                    "interaction_risk": "LOW",
                    "interaction_details": "No significant drug interactions or therapeutic duplications detected.",
                    "duplicate": False,
                    "allergy_conflict": False,
                    "alternatives": [],
                    "recommendation": "Proceed with prescription. Keep standard monitor settings.",
                    "severity_score": 1,
                    "confidence": 0.92
                }

        system_prompt = (
            "You are an expert clinical pharmacy AI assistant. Your task is to analyze a patient's current medication list, "
            "a newly prescribed medicine, and their allergies to check for drug-drug interactions, therapeutic duplicates, "
            "and allergy conflicts.\n\n"
            "You must return a JSON object with the following schema:\n"
            "{\n"
            '  "interaction_risk": "HIGH" | "MEDIUM" | "LOW",\n'
            '  "interaction_details": "Detailed description of any drug-drug interactions, duplicate warnings, or allergy conflicts found.",\n'
            '  "duplicate": true | false,\n'
            '  "allergy_conflict": true | false,\n'
            '  "alternatives": ["Alternative medicine name 1", "Alternative medicine name 2"],\n'
            '  "recommendation": "Clinical advice on how to proceed, e.g., \'DO NOT prescribe\', \'Proceed with caution\', etc.",\n'
            '  "severity_score": <integer between 1 and 10 representing risk severity (1=lowest risk, 10=highest risk)>,\n'
            '  "confidence": <float between 0.0 and 1.0 representing confidence in this assessment>\n'
            "}"
        )
        user_prompt = (
            f"Current Medicines: {current_medicines}\n"
            f"New Medicine to Check: {new_medicine}\n"
            f"Patient Allergies: {allergies}"
        )
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        try:
            logger.info(f"Calling Cerebras API ({CEREBRAS_MODEL}) for drug interactions. Checking: {new_medicine}")
            response = post(
                CEREBRAS_API_URL,
                headers=self.headers,
                json=payload,
                timeout=15
            )
            if response.status_code != 200:
                logger.error(f"Cerebras API error: {response.status_code} - {response.text}")
                return {"error": f"Cerebras API error: {response.status_code}"}
            
            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            result = json.loads(content)
            logger.info(f"Cerebras API response received: duplicate={result.get('duplicate', False)}")
            return result
        except Exception as e:
            logger.error(f"Cerebras API call failed: {str(e)}")
            return {"error": str(e)}

    def recommend_medicine(self, disease: str, current_medicines: list, allergies: list) -> dict:
        """
        Recommend alternative medicines based on disease
        Args:
            disease: Disease name (e.g., "Type 2 Diabetes")
            current_medicines: List of current medicines
            allergies: List of patient allergies
        Returns:
            dict: Cerebras API response with recommended medicines
        """
        logger.info(f"Calling Cerebras API for medicine recommendation: {disease}")
        
        if self.mock_mode:
            logger.info("Mock Mode active: Generating simulated response for medicine recommendation.")
            # Standard recommendations based on disease
            if "diabetes" in disease.lower():
                recs = [
                    {
                        "medicine": "Glipizide",
                        "dosage": "5mg",
                        "effectiveness": 8,
                        "safety_score": 9,
                        "reasoning": "A sulfonylurea that stimulates beta cells to release insulin. Effective choice for Type 2 Diabetes."
                    },
                    {
                        "medicine": "Sitagliptin",
                        "dosage": "100mg",
                        "effectiveness": 7,
                        "safety_score": 10,
                        "reasoning": "DPP-4 inhibitor that increases insulin release and decreases glucagon. Very low hypoglycemia risk."
                    }
                ]
            else:
                recs = [
                    {
                        "medicine": "Amoxicillin",
                        "dosage": "500mg",
                        "effectiveness": 9,
                        "safety_score": 8,
                        "reasoning": "Broad-spectrum penicillin derivative suitable for bacterial infections."
                    }
                ]
            
            # Filter out recommendations that conflict with allergies
            filtered_recs = []
            for r in recs:
                med_name = r["medicine"].lower()
                if not any(allergy.lower() in med_name for allergy in allergies):
                    filtered_recs.append(r)
            
            return {
                "disease": disease,
                "recommended_medicines": filtered_recs,
                "contraindications": ["None major, monitor renal function."],
                "recommendation_ranking": [r["medicine"] for r in filtered_recs]
            }

        system_prompt = (
            "You are an expert clinical medicine recommender AI assistant. Your task is to recommend alternative or suitable "
            "medicines for a given disease, taking into account the patient's current medications (to avoid interactions) "
            "and allergies (to avoid conflict).\n\n"
            "You must return a JSON object with the following schema:\n"
            "{\n"
            '  "recommended_medicines": [\n'
            "    {\n"
            '      "medicine": "Medicine name",\n'
            '      "dosage": "Recommended dosage, e.g., 500mg",\n'
            '      "effectiveness": <integer between 1 and 10>,\n'
            '      "safety_score": <integer between 1 and 10>,\n'
            '      "reasoning": "Clinical reasoning for this recommendation"\n'
            "    }\n"
            "  ],\n"
            '  "contraindications": ["List of contraindications or precautions for these recommendations"],\n'
            '  "recommendation_ranking": ["Ordered list of recommended medicine names by rank"]\n'
            "}"
        )
        user_prompt = (
            f"Disease: {disease}\n"
            f"Current Medicines to Avoid/Check: {current_medicines}\n"
            f"Patient Allergies: {allergies}"
        )
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        try:
            logger.info(f"Calling Cerebras API ({CEREBRAS_MODEL}) for medicine recommendation: {disease}")
            response = post(
                CEREBRAS_API_URL,
                headers=self.headers,
                json=payload,
                timeout=15
            )
            if response.status_code != 200:
                logger.error(f"Cerebras API error: {response.status_code} - {response.text}")
                return {"error": f"Cerebras API error: {response.status_code}"}
            
            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            result = json.loads(content)
            logger.info(f"Cerebras API response received: recommendations={len(result.get('recommended_medicines', []))}")
            return result
        except Exception as e:
            logger.error(f"Cerebras API call failed: {str(e)}")
            return {"error": str(e)}

    def check_allergy(self, medicine: str, allergies: list) -> dict:
        """
        Check if medicine conflicts with allergies
        Args:
            medicine: Medicine name (e.g., "Penicillin")
            allergies: List of patient allergies
        Returns:
            dict: Cerebras API response with allergy conflict details
        """
        logger.info(f"Calling Cerebras API for allergy check: {medicine}")
        
        if self.mock_mode:
            logger.info("Mock Mode active: Generating simulated response for allergy check.")
            med_name = medicine.split()[0].lower()
            allergy_conflict = any(a.lower() == med_name for a in allergies)
            severity = "HIGH" if allergy_conflict else "LOW"
            
            return {
                "allergy_conflict": allergy_conflict,
                "conflicting_allergy": medicine.split()[0] if allergy_conflict else None,
                "severity": severity,
                "alert_level": "HIGH" if allergy_conflict else "LOW",
                "alert_message": f"⚠ ALLERGY CONFLICT: Patient is allergic to {medicine.split()[0]} (Severity: {severity})" if allergy_conflict else "No allergy conflict detected.",
                "recommendation": f"DO NOT prescribe {medicine} due to active allergy records." if allergy_conflict else "Proceed with prescription.",
                "suggested_alternatives": ["Amoxicillin", "Ciprofloxacin"] if med_name == "penicillin" else []
            }

        system_prompt = (
            "You are an expert clinical allergy screening AI assistant. Your task is to check if a prescribed medicine "
            "conflicts with a patient's known allergies.\n\n"
            "You must return a JSON object with the following schema:\n"
            "{\n"
            '  "allergy_conflict": true | false,\n'
            '  "conflicting_allergy": "Name of the conflicting allergy from the patient\'s list" or null,\n'
            '  "severity": "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN",\n'
            '  "alert_level": "HIGH" | "MEDIUM" | "LOW",\n'
            '  "alert_message": "Warning message or confirmation of safety",\n'
            '  "recommendation": "Clinical recommendation on whether to prescribe or avoid",\n'
            '  "suggested_alternatives": ["Alternative medicine 1", "Alternative medicine 2"]\n'
            "}"
        )
        user_prompt = (
            f"Medicine: {medicine}\n"
            f"Patient Allergies: {allergies}"
        )
        payload = {
            "model": CEREBRAS_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        try:
            logger.info(f"Calling Cerebras API ({CEREBRAS_MODEL}) for allergy check: {medicine}")
            response = post(
                CEREBRAS_API_URL,
                headers=self.headers,
                json=payload,
                timeout=15
            )
            if response.status_code != 200:
                logger.error(f"Cerebras API error: {response.status_code} - {response.text}")
                return {"error": f"Cerebras API error: {response.status_code}"}
            
            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"].strip()
            if content.startswith("```"):
                lines = content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines).strip()
            result = json.loads(content)
            logger.info(f"Cerebras API response received: allergy_conflict={result.get('allergy_conflict', False)}")
            return result
        except Exception as e:
            logger.error(f"Cerebras API call failed: {str(e)}")
            return {"error": str(e)}


# ============================================
# AI AGENT CORE CLASS
# ============================================
class AIAgent:
    """
    AI Agent Core Class
    Main logic for all AI features
    """
    def __init__(self):
        """Initialize AI Agent"""
        self.mongo = MongoDBConnector()
        self.cerebras = CerebrasAPIClient()
        logger.info("AI Agent initialized")

    # ============================================
    # FEATURE 1: DUPLICATE MEDICINE DETECTION
    # ============================================
    def detect_duplicate_medicine(self, patient_id: str, new_medicine: str, new_dosage: str) -> dict:
        """
        Detect duplicate medicine using MongoDB query + String Matching
        Technology: MongoDB query + Simple String Matching
        Why: Fast, no ML needed for exact matches
        Args:
            patient_id: Patient ID (e.g., "Priya_123")
            new_medicine: New medicine name (e.g., "Metformin")
            new_dosage: New dosage (e.g., "500mg")
        Returns:
            dict: Duplicate detection result
        """
        logger.info(f"Detecting duplicate medicine for patient {patient_id}: {new_medicine} {new_dosage}")
        
        # Step 1: Calculate date threshold (DUPLICATE_TOLERANCE_DAYS days ago)
        date_threshold = datetime.now() - timedelta(days=DUPLICATE_TOLERANCE_DAYS)
        
        # Step 2: Query MongoDB for existing prescriptions
        prescriptions_collection = self.mongo.get_prescriptions_collection()
        duplicate = prescriptions_collection.find_one({
            "patient_id": patient_id,
            "medicine": new_medicine,
            "dosage": new_dosage,
            "date": {"$gte": date_threshold}
        })
        
        # Step 3: Check if duplicate found
        if duplicate:
            logger.info(f"DUPLICATE FOUND: {new_medicine} {new_dosage} prescribed by {duplicate['doctor_id']} on {duplicate['date']}")
            # Format datetime for JSON responsiveness
            dup_date = duplicate["date"].isoformat() if isinstance(duplicate["date"], datetime) else str(duplicate["date"])
            result = {
                "is_duplicate": True,
                "duplicate_medicine": new_medicine,
                "duplicate_dosage": new_dosage,
                "existing_prescription": {
                    "doctor_id": duplicate["doctor_id"],
                    "date": dup_date,
                    "dosage": duplicate["dosage"],
                    "frequency": duplicate.get("frequency", "Unknown")
                },
                "alert_level": "MEDIUM",
                "alert_message": f"⚠ DUPLICATE: {new_medicine} {new_dosage} already prescribed within the last {DUPLICATE_TOLERANCE_DAYS} days.",
                "recommendation": f"Continue existing {new_medicine} prescription. Do not prescribe duplicate."
            }
        else:
            logger.info(f"No duplicate found for {new_medicine} {new_dosage}")
            result = {
                "is_duplicate": False,
                "alert_level": "LOW",
                "alert_message": f"No duplicate detected for {new_medicine} {new_dosage}.",
                "recommendation": "Proceed with prescription"
            }
        return result

    # ============================================
    # FEATURE 2: DRUG INTERACTION CHECKER
    # ============================================
    def check_drug_interactions(self, patient_id: str, new_medicine: str, new_dosage: str) -> dict:
        """
        Check drug interactions using Cerebras API
        Technology: Cerebras API (LLM analysis)
        Why: Leverage Cerebras fast LLM inference for checking interactions and allergy matches
        Args:
            patient_id: Patient ID
            new_medicine: New medicine name
            new_dosage: New dosage
        Returns:
            dict: Drug interaction check result
        """
        logger.info(f"Checking drug interactions for patient {patient_id}: {new_medicine} {new_dosage}")
        
        # Step 1: Get patient's current medicines from MongoDB
        current_medicines = self.get_current_medicines(patient_id)
        
        # Step 2: Get patient's allergies from MongoDB
        allergies = self.get_patient_allergies(patient_id)
        
        # Step 3: Prepare new medicine string
        new_medicine_string = f"{new_medicine} {new_dosage}"
        
        # Step 4: Call Cerebras API
        cerebras_response = self.cerebras.check_drug_interactions(
            current_medicines=current_medicines,
            new_medicine=new_medicine_string,
            allergies=allergies
        )
        
        # Step 5: Parse Cerebras response
        if "error" in cerebras_response:
            logger.error(f"Cerebras API error: {cerebras_response['error']}")
            return {
                "error": cerebras_response["error"],
                "interaction_risk": "UNKNOWN",
                "recommendation": "Unable to check drug interactions due to API/network error."
            }
            
        # Step 6: Build result
        result = {
            "interaction_risk": cerebras_response.get("interaction_risk", "LOW"),
            "interaction_details": cerebras_response.get("interaction_details", ""),
            "duplicate_detected": cerebras_response.get("duplicate", False),
            "allergy_conflict": cerebras_response.get("allergy_conflict", False),
            "alternatives": cerebras_response.get("alternatives", []),
            "recommendation": cerebras_response.get("recommendation", ""),
            "severity_score": cerebras_response.get("severity_score", 0),
            "confidence": cerebras_response.get("confidence", 0)
        }
        logger.info(f"Drug interaction check complete: risk={result['interaction_risk']}")
        return result

    # ============================================
    # FEATURE 3: ALLERGY CONFLICT DETECTOR
    # ============================================
    def check_allergy_conflict(self, patient_id: str, new_medicine: str) -> dict:
        """
        Check allergy conflict using MongoDB + Rule-Based Engine
        Technology: MongoDB query + Rule-Based Engine
        Why: Simple IF medicine IN allergies THEN alert, no ML needed
        Args:
            patient_id: Patient ID
            new_medicine: New medicine name
        Returns:
            dict: Allergy conflict check result
        """
        logger.info(f"Checking allergy conflict for patient {patient_id}: {new_medicine}")
        
        # Step 1: Get patient's allergies from MongoDB
        allergies_collection = self.mongo.get_allergies_collection()
        allergies = list(allergies_collection.find({"patient_id": patient_id}))
        allergy_list = [allergy["allergy_name"].lower() for allergy in allergies]
        
        # Step 2: Extract medicine name (remove dosage)
        medicine_name = new_medicine.split()[0]  # "Metformin 500mg" -> "Metformin"
        
        # Step 3: Rule-Based Engine: Simple string matching
        allergy_conflict = medicine_name.lower() in allergy_list
        
        # Step 4: Build result
        if allergy_conflict:
            # Find allergy severity
            allergy_severity = "UNKNOWN"
            for allergy in allergies:
                if allergy["allergy_name"].lower() == medicine_name.lower():
                    allergy_severity = allergy.get("severity", "HIGH")
                    break
                    
            logger.info(f"ALLERGY CONFLICT: Patient has {allergy_severity} severity allergy to {medicine_name}")
            result = {
                "allergy_conflict": True,
                "conflicting_allergy": medicine_name,
                "severity": allergy_severity,
                "alert_level": allergy_severity,
                "alert_message": f"⚠ ALLERGY CONFLICT: Patient has {allergy_severity} severity allergy to {medicine_name}.",
                "recommendation": f"DO NOT prescribe {new_medicine}. Patient has allergy.",
                "suggested_alternatives": self.get_alternative_medicines(medicine_name)
            }
        else:
            logger.info(f"No allergy conflict for {medicine_name}")
            result = {
                "allergy_conflict": False,
                "alert_level": "LOW",
                "alert_message": f"No allergy conflict for {new_medicine}.",
                "recommendation": "Proceed with prescription"
            }
        return result

    # ============================================
    # FEATURE 4: MEDICINE RECOMMENDATION
    # ============================================
    def recommend_alternative_medicine(self, patient_id: str, disease: str, current_medicine: str) -> dict:
        """
        Recommend alternative medicines using Cerebras API (AI)
        Technology: Cerebras API (LLM analysis)
        Why: Suggests alternatives based on disease and filters out conflicts
        Args:
            patient_id: Patient ID
            disease: Disease name (e.g., "Type 2 Diabetes")
            current_medicine: Current medicine to replace (e.g., "Metformin")
        Returns:
            dict: Medicine recommendation result
        """
        logger.info(f"Recommending alternative medicine for patient {patient_id}: replace {current_medicine} for {disease}")
        
        # Step 1: Get patient's current medicines from MongoDB
        current_medicines = self.get_current_medicines(patient_id)
        
        # Step 2: Get patient's allergies from MongoDB
        allergies = self.get_patient_allergies(patient_id)
        
        # Step 3: Remove current medicine from list (we want alternatives)
        # Strip dosages for matching or do split comparison
        current_medicines_filtered = [m for m in current_medicines if current_medicine.lower() not in m.lower()]
        
        # Step 4: Call Cerebras API for recommendation
        cerebras_response = self.cerebras.recommend_medicine(
            disease=disease,
            current_medicines=current_medicines_filtered,
            allergies=allergies
        )
        
        # Step 5: Parse Cerebras response
        if "error" in cerebras_response:
            logger.error(f"Cerebras API error: {cerebras_response['error']}")
            return {
                "error": cerebras_response["error"],
                "recommended_medicines": [],
                "recommendation": "Unable to recommend alternative medicines due to API/network error."
            }
            
        # Step 6: Build result
        recommended_medicines = cerebras_response.get("recommended_medicines", [])
        
        # Format recommended medicines
        formatted_recommendations = [
            {
                "medicine": m.get("medicine", ""),
                "dosage": m.get("dosage", ""),
                "effectiveness": m.get("effectiveness", 0),
                "safety_score": m.get("safety_score", 0),
                "reasoning": m.get("reasoning", "")
            } for m in recommended_medicines
        ]
        
        result = {
            "disease": disease,
            "recommended_medicines": formatted_recommendations,
            "contraindications": cerebras_response.get("contraindications", []),
            "recommendation_ranking": cerebras_response.get("recommendation_ranking", []),
            "total_recommendations": len(formatted_recommendations)
        }
        logger.info(f"Medicine recommendation complete: {len(formatted_recommendations)} alternatives found.")
        return result

    # ============================================
    # FEATURE 5: PRESCRIPTION PATTERN ANALYSIS
    # ============================================
    def analyze_prescription_pattern(self, doctor_id: str) -> dict:
        """
        Analyze prescription patterns using Python + Pandas + Scikit-learn
        Technology: Python + Pandas + Scikit-learn (Isolation Forest)
        Why: Detects outliers (unusual prescribing patterns), no need for complex ML
        Args:
            doctor_id: Doctor ID (e.g., "Dr_Arun_456")
        Returns:
            dict: Prescription pattern analysis result
        """
        logger.info(f"Analyzing prescription pattern for doctor {doctor_id}")
        
        # Step 1: Query MongoDB for doctor's prescription history
        prescriptions_collection = self.mongo.get_prescriptions_collection()
        prescriptions = list(prescriptions_collection.find({"doctor_id": doctor_id}))
        
        if len(prescriptions) == 0:
            logger.warning(f"No prescriptions found for doctor {doctor_id}")
            return {
                "doctor_id": doctor_id,
                "anomaly_score": 0.0,
                "pattern_type": "NO_DATA",
                "recommendation": "No prescription data available"
            }
            
        # Step 2: Convert to Pandas DataFrame
        df = pd.DataFrame(prescriptions)
        
        # Step 3: Aggregate medicine counts
        medicine_counts = df.groupby("medicine").size().reset_index(name="count")
        
        # Step 4: Calculate statistics
        mean_count = medicine_counts["count"].mean()
        std_count = medicine_counts["count"].std()
        
        # If too few unique medications, IsolationForest is not suitable
        if len(medicine_counts) < 2:
            logger.info("Not enough unique medicines to run Isolation Forest. Standard normal template returned.")
            return {
                "doctor_id": doctor_id,
                "total_prescriptions": len(prescriptions),
                "total_medicines": len(medicine_counts),
                "mean_prescription_count": float(mean_count),
                "std_prescription_count": 0.0,
                "anomaly_score": 0.0,
                "pattern_type": "NORMAL",
                "flagged_medicines": [],
                "recommendation": "Prescription patterns are within normal ranges."
            }
            
        # Step 5: Detect outliers using Isolation Forest (Scikit-learn)
        X = medicine_counts["count"].values.reshape(-1, 1)
        contamination_val = min(0.1, 1.0 / len(medicine_counts))
        model = IsolationForest(contamination=contamination_val, random_state=42)
        outliers = model.fit_predict(X)
        
        # Step 6: Extract flagged medicines (outliers, fitted value is -1)
        flagged_indices = np.where(outliers == -1)[0]
        flagged_medicines = medicine_counts.iloc[flagged_indices]["medicine"].tolist()
        
        # Step 7: Calculate anomaly score (0-10)
        # Ratio of outlier points to total points scaled to 10
        anomaly_score = min(10.0, (len(flagged_medicines) / len(medicine_counts)) * 10.0)
        
        # Step 8: Determine pattern type
        if anomaly_score > 7.0:
            pattern_type = "OVER_PRESCRIBING"
        elif anomaly_score > 5.0:
            pattern_type = "UNUSUAL_PATTERN"
        elif anomaly_score > 3.0:
            pattern_type = "MILD_ANOMALY"
        else:
            pattern_type = "NORMAL"
            
        # Step 9: Build result
        recommendation = "Prescription patterns are within normal ranges."
        if flagged_medicines:
            recommendation = f"Doctor prescribes {flagged_medicines} at an unusually high or low frequency."
            
        result = {
            "doctor_id": doctor_id,
            "total_prescriptions": len(prescriptions),
            "total_medicines": len(medicine_counts),
            "mean_prescription_count": float(mean_count),
            "std_prescription_count": float(std_count) if not pd.isna(std_count) else 0.0,
            "anomaly_score": float(anomaly_score),
            "pattern_type": pattern_type,
            "flagged_medicines": flagged_medicines,
            "recommendation": recommendation
        }
        logger.info(f"Prescription pattern analysis complete: anomaly_score={anomaly_score}")
        return result

    # ============================================
    # HELPER FUNCTIONS
    # ============================================
    def get_current_medicines(self, patient_id: str, days: int = PRESCRIPTION_HISTORY_DAYS) -> list:
        """
        Get patient's current medicines from MongoDB
        Args:
            patient_id: Patient ID
            days: Days to look back (default: 90)
        Returns:
            list: List of current medicines (e.g. ["Metformin 500mg", "Aspirin 75mg"])
        """
        date_threshold = datetime.now() - timedelta(days=days)
        
        prescriptions_collection = self.mongo.get_prescriptions_collection()
        prescriptions = prescriptions_collection.find({
            "patient_id": patient_id,
            "date": {"$gte": date_threshold}
        })
        
        medicines = []
        for presc in prescriptions:
            medicine_string = f"{presc['medicine']} {presc['dosage']}"
            if medicine_string not in medicines:
                medicines.append(medicine_string)
        return medicines

    def get_patient_allergies(self, patient_id: str) -> list:
        """
        Get patient's allergies from MongoDB
        Args:
            patient_id: Patient ID
        Returns:
            list: List of allergies (e.g. ["Penicillin", "Aspirin"])
        """
        allergies_collection = self.mongo.get_allergies_collection()
        allergies = allergies_collection.find({"patient_id": patient_id})
        allergy_list = [allergy["allergy_name"] for allergy in allergies]
        return allergy_list

    def get_alternative_medicines(self, medicine: str) -> list:
        """
        Get alternative medicines (simple hardcoded list for demo fallback)
        Args:
            medicine: Medicine name
        Returns:
            list: List of alternative medicines
        """
        alternatives = {
            "Penicillin": ["Amoxicillin", "Ciprofloxacin", "Azithromycin"],
            "Metformin": ["Glipizide", "Insulin", "Sitagliptin"],
            "Aspirin": ["Ibuprofen", "Aceclofenac", "Naproxen"],
            "Paracetamol": ["Ibuprofen", "Naproxen", "Iodamide"]
        }
        return alternatives.get(medicine, ["Consult doctor for alternatives"])

    # ============================================
    # MAIN AI ANALYSIS FUNCTION (COMBINES ALL FEATURES)
    # ============================================
    def analyze_prescription(self, patient_id: str, doctor_id: str, new_medicine: str, new_dosage: str, disease: str = None) -> dict:
        """
        Complete AI analysis for a new prescription
        Combines all 5 features: Duplicate, Interaction, Allergy, Recommendation, Pattern
        Args:
            patient_id: Patient ID
            doctor_id: Doctor ID
            new_medicine: New medicine name
            new_dosage: New dosage
            disease: Disease name (optional, for recommendation)
        Returns:
            dict: Complete AI analysis result
        """
        logger.info(f"Starting complete AI analysis for patient {patient_id}: {new_medicine} {new_dosage}")
        
        # Initialize result
        result = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "new_medicine": new_medicine,
            "new_dosage": new_dosage,
            "analysis_timestamp": datetime.now().isoformat(),
            "features": {}
        }
        
        # Feature 1: Duplicate Medicine Detection
        logger.info("=== Feature 1: Duplicate Medicine Detection ===")
        duplicate_result = self.detect_duplicate_medicine(patient_id, new_medicine, new_dosage)
        result["features"]["duplicate_detection"] = duplicate_result
        
        # Feature 2: Drug Interaction Checker
        logger.info("=== Feature 2: Drug Interaction Checker ===")
        interaction_result = self.check_drug_interactions(patient_id, new_medicine, new_dosage)
        result["features"]["drug_interactions"] = interaction_result
        
        # Feature 3: Allergy Conflict Detector
        logger.info("=== Feature 3: Allergy Conflict Detector ===")
        allergy_result = self.check_allergy_conflict(patient_id, new_medicine)
        result["features"]["allergy_conflict"] = allergy_result
        
        # Feature 4: Medicine Recommendation (if disease provided)
        if disease:
            logger.info("=== Feature 4: Medicine Recommendation ===")
            recommendation_result = self.recommend_alternative_medicine(patient_id, disease, new_medicine)
            result["features"]["medicine_recommendation"] = recommendation_result
            
        # Feature 5: Prescription Pattern Analysis
        logger.info("=== Feature 5: Prescription Pattern Analysis ===")
        pattern_result = self.analyze_prescription_pattern(doctor_id)
        result["features"]["prescription_pattern"] = pattern_result
        
        # Calculate overall alert level
        alert_levels = [
            duplicate_result.get("alert_level", "LOW"),
            interaction_result.get("interaction_risk", "LOW"),
            allergy_result.get("alert_level", "LOW")
        ]
        
        # Determine highest alert level
        if "CRITICAL" in alert_levels:
            overall_alert = "CRITICAL"
        elif "HIGH" in alert_levels:
            overall_alert = "HIGH"
        elif "MEDIUM" in alert_levels:
            overall_alert = "MEDIUM"
        else:
            overall_alert = "LOW"
            
        result["overall_alert_level"] = overall_alert
        
        # Generate final recommendation
        if overall_alert in ("CRITICAL", "HIGH"):
            result["final_recommendation"] = "DO NOT prescribe. Critical/High alert detected. Review patient allergies or interactions immediately."
        elif overall_alert == "MEDIUM":
            result["final_recommendation"] = "Consider alternative. Medium alert detected. Review duplicate prescribing status."
        else:
            result["final_recommendation"] = "Proceed with prescription. No alerts detected."
            
        logger.info(f"AI analysis complete: overall_alert={overall_alert}")
        return result

    # ============================================
    # CLOSE CONNECTIONS
    # ============================================
    def close(self):
        """Close all connections"""
        self.mongo.close()
        logger.info("AI Agent connections closed")


# ============================================
# MAIN EXECUTION (TEST SCRIPT)
# ============================================
def main():
    """
    Main function to test AI Agent
    Run: python ai_core.py
    """
    print("=" * 80)
    print("AI AGENT CORE MODULE - STANDALONE TEST")
    print("=" * 80)
    
    # Initialize AI Agent
    ai = AIAgent()
    
    # Common test variables
    test_patient_id = "Priya_123"
    test_medicine = "Metformin"
    test_dosage = "500mg"
    test_disease = "Type 2 Diabetes"
    
    # ============================================
    # TEST CASE 1: Duplicate Medicine Detection
    # ============================================
    print("\n" + "=" * 80)
    print("TEST CASE 1: Duplicate Medicine Detection")
    print("=" * 80)
    duplicate_result = ai.detect_duplicate_medicine(test_patient_id, test_medicine, test_dosage)
    print(f"\nPatient ID: {test_patient_id}")
    print(f"Medicine: {test_medicine} {test_dosage}")
    print(f"\nResult:")
    print(json.dumps(duplicate_result, indent=2))
    
    # ============================================
    # TEST CASE 2: Drug Interaction Checker
    # ============================================
    print("\n" + "=" * 80)
    print("TEST CASE 2: Drug Interaction Checker")
    print("=" * 80)
    interaction_result = ai.check_drug_interactions(test_patient_id, test_medicine, test_dosage)
    print(f"\nPatient ID: {test_patient_id}")
    print(f"Medicine: {test_medicine} {test_dosage}")
    print(f"\nResult:")
    print(json.dumps(interaction_result, indent=2))
    
    # ============================================
    # TEST CASE 3: Allergy Conflict Detector
    # ============================================
    print("\n" + "=" * 80)
    print("TEST CASE 3: Allergy Conflict Detector")
    print("=" * 80)
    test_allergy_medicine = "Penicillin"
    allergy_result = ai.check_allergy_conflict(test_patient_id, test_allergy_medicine)
    print(f"\nPatient ID: {test_patient_id}")
    print(f"Medicine: {test_allergy_medicine}")
    print(f"\nResult:")
    print(json.dumps(allergy_result, indent=2))
    
    # ============================================
    # TEST CASE 4: Medicine Recommendation
    # ============================================
    print("\n" + "=" * 80)
    print("TEST CASE 4: Medicine Recommendation")
    print("=" * 80)
    recommendation_result = ai.recommend_alternative_medicine(test_patient_id, test_disease, test_medicine)
    print(f"\nPatient ID: {test_patient_id}")
    print(f"Disease: {test_disease}")
    print(f"Current Medicine: {test_medicine}")
    print(f"\nResult:")
    print(json.dumps(recommendation_result, indent=2))
    
    # ============================================
    # TEST CASE 5: Prescription Pattern Analysis
    # ============================================
    print("\n" + "=" * 80)
    print("TEST CASE 5: Prescription Pattern Analysis")
    print("=" * 80)
    test_doctor_id = "Dr_Arun_456"
    pattern_result = ai.analyze_prescription_pattern(test_doctor_id)
    print(f"\nDoctor ID: {test_doctor_id}")
    print(f"\nResult:")
    print(json.dumps(pattern_result, indent=2))
    
    # ============================================
    # TEST CASE 6: Complete AI Analysis (All Features Combined)
    # ============================================
    print("\n" + "=" * 80)
    print("TEST CASE 6: Complete AI Analysis (All Features Combined)")
    print("=" * 80)
    complete_result = ai.analyze_prescription(
        patient_id=test_patient_id,
        doctor_id="Dr_Kumar_456",
        new_medicine=test_medicine,
        new_dosage=test_dosage,
        disease=test_disease
    )
    print(f"\nPatient ID: {test_patient_id}")
    print(f"Doctor ID: Dr_Kumar_456")
    print(f"Medicine: {test_medicine} {test_dosage}")
    print(f"Disease: {test_disease}")
    print(f"\nOverall Alert Level: {complete_result['overall_alert_level']}")
    print(f"\nFinal Recommendation: {complete_result['final_recommendation']}")
    print(f"\nComplete Result:")
    print(json.dumps(complete_result, indent=2))
    
    # Save result to output file
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)
    with open(output_path, 'w') as f:
        json.dump(complete_result, f, indent=2)
    print(f"\nResult saved to: {output_path}")
    
    # Close connections
    ai.close()
    print("\n" + "=" * 80)
    print("AI AGENT TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
