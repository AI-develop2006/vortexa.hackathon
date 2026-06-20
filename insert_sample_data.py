import json
import os
from datetime import datetime
from pymongo import MongoClient
import config
from config import logger

def populate_database():
    logger.info("Starting database population with sample test data...")
    
    # Path to test data file
    json_path = os.path.join("test_data", "sample_patient.json")
    if not os.path.exists(json_path):
        logger.error(f"Sample data file not found at: {json_path}")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Connect to MongoDB
    try:
        client = MongoClient(config.MONGODB_URI, timeoutMS=5000)
        db = client[config.MONGODB_DATABASE]
        
        # Reset collections
        db[config.COLLECTION_PRESCRIPTIONS].drop()
        db[config.COLLECTION_ALLERGIES].drop()
        db["patients"].drop()
        db["doctor"].drop()
        
        logger.info("Cleared existing collections: patients, prescriptions, allergies, doctor")
        
        # Insert patients
        patients = data.get("patients", [])
        if patients:
            db["patients"].insert_many(patients)
            logger.info(f"Inserted {len(patients)} patients")
            
        # Insert doctors
        doctors = data.get("doctors", [])
        if doctors:
            db["doctor"].insert_many(doctors)
            logger.info(f"Inserted {len(doctors)} doctor records")
            
        # Insert allergies
        allergies = data.get("allergies", [])
        if allergies:
            db[config.COLLECTION_ALLERGIES].insert_many(allergies)
            logger.info(f"Inserted {len(allergies)} allergy records")
            
        # Insert prescriptions (convert date strings to datetime objects)
        prescriptions = data.get("prescriptions", [])
        formatted_prescriptions = []
        for presc in prescriptions:
            presc_copy = presc.copy()
            if "date" in presc_copy:
                # Convert ISO format date string to datetime
                presc_copy["date"] = datetime.fromisoformat(presc_copy["date"])
            formatted_prescriptions.append(presc_copy)
            
        if formatted_prescriptions:
            db[config.COLLECTION_PRESCRIPTIONS].insert_many(formatted_prescriptions)
            logger.info(f"Inserted {len(formatted_prescriptions)} prescription records")
            
        logger.info("Database population complete!")
        
    except Exception as e:
        logger.error(f"Error populating database: {str(e)}")
    finally:
        client.close()

if __name__ == "__main__":
    populate_database()
