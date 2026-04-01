import sqlite3
import pandas as pd
from datetime import datetime

def view_database():
    """View all predictions stored in database"""
    
    try:
        conn = sqlite3.connect('stroke_predictions.db')
        
        # Get all predictions
        df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
        
        print("\n" + "="*100)
        print(" STROKE PREDICTION DATABASE - ALL RECORDS")
        print("="*100)
        
        if len(df) == 0:
            print("No predictions found in database yet.")
            print("Make some predictions using the web app first!")
        else:
            print(f"\n Total Predictions: {len(df)}\n")
            
            # Display summary
            print(df.to_string(index=False))
            
            print("\n" + "="*100)
            print(" SUMMARY STATISTICS")
            print("="*100)
            
            # Risk level distribution
            print("\n Risk Level Distribution:")
            risk_counts = df['risk_level'].value_counts()
            for risk, count in risk_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {risk}: {count} ({percentage:.1f}%)")
            
            # Gender distribution
            print("\n Gender Distribution:")
            gender_counts = df['gender'].value_counts()
            for gender, count in gender_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {gender}: {count} ({percentage:.1f}%)")
            
            # Average metrics
            print("\n Average Metrics:")
            print(f"  Average Age: {df['age'].mean():.1f} years")
            print(f"  Average BMI: {df['bmi'].mean():.1f}")
            print(f"  Average Glucose: {df['avg_glucose_level'].mean():.1f} mg/dL")
            print(f"  Average Stroke Probability: {df['stroke_probability'].mean()*100:.1f}%")
            
            # High-risk patients
            high_risk = df[df['risk_level'].isin(['High', 'Very High'])]
            print(f"\n  High-Risk Patients: {len(high_risk)} ({len(high_risk)/len(df)*100:.1f}%)")
            
            # Patients with hypertension
            hypertension_count = df[df['hypertension'] == 1].shape[0]
            print(f" Patients with Hypertension: {hypertension_count} ({hypertension_count/len(df)*100:.1f}%)")
            
            # Patients with heart disease
            heart_disease_count = df[df['heart_disease'] == 1].shape[0]
            print(f"  Patients with Heart Disease: {heart_disease_count} ({heart_disease_count/len(df)*100:.1f}%)")
            
            # Smoking status
            print("\n🚬 Smoking Status:")
            smoking_counts = df['smoking_status'].value_counts()
            for status, count in smoking_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {status}: {count} ({percentage:.1f}%)")
        
        conn.close()
        
        print("\n" + "="*100 + "\n")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def view_patient(patient_id):
    """View history for specific patient"""
    
    try:
        conn = sqlite3.connect('stroke_predictions.db')
        
        df = pd.read_sql_query(
            "SELECT * FROM predictions WHERE patient_id = ? ORDER BY timestamp DESC",
            conn,
            params=(patient_id,)
        )
        
        if len(df) == 0:
            print(f"No records found for patient: {patient_id}")
        else:
            print(f"\n Patient History: {patient_id}")
            print("="*100)
            print(df.to_string(index=False))
            print("="*100)
            print(f"\nTotal Assessments: {len(df)}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # View specific patient
        patient_id = sys.argv[1]
        view_patient(patient_id)
    else:
        # View all predictions
        view_database()