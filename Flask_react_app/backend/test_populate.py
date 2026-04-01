# save as test_populate.py
import sqlite3
from datetime import datetime, timedelta
import random

def add_test_data():
    conn = sqlite3.connect('stroke_predictions.db')
    cursor = conn.cursor()
    
    # Add 50 sample patients with varied ages
    for i in range(50):
        # Create age distribution
        if i < 10:  # 20% young
            age = random.randint(20, 29)
        elif i < 30:  # 40% middle-aged
            age = random.randint(30, 49)
        elif i < 45:  # 30% older
            age = random.randint(50, 64)
        else:  # 10% elderly
            age = random.randint(65, 85)
        
        cursor.execute('''
            INSERT INTO predictions (
                patient_id, timestamp, gender, age, hypertension, heart_disease,
                ever_married, work_type, residence_type, avg_glucose_level, bmi,
                smoking_status, prediction, stroke_probability, no_stroke_probability,
                risk_level, confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"TEST{i:03d}",
            (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat(),
            random.choice(['Male', 'Female']),
            age,
            random.choice([0, 1]),
            random.choice([0, 1]),
            random.choice(['Yes', 'No']),
            random.choice(['Private', 'Self-employed', 'Govt_job']),
            random.choice(['Urban', 'Rural']),
            round(random.uniform(70, 250), 1),
            round(random.uniform(18, 40), 1),
            random.choice(['never smoked', 'formerly smoked', 'smokes']),
            random.choice([0, 1]),
            round(random.uniform(0.1, 0.9), 3),
            round(random.uniform(0.1, 0.9), 3),
            random.choice(['Low', 'Medium', 'High', 'Very High']),
            round(random.uniform(70, 98), 1),
            f"Test patient {i}"
        ))
    
    conn.commit()
    
    # Verify
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT 
            CASE 
                WHEN age < 30 THEN 'Under 30'
                WHEN age < 50 THEN '30-49'
                WHEN age < 65 THEN '50-64'
                ELSE '65+'
            END as age_group,
            COUNT(*) as count
        FROM predictions 
        GROUP BY age_group
    """)
    
    print("Test data added successfully!")
    print(f"Total patients: {total}")
    print("\nAge distribution:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]} patients")
    
    conn.close()

if __name__ == "__main__":
    add_test_data()