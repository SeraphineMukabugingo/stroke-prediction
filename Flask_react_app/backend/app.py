from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib
import os
import sqlite3
from datetime import datetime, timedelta
import calendar

app = Flask(__name__)
CORS(app)

# ============================================================================
# DATABASE SETUP
# ============================================================================

def init_database():
    """Initialize SQLite database to store predictions"""
    conn = sqlite3.connect('stroke_predictions.db')
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            timestamp TEXT,
            gender TEXT,
            age REAL,
            hypertension INTEGER,
            heart_disease INTEGER,
            ever_married TEXT,
            work_type TEXT,
            residence_type TEXT,
            avg_glucose_level REAL,
            bmi REAL,
            smoking_status TEXT,
            prediction INTEGER,
            stroke_probability REAL,
            no_stroke_probability REAL,
            risk_level TEXT,
            confidence REAL,
            notes TEXT
        )
    ''')
    
    # Create risk factors table for detailed analysis
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS risk_factors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            factor_name TEXT,
            factor_value REAL,
            contribution REAL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(" Database initialized: stroke_predictions.db")

# Initialize database when app starts
init_database()

# ============================================================================
# MODEL SETUP
# ============================================================================

def create_model():
    df = pd.read_csv('healthcare-dataset-stroke-data.csv')
    y = df['stroke']
    X = df.drop(['stroke', 'id'], axis=1, errors='ignore')
    
    numerical = ['avg_glucose_level', 'bmi', 'age']
    categorical = ['hypertension', 'heart_disease', 'ever_married', 
                   'work_type', 'Residence_type', 'smoking_status']
    
    num_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', PowerTransformer(method='yeo-johnson', standardize=True))
    ])
    
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numerical),
        ('cat', cat_transformer, categorical)
    ])
    
    pipeline = Pipeline([
        ('transformer', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', LinearDiscriminantAnalysis())
    ])
    
    print("Training model...")
    pipeline.fit(X, y)
    joblib.dump(pipeline, 'model.pkl')
    print("Model trained and saved!")
    return pipeline

try:
    model = joblib.load('model.pkl')
    print("Model loaded from file!")
except:
    print("Model not found. Training new model...")
    model = create_model()

# ============================================================================
# DATABASE HELPER FUNCTIONS
# ============================================================================

def save_prediction_to_db(data, result):
    """Save prediction to database"""
    try:
        conn = sqlite3.connect('stroke_predictions.db')
        cursor = conn.cursor()
        
        # Generate patient ID if not provided
        patient_id = data.get('patient_id', f"PATIENT_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        # Insert prediction into database
        cursor.execute('''
            INSERT INTO predictions (
                patient_id, timestamp, gender, age, hypertension, heart_disease,
                ever_married, work_type, residence_type, avg_glucose_level, bmi,
                smoking_status, prediction, stroke_probability, no_stroke_probability,
                risk_level, confidence, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            datetime.now().isoformat(),
            data.get('gender'),
            data.get('age'),
            data.get('hypertension'),
            data.get('heart_disease'),
            data.get('ever_married'),
            data.get('work_type'),
            data.get('Residence_type'),
            data.get('avg_glucose_level'),
            data.get('bmi'),
            data.get('smoking_status'),
            result['prediction'],
            result['probability']['stroke'],
            result['probability']['no_stroke'],
            result['risk_level'],
            result['confidence'],
            data.get('notes', '')
        ))
        
        prediction_id = cursor.lastrowid
        
        # Save risk factor contributions
        if 'risk_factors' in result:
            for factor in result['risk_factors']:
                cursor.execute('''
                    INSERT INTO risk_factors (prediction_id, factor_name, factor_value, contribution)
                    VALUES (?, ?, ?, ?)
                ''', (prediction_id, factor['name'], factor['value'], factor['contribution']))
        
        conn.commit()
        conn.close()
        
        print(f"Prediction saved to database! ID: {prediction_id}, Patient: {patient_id}")
        return prediction_id, patient_id
        
    except Exception as e:
        print(f" Error saving to database: {str(e)}")
        return None, None

def calculate_risk_factors(data):
    """Calculate individual risk factor contributions"""
    factors = []
    
    # Age risk (increases after 50)
    age = float(data.get('age', 0))
    age_risk = max(0, (age - 50) * 0.02) if age > 50 else 0
    factors.append({
        'name': 'Age',
        'value': age,
        'contribution': age_risk,
        'unit': 'years',
        'risk_category': 'High' if age > 65 else 'Medium' if age > 50 else 'Low'
    })
    
    # BMI risk
    bmi = float(data.get('bmi', 0))
    bmi_risk = 0
    if bmi > 30:
        bmi_risk = 0.15
    elif bmi > 25:
        bmi_risk = 0.08
    factors.append({
        'name': 'BMI',
        'value': bmi,
        'contribution': bmi_risk,
        'unit': 'kg/m²',
        'risk_category': 'High' if bmi > 30 else 'Medium' if bmi > 25 else 'Normal'
    })
    
    # Glucose level risk
    glucose = float(data.get('avg_glucose_level', 0))
    glucose_risk = 0
    if glucose > 140:
        glucose_risk = 0.20
    elif glucose > 100:
        glucose_risk = 0.10
    factors.append({
        'name': 'Glucose Level',
        'value': glucose,
        'contribution': glucose_risk,
        'unit': 'mg/dL',
        'risk_category': 'High' if glucose > 140 else 'Medium' if glucose > 100 else 'Normal'
    })
    
    # Hypertension risk
    hypertension = int(data.get('hypertension', 0))
    hypertension_risk = 0.25 if hypertension == 1 else 0
    factors.append({
        'name': 'Hypertension',
        'value': hypertension,
        'contribution': hypertension_risk,
        'unit': 'Yes/No',
        'risk_category': 'High' if hypertension == 1 else 'Low'
    })
    
    # Heart disease risk
    heart_disease = int(data.get('heart_disease', 0))
    heart_risk = 0.30 if heart_disease == 1 else 0
    factors.append({
        'name': 'Heart Disease',
        'value': heart_disease,
        'contribution': heart_risk,
        'unit': 'Yes/No',
        'risk_category': 'High' if heart_disease == 1 else 'Low'
    })
    
    # Smoking risk
    smoking = data.get('smoking_status', 'never smoked')
    smoking_risk = 0.15 if smoking == 'smokes' else 0.10 if smoking == 'formerly smoked' else 0
    factors.append({
        'name': 'Smoking Status',
        'value': smoking,
        'contribution': smoking_risk,
        'unit': 'Category',
        'risk_category': 'High' if smoking == 'smokes' else 'Medium' if smoking == 'formerly smoked' else 'Low'
    })
    
    return factors

# ============================================================================
# DASHBOARD TEMPLATE
# ============================================================================

DASHBOARD_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stroke Prediction Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --info-color: #1abc9c;
            --female-color: #e84393;
            --male-color: #0984e3;
            --light-bg: #f8f9fa;
            --dark-bg: #2c3e50;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f6fa;
        }
        
        .dashboard-header {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        
        .stat-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            border-left: 5px solid;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        
        .stat-card.gender-female { border-left-color: var(--female-color); }
        .stat-card.gender-male { border-left-color: var(--male-color); }
        .stat-card.risk-high { border-left-color: var(--danger-color); }
        .stat-card.risk-medium { border-left-color: var(--warning-color); }
        .stat-card.risk-low { border-left-color: var(--success-color); }
        
        .risk-badge {
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            font-size: 0.85rem;
        }
        
        .badge-high { background-color: var(--danger-color); color: white; }
        .badge-medium { background-color: var(--warning-color); color: white; }
        .badge-low { background-color: var(--success-color); color: white; }
        
        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }
        
        .gender-icon {
            font-size: 2rem;
            margin-right: 1rem;
        }
        
        .risk-meter {
            height: 10px;
            border-radius: 5px;
            background: linear-gradient(to right, var(--success-color), var(--warning-color), var(--danger-color));
            margin: 1rem 0;
            position: relative;
        }
        
        .risk-indicator {
            position: absolute;
            top: -5px;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            border: 3px solid var(--dark-bg);
            transform: translateX(-50%);
        }
        
        .patient-row:hover {
            background-color: rgba(52, 152, 219, 0.1);
            cursor: pointer;
        }
        
        .factor-bar {
            height: 8px;
            border-radius: 4px;
            margin: 5px 0;
            background-color: #e0e0e0;
            overflow: hidden;
        }
        
        .factor-fill {
            height: 100%;
            border-radius: 4px;
        }
        
        .nav-tabs .nav-link.active {
            background-color: var(--secondary-color);
            color: white;
            border-color: var(--secondary-color);
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <h1><i class="fas fa-heartbeat"></i> Stroke Prediction Dashboard</h1>
                    <p class="lead mb-0">Real-time monitoring and analysis of stroke risk assessments</p>
                </div>
                <div class="col-md-4 text-end">
                    <span class="badge bg-light text-dark fs-6 p-2">
                        <i class="fas fa-database"></i> Last updated: <span id="last-updated">Just now</span>
                    </span>
                </div>
            </div>
        </div>
    </div>

    <div class="container-fluid">
        <!-- Summary Stats -->
        <div class="row mb-4">
            <div class="col-xl-3 col-md-6">
                <div class="stat-card">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-users gender-icon" style="color: var(--secondary-color);"></i>
                        <div>
                            <h3 class="mb-0" id="total-patients">0</h3>
                            <p class="text-muted mb-0">Total Patients</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="stat-card gender-female">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-female gender-icon" style="color: var(--female-color);"></i>
                        <div>
                            <h3 class="mb-0" id="female-patients">0</h3>
                            <p class="text-muted mb-0">Female Patients</p>
                            <small class="text-muted" id="female-percentage">0%</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="stat-card gender-male">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-male gender-icon" style="color: var(--male-color);"></i>
                        <div>
                            <h3 class="mb-0" id="male-patients">0</h3>
                            <p class="text-muted mb-0">Male Patients</p>
                            <small class="text-muted" id="male-percentage">0%</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-xl-3 col-md-6">
                <div class="stat-card risk-high">
                    <div class="d-flex align-items-center">
                        <i class="fas fa-exclamation-triangle gender-icon" style="color: var(--danger-color);"></i>
                        <div>
                            <h3 class="mb-0" id="high-risk-patients">0</h3>
                            <p class="text-muted mb-0">High Risk Patients</p>
                            <small class="text-muted" id="high-risk-percentage">0%</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="row mb-4">
            <div class="col-lg-8">
                <div class="chart-container">
                    <h4><i class="fas fa-chart-line"></i> Risk Distribution Over Time</h4>
                    <div id="time-series-chart"></div>
                </div>
            </div>
            <div class="col-lg-4">
                <div class="chart-container">
                    <h4><i class="fas fa-chart-pie"></i> Risk Level Distribution</h4>
                    <div id="risk-pie-chart"></div>
                </div>
            </div>
        </div>

        <!-- Gender Analysis -->
        <div class="row mb-4">
            <div class="col-lg-6">
                <div class="chart-container">
                    <h4><i class="fas fa-venus"></i> Female Patients Analysis</h4>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="text-center p-3">
                                <h2 id="female-high-risk">0</h2>
                                <p>High Risk Females</p>
                                <div class="risk-meter">
                                    <div class="risk-indicator" id="female-risk-indicator"></div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div id="female-age-distribution"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="col-lg-6">
                <div class="chart-container">
                    <h4><i class="fas fa-mars"></i> Male Patients Analysis</h4>
                    <div class="row">
                        <div class="col-md-6">
                            <div class="text-center p-3">
                                <h2 id="male-high-risk">0</h2>
                                <p>High Risk Males</p>
                                <div class="risk-meter">
                                    <div class="risk-indicator" id="male-risk-indicator"></div>
                                </div>
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div id="male-age-distribution"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Detailed Statistics -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="chart-container">
                    <h4><i class="fas fa-chart-bar"></i> Detailed Statistics</h4>
                    <ul class="nav nav-tabs" id="statsTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="age-tab" data-bs-toggle="tab" data-bs-target="#age">Age Analysis</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="bmi-tab" data-bs-toggle="tab" data-bs-target="#bmi">BMI Analysis</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="glucose-tab" data-bs-toggle="tab" data-bs-target="#glucose">Glucose Analysis</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="factors-tab" data-bs-toggle="tab" data-bs-target="#factors">Risk Factors</button>
                        </li>
                    </ul>
                    <div class="tab-content p-3" id="statsTabContent">
                        <div class="tab-pane fade show active" id="age">
                            <div id="age-distribution-chart"></div>
                        </div>
                        <div class="tab-pane fade" id="bmi">
                            <div id="bmi-distribution-chart"></div>
                        </div>
                        <div class="tab-pane fade" id="glucose">
                            <div id="glucose-distribution-chart"></div>
                        </div>
                        <div class="tab-pane fade" id="factors">
                            <div id="risk-factors-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Recent Predictions -->
        <div class="row">
            <div class="col-12">
                <div class="chart-container">
                    <h4><i class="fas fa-history"></i> Recent Predictions</h4>
                    <div class="table-responsive">
                        <table class="table table-hover">
                            <thead>
                                <tr>
                                    <th>Patient ID</th>
                                    <th>Gender</th>
                                    <th>Age</th>
                                    <th>Risk Level</th>
                                    <th>Stroke Probability</th>
                                    <th>Hypertension</th>
                                    <th>Heart Disease</th>
                                    <th>Date</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="recent-predictions">
                                <!-- Filled by JavaScript -->
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    
    <!-- Dashboard JavaScript -->
    <script>
        let chartData = null;
        
        // Function to update dashboard data
        async function updateDashboard() {
            try {
                const response = await fetch('/dashboard-data');
                const data = await response.json();
                chartData = data;
                
                // Update summary stats
                document.getElementById('total-patients').textContent = data.total_predictions.toLocaleString();
                document.getElementById('female-patients').textContent = data.gender_distribution?.Female || 0;
                document.getElementById('male-patients').textContent = data.gender_distribution?.Male || 0;
                document.getElementById('high-risk-patients').textContent = data.high_risk_patients;
                
                // Calculate percentages
                const total = data.total_predictions;
                const femalePercent = total > 0 ? ((data.gender_distribution?.Female || 0) / total * 100).toFixed(1) : 0;
                const malePercent = total > 0 ? ((data.gender_distribution?.Male || 0) / total * 100).toFixed(1) : 0;
                const highRiskPercent = total > 0 ? (data.high_risk_patients / total * 100).toFixed(1) : 0;
                
                document.getElementById('female-percentage').textContent = `${femalePercent}%`;
                document.getElementById('male-percentage').textContent = `${malePercent}%`;
                document.getElementById('high-risk-percentage').textContent = `${highRiskPercent}%`;
                
                // Update time indicator
                document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                
                // Update charts
                updateTimeSeriesChart(data);
                updateRiskPieChart(data);
                updateGenderAnalysis(data);
                updateStatistics(data);
                updateRecentPredictions(data);
                
            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }
        
        // Update time series chart
        function updateTimeSeriesChart(data) {
            if (data.predictions_by_date && data.predictions_by_date.length > 0) {
                const dates = data.predictions_by_date.map(d => d.date);
                const counts = data.predictions_by_date.map(d => d.count);
                
                const trace = {
                    x: dates,
                    y: counts,
                    type: 'scatter',
                    mode: 'lines+markers',
                    line: { color: '#3498db', width: 3 },
                    marker: { size: 8 },
                    fill: 'tozeroy',
                    fillcolor: 'rgba(52, 152, 219, 0.2)'
                };
                
                const layout = {
                    title: 'Daily Predictions',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Number of Predictions' },
                    hovermode: 'closest',
                    showlegend: false,
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)'
                };
                
                Plotly.newPlot('time-series-chart', [trace], layout);
            }
        }
        
        // Update risk pie chart
        function updateRiskPieChart(data) {
            if (data.risk_distribution) {
                const labels = Object.keys(data.risk_distribution);
                const values = Object.values(data.risk_distribution);
                const colors = labels.map(label => {
                    if (label.includes('High')) return '#e74c3c';
                    if (label.includes('Medium')) return '#f39c12';
                    if (label.includes('Low')) return '#27ae60';
                    return '#95a5a6';
                });
                
                const trace = {
                    labels: labels,
                    values: values,
                    type: 'pie',
                    hole: 0.4,
                    marker: { colors: colors },
                    textinfo: 'label+percent',
                    hoverinfo: 'label+value+percent'
                };
                
                const layout = {
                    showlegend: true,
                    legend: { orientation: 'h' },
                    plot_bgcolor: 'rgba(0,0,0,0)',
                    paper_bgcolor: 'rgba(0,0,0,0)'
                };
                
                Plotly.newPlot('risk-pie-chart', [trace], layout);
            }
        }
        
        // Update gender analysis
        function updateGenderAnalysis(data) {
            // Update female stats
            document.getElementById('female-high-risk').textContent = 
                data.gender_risk_stats?.Female?.high_risk || 0;
            
            // Update male stats
            document.getElementById('male-high-risk').textContent = 
                data.gender_risk_stats?.Male?.high_risk || 0;
            
            // Update risk indicators
            const femaleRisk = data.gender_risk_stats?.Female?.risk_percentage || 0;
            const maleRisk = data.gender_risk_stats?.Male?.risk_percentage || 0;
            
            document.getElementById('female-risk-indicator').style.left = `${femaleRisk}%`;
            document.getElementById('male-risk-indicator').style.left = `${maleRisk}%`;
            
            // Update age distribution charts
            if (data.age_distribution_by_gender) {
                updateGenderAgeChart('female', data.age_distribution_by_gender.Female || []);
                updateGenderAgeChart('male', data.age_distribution_by_gender.Male || []);
            }
        }
        
        function updateGenderAgeChart(gender, data) {
            const ages = data.map(d => d.age_group);
            const counts = data.map(d => d.count);
            
            const trace = {
                x: ages,
                y: counts,
                type: 'bar',
                marker: {
                    color: gender === 'female' ? '#e84393' : '#0984e3'
                },
                name: gender === 'female' ? 'Female' : 'Male'
            };
            
            const layout = {
                title: `${gender === 'female' ? 'Female' : 'Male'} Age Distribution`,
                xaxis: { title: 'Age Group' },
                yaxis: { title: 'Count' },
                plot_bgcolor: 'rgba(0,0,0,0)',
                paper_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot(`${gender}-age-distribution`, [trace], layout);
        }
        
        // Update statistics
        function updateStatistics(data) {
            // Age distribution
            if (data.age_distribution) {
                const ageTrace = {
                    x: data.age_distribution.map(d => d.age),
                    y: data.age_distribution.map(d => d.count),
                    type: 'histogram',
                    marker: { color: '#3498db' },
                    name: 'Age Distribution'
                };
                
                Plotly.newPlot('age-distribution-chart', [ageTrace], {
                    title: 'Patient Age Distribution',
                    xaxis: { title: 'Age' },
                    yaxis: { title: 'Count' }
                });
            }
            
            // BMI distribution
            if (data.bmi_stats) {
                const bmiTrace = {
                    x: ['Underweight', 'Normal', 'Overweight', 'Obese'],
                    y: [
                        data.bmi_stats.underweight || 0,
                        data.bmi_stats.normal || 0,
                        data.bmi_stats.overweight || 0,
                        data.bmi_stats.obese || 0
                    ],
                    type: 'bar',
                    marker: { color: '#2ecc71' }
                };
                
                Plotly.newPlot('bmi-distribution-chart', [bmiTrace], {
                    title: 'BMI Categories',
                    xaxis: { title: 'BMI Category' },
                    yaxis: { title: 'Count' }
                });
            }
            
            // Glucose distribution
            if (data.glucose_stats) {
                const glucoseTrace = {
                    x: ['Normal', 'Prediabetes', 'Diabetes'],
                    y: [
                        data.glucose_stats.normal || 0,
                        data.glucose_stats.prediabetes || 0,
                        data.glucose_stats.diabetes || 0
                    ],
                    type: 'bar',
                    marker: { color: '#e74c3c' }
                };
                
                Plotly.newPlot('glucose-distribution-chart', [glucoseTrace], {
                    title: 'Glucose Level Categories',
                    xaxis: { title: 'Category' },
                    yaxis: { title: 'Count' }
                });
            }
            
            // Risk factors
            if (data.risk_factors) {
                const factors = Object.keys(data.risk_factors);
                const percentages = Object.values(data.risk_factors);
                
                const factorTrace = {
                    x: percentages,
                    y: factors,
                    type: 'bar',
                    orientation: 'h',
                    marker: { color: '#f39c12' }
                };
                
                Plotly.newPlot('risk-factors-chart', [factorTrace], {
                    title: 'Top Risk Factors',
                    xaxis: { title: 'Percentage of Patients' },
                    yaxis: { title: 'Risk Factor' }
                });
            }
        }
        
        // Update recent predictions table
        function updateRecentPredictions(data) {
            const tbody = document.getElementById('recent-predictions');
            tbody.innerHTML = '';
            
            if (data.recent_predictions && data.recent_predictions.length > 0) {
                data.recent_predictions.forEach(pred => {
                    const row = document.createElement('tr');
                    row.className = 'patient-row';
                    
                    const riskClass = pred.risk_level.includes('High') ? 'badge-high' : 
                                    pred.risk_level.includes('Medium') ? 'badge-medium' : 'badge-low';
                    
                    row.innerHTML = `
                        <td>${pred.patient_id}</td>
                        <td>
                            <i class="fas ${pred.gender === 'Female' ? 'fa-female text-danger' : 'fa-male text-primary'}"></i>
                            ${pred.gender}
                        </td>
                        <td>${pred.age}</td>
                        <td><span class="risk-badge ${riskClass}">${pred.risk_level}</span></td>
                        <td>
                            <div class="progress" style="height: 8px;">
                                <div class="progress-bar bg-danger" style="width: ${pred.stroke_probability * 100}%"></div>
                            </div>
                            <small>${(pred.stroke_probability * 100).toFixed(1)}%</small>
                        </td>
                        <td>
                            <span class="badge ${pred.hypertension ? 'bg-danger' : 'bg-success'}">
                                ${pred.hypertension ? 'Yes' : 'No'}
                            </span>
                        </td>
                        <td>
                            <span class="badge ${pred.heart_disease ? 'bg-danger' : 'bg-success'}">
                                ${pred.heart_disease ? 'Yes' : 'No'}
                            </span>
                        </td>
                        <td>${new Date(pred.timestamp).toLocaleDateString()}</td>
                        <td>
                            <button class="btn btn-sm btn-outline-primary" onclick="viewDetails('${pred.patient_id}')">
                                <i class="fas fa-eye"></i>
                            </button>
                        </td>
                    `;
                    tbody.appendChild(row);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="9" class="text-center">No predictions yet</td></tr>';
            }
        }
        
        // View patient details
        function viewDetails(patientId) {
            window.location.href = `/patient/${patientId}`;
        }
        
        // Initial load and periodic updates
        document.addEventListener('DOMContentLoaded', function() {
            updateDashboard();
            setInterval(updateDashboard, 30000); // Update every 30 seconds
        });
    </script>
</body>
</html>
'''

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def home():
    return send_from_directory('.', 'stroke_prediction.html')

@app.route('/dashboard')
def dashboard():
    """Serve the enhanced dashboard"""
    return render_template_string(DASHBOARD_TEMPLATE)







@app.route('/dashboard-data', methods=['GET'])
def get_dashboard_data():
    """Get comprehensive dashboard data"""
    try:
        conn = sqlite3.connect('stroke_predictions.db')
        cursor = conn.cursor()
        
        # Basic stats
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total = cursor.fetchone()[0] or 0
        
        # Gender distribution
        cursor.execute("SELECT gender, COUNT(*) FROM predictions GROUP BY gender")
        gender_dist = dict(cursor.fetchall())
        
        # Risk level distribution
        cursor.execute("SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level")
        risk_dist = dict(cursor.fetchall())
        
        # High risk patients
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE risk_level IN ('High', 'Very High') 
            AND stroke_probability > 0.5
        """)
        high_risk = cursor.fetchone()[0] or 0
        
        # Predictions by date (last 30 days)
        cursor.execute("""
            SELECT DATE(timestamp) as date, COUNT(*) as count 
            FROM predictions 
            WHERE timestamp >= date('now', '-30 days')
            GROUP BY DATE(timestamp) 
            ORDER BY date
        """)
        predictions_by_date = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Age distribution
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
            ORDER BY 
                CASE age_group
                    WHEN 'Under 30' THEN 1
                    WHEN '30-49' THEN 2
                    WHEN '50-64' THEN 3
                    ELSE 4
                END
        """)
        age_dist = [{'age_group': row[0], 'count': row[1]} for row in cursor.fetchall()]
        
        # Gender-specific age distribution
        cursor.execute("""
            SELECT gender,
                CASE 
                    WHEN age < 30 THEN 'Under 30'
                    WHEN age < 50 THEN '30-49'
                    WHEN age < 65 THEN '50-64'
                    ELSE '65+'
                END as age_group,
                COUNT(*) as count
            FROM predictions 
            GROUP BY gender, age_group
            ORDER BY gender, 
                CASE age_group
                    WHEN 'Under 30' THEN 1
                    WHEN '30-49' THEN 2
                    WHEN '50-64' THEN 3
                    ELSE 4
                END
        """)
        gender_age_data = {}
        for row in cursor.fetchall():
            gender = row[0]
            if gender not in gender_age_data:
                gender_age_data[gender] = []
            gender_age_data[gender].append({'age_group': row[1], 'count': row[2]})
        
        # ============================================================================
        # CORRECTED GENDER RISK STATISTICS SECTION
        # ============================================================================
        gender_risk_stats = {}
        for gender in ['Male', 'Female']:
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN risk_level IN ('High', 'Very High') THEN 1 ELSE 0 END) as high_risk
                FROM predictions 
                WHERE gender = ?
            """, (gender,))
            
            result = cursor.fetchone()
            if result:
                total_gender, high_risk_gender = result
                total_gender = int(total_gender) if total_gender else 0
                high_risk_gender = int(high_risk_gender) if high_risk_gender else 0
            else:
                total_gender, high_risk_gender = 0, 0
            
            # CORRECT PYTHON SYNTAX - NO JavaScript ternary operator
            if total_gender and total_gender > 0:
                if high_risk_gender:
                    risk_percentage = (high_risk_gender / total_gender) * 100
                else:
                    risk_percentage = 0
            else:
                risk_percentage = 0
            
            gender_risk_stats[gender] = {
                'total': total_gender,
                'high_risk': high_risk_gender,
                'risk_percentage': round(risk_percentage, 2)
            }
        # ============================================================================
        
        # BMI statistics
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN bmi < 18.5 THEN 1 ELSE 0 END) as underweight,
                SUM(CASE WHEN bmi >= 18.5 AND bmi < 25 THEN 1 ELSE 0 END) as normal,
                SUM(CASE WHEN bmi >= 25 AND bmi < 30 THEN 1 ELSE 0 END) as overweight,
                SUM(CASE WHEN bmi >= 30 THEN 1 ELSE 0 END) as obese
            FROM predictions 
            WHERE bmi IS NOT NULL
        """)
        bmi_stats = cursor.fetchone()
        bmi_stats_dict = {
            'underweight': bmi_stats[0] or 0 if bmi_stats else 0,
            'normal': bmi_stats[1] or 0 if bmi_stats else 0,
            'overweight': bmi_stats[2] or 0 if bmi_stats else 0,
            'obese': bmi_stats[3] or 0 if bmi_stats else 0
        }
        
        # Glucose statistics
        cursor.execute("""
            SELECT 
                SUM(CASE WHEN avg_glucose_level < 100 THEN 1 ELSE 0 END) as normal,
                SUM(CASE WHEN avg_glucose_level >= 100 AND avg_glucose_level < 126 THEN 1 ELSE 0 END) as prediabetes,
                SUM(CASE WHEN avg_glucose_level >= 126 THEN 1 ELSE 0 END) as diabetes
            FROM predictions 
            WHERE avg_glucose_level IS NOT NULL
        """)
        glucose_stats = cursor.fetchone()
        glucose_stats_dict = {
            'normal': glucose_stats[0] or 0 if glucose_stats else 0,
            'prediabetes': glucose_stats[1] or 0 if glucose_stats else 0,
            'diabetes': glucose_stats[2] or 0 if glucose_stats else 0
        }
        
        # Risk factors prevalence
        cursor.execute("""
            SELECT 
                ROUND(AVG(hypertension) * 100, 1) as hypertension_pct,
                ROUND(AVG(heart_disease) * 100, 1) as heart_disease_pct,
                ROUND(AVG(CASE WHEN smoking_status IN ('smokes', 'formerly smoked') THEN 1 ELSE 0 END) * 100, 1) as smoking_pct,
                ROUND(AVG(CASE WHEN age > 65 THEN 1 ELSE 0 END) * 100, 1) as elderly_pct
            FROM predictions
        """)
        risk_factors = cursor.fetchone()
        risk_factors_dict = {
            'Hypertension': risk_factors[0] or 0 if risk_factors else 0,
            'Heart Disease': risk_factors[1] or 0 if risk_factors else 0,
            'Smoking History': risk_factors[2] or 0 if risk_factors else 0,
            'Age > 65': risk_factors[3] or 0 if risk_factors else 0
        }
        
        # Recent predictions (last 10)
        recent_predictions = pd.read_sql_query("""
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 10
        """, conn).to_dict('records')
        
        conn.close()
        
        return jsonify({
            'total_predictions': total,
            'gender_distribution': gender_dist,
            'risk_distribution': risk_dist,
            'high_risk_patients': high_risk,
            'predictions_by_date': predictions_by_date,
            'age_distribution': age_dist,
            'age_distribution_by_gender': gender_age_data,
            'gender_risk_stats': gender_risk_stats,
            'bmi_stats': bmi_stats_dict,
            'glucose_stats': glucose_stats_dict,
            'risk_factors': risk_factors_dict,
            'recent_predictions': recent_predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"Error getting dashboard data: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500







@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)
        
        df = pd.DataFrame([data])
        
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0]
        
        # Calculate risk level with more granularity
        stroke_prob = prob[1]
        if stroke_prob < 0.2:
            risk = 'Very Low'
        elif stroke_prob < 0.4:
            risk = 'Low'
        elif stroke_prob < 0.6:
            risk = 'Medium'
        elif stroke_prob < 0.8:
            risk = 'High'
        else:
            risk = 'Very High'
        
        # Calculate risk factors
        risk_factors = calculate_risk_factors(data)
        
        result = {
            'prediction': int(pred),
            'probability': {'no_stroke': float(prob[0]), 'stroke': float(prob[1])},
            'risk_level': risk,
            'confidence': float(max(prob)) * 100,
            'risk_factors': risk_factors,
            'interpretation': get_interpretation(stroke_prob, risk_factors),
            'recommendations': get_recommendations(stroke_prob, risk_factors)
        }
        
        print("Prediction result:", result)
        
        # SAVE TO DATABASE
        prediction_id, patient_id = save_prediction_to_db(data, result)
        
        # Add database info to result
        if prediction_id:
            result['prediction_id'] = prediction_id
            result['patient_id'] = patient_id
            result['saved_to_database'] = True
        
        return jsonify(result)
        
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)}), 500

def get_interpretation(stroke_prob, risk_factors):
    """Generate interpretation based on probability and risk factors"""
    if stroke_prob < 0.2:
        return "Low stroke risk. Maintain healthy lifestyle."
    elif stroke_prob < 0.4:
        return "Moderate risk. Monitor risk factors regularly."
    elif stroke_prob < 0.6:
        return "Elevated risk. Consider lifestyle changes and regular checkups."
    elif stroke_prob < 0.8:
        return "High risk. Consult healthcare provider for preventive measures."
    else:
        return "Very high risk. Immediate medical consultation recommended."

def get_recommendations(stroke_prob, risk_factors):
    """Generate personalized recommendations"""
    recommendations = []
    
    if stroke_prob > 0.4:
        recommendations.append("Regular blood pressure monitoring")
        recommendations.append("Annual health checkups")
    
    # Check individual risk factors
    for factor in risk_factors:
        if factor['contribution'] > 0.1:
            if factor['name'] == 'BMI' and factor['value'] > 25:
                recommendations.append("Weight management program")
            elif factor['name'] == 'Glucose Level' and factor['value'] > 100:
                recommendations.append("Diabetes screening and management")
            elif factor['name'] == 'Smoking Status' and factor['value'] in ['smokes', 'formerly smoked']:
                recommendations.append("Smoking cessation support")
    
    if not recommendations:
        recommendations.append("Maintain current healthy lifestyle")
        recommendations.append("Regular exercise and balanced diet")
    
    return recommendations[:5]  # Return top 5 recommendations

# Keep other endpoints (history, patient, stats, delete) as they were...

if __name__ == '__main__':
    print("\n" + "="*60)
    print(" ENHANCED STROKE PREDICTION APP IS RUNNING!")
    print("="*60)
    print(" Backend: http://127.0.0.1:5000")
    print(" Dashboard: http://127.0.0.1:5000/dashboard")
    print(" Database: stroke_predictions.db")
    print("="*60)
    print("\n Available Endpoints:")
    print("  GET    /dashboard           - Enhanced dashboard")
    print("  GET    /dashboard-data      - Dashboard JSON data")
    print("  POST   /predict             - Make prediction & save to DB")
    print("  GET    /history             - View all predictions")
    print("  GET    /patient/<id>        - View patient history")
    print("  GET    /stats               - View statistics")
    print("  DELETE /prediction/<id>     - Delete prediction")
    print("="*60 + "\n")
    app.run(debug=True, port=5000, host='127.0.0.1')