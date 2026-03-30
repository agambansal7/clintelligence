#!/usr/bin/env python3
"""
ML MODEL TRAINING - Day 2 Sprint

Train real machine learning models on the ingested trial data.

Models to build:
1. Amendment Predictor - Will this protocol be amended?
2. Enrollment Delay Predictor - Will enrollment be delayed?
3. Termination Predictor - Will this trial be terminated early?
4. Enrollment Velocity Predictor - How fast will this trial enroll?

Usage:
    python day2_train_models.py --db ./data/trials.db --output ./models
"""

import sqlite3
import json
import pickle
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


class TrialFeatureExtractor:
    """Extract ML features from trial data."""
    
    def __init__(self):
        self.tfidf_eligibility = TfidfVectorizer(max_features=200, stop_words='english')
        self.tfidf_endpoints = TfidfVectorizer(max_features=100, stop_words='english')
        self.phase_encoder = LabelEncoder()
        self.sponsor_encoder = LabelEncoder()
        self.fitted = False
    
    def extract_eligibility_features(self, criteria: str) -> Dict[str, Any]:
        """Extract features from eligibility criteria text."""
        if not criteria:
            return {
                "num_inclusion": 0,
                "num_exclusion": 0,
                "criteria_length": 0,
                "has_age_restriction": 0,
                "has_lab_values": 0,
                "has_prior_therapy": 0,
                "has_performance_status": 0,
                "complexity_score": 0,
            }
        
        criteria_lower = criteria.lower()
        
        # Count inclusion/exclusion criteria
        inclusion_match = re.search(r'inclusion criteria[:\s]*(.*?)(?=exclusion|$)', 
                                    criteria_lower, re.DOTALL)
        exclusion_match = re.search(r'exclusion criteria[:\s]*(.*?)$', 
                                    criteria_lower, re.DOTALL)
        
        num_inclusion = len(re.findall(r'\n\s*[-•*]\s*', inclusion_match.group(1))) if inclusion_match else 0
        num_exclusion = len(re.findall(r'\n\s*[-•*]\s*', exclusion_match.group(1))) if exclusion_match else 0
        
        # Detect specific restrictions
        features = {
            "num_inclusion": num_inclusion,
            "num_exclusion": num_exclusion,
            "criteria_length": len(criteria),
            "has_age_restriction": 1 if re.search(r'\d+\s*years?', criteria_lower) else 0,
            "has_lab_values": 1 if re.search(r'(hba1c|egfr|alt|ast|bilirubin|creatinine)', criteria_lower) else 0,
            "has_prior_therapy": 1 if re.search(r'prior.*(therapy|treatment|chemotherapy)', criteria_lower) else 0,
            "has_performance_status": 1 if re.search(r'(ecog|karnofsky|performance)', criteria_lower) else 0,
            "complexity_score": num_inclusion + (num_exclusion * 1.5),  # Exclusions weighted more
        }
        
        return features
    
    def extract_endpoint_features(self, primary_outcomes: str, secondary_outcomes: str) -> Dict[str, Any]:
        """Extract features from endpoint definitions."""
        try:
            primary = json.loads(primary_outcomes) if primary_outcomes else []
            secondary = json.loads(secondary_outcomes) if secondary_outcomes else []
        except:
            primary, secondary = [], []
        
        # Combine all endpoint text
        all_text = " ".join([
            e.get("measure", "") + " " + e.get("timeFrame", "")
            for e in primary + secondary
        ]).lower()
        
        features = {
            "num_primary_endpoints": len(primary),
            "num_secondary_endpoints": len(secondary),
            "has_survival_endpoint": 1 if "survival" in all_text else 0,
            "has_response_endpoint": 1 if "response" in all_text else 0,
            "has_safety_endpoint": 1 if re.search(r'(adverse|safety|tolerability)', all_text) else 0,
            "has_qol_endpoint": 1 if re.search(r'(quality of life|qol|hrqol)', all_text) else 0,
            "has_biomarker_endpoint": 1 if re.search(r'(hba1c|ldl|biomarker)', all_text) else 0,
            "endpoint_complexity": len(primary) + len(secondary) * 0.5,
        }
        
        # Extract timeframe if possible
        timeframe_months = 0
        for endpoint in primary:
            tf = endpoint.get("timeFrame", "").lower()
            month_match = re.search(r'(\d+)\s*months?', tf)
            week_match = re.search(r'(\d+)\s*weeks?', tf)
            if month_match:
                timeframe_months = max(timeframe_months, int(month_match.group(1)))
            elif week_match:
                timeframe_months = max(timeframe_months, int(week_match.group(1)) / 4)
        
        features["primary_timeframe_months"] = timeframe_months
        
        return features
    
    def extract_trial_features(self, row: pd.Series) -> Dict[str, Any]:
        """Extract all features from a trial row."""
        features = {}
        
        # Basic features
        features["enrollment"] = row.get("enrollment", 0) or 0
        features["num_locations"] = len(json.loads(row.get("locations", "[]") or "[]"))
        
        # Phase encoding
        phase = row.get("phase", "")
        features["is_phase1"] = 1 if "PHASE1" in phase else 0
        features["is_phase2"] = 1 if "PHASE2" in phase else 0
        features["is_phase3"] = 1 if "PHASE3" in phase else 0
        features["is_phase4"] = 1 if "PHASE4" in phase else 0
        
        # Study type
        features["is_interventional"] = 1 if row.get("study_type") == "INTERVENTIONAL" else 0
        
        # Eligibility features
        elig_features = self.extract_eligibility_features(row.get("eligibility_criteria", ""))
        features.update(elig_features)
        
        # Endpoint features
        endpoint_features = self.extract_endpoint_features(
            row.get("primary_outcomes", ""),
            row.get("secondary_outcomes", "")
        )
        features.update(endpoint_features)
        
        # Sponsor features (simplified)
        sponsor = row.get("sponsor", "").lower()
        features["is_pharma_sponsor"] = 1 if any(p in sponsor for p in 
            ["pfizer", "novartis", "roche", "merck", "johnson", "astrazeneca", 
             "sanofi", "gsk", "lilly", "abbvie", "bristol", "amgen"]) else 0
        features["is_academic_sponsor"] = 1 if any(p in sponsor for p in 
            ["university", "hospital", "institute", "college", "medical center"]) else 0
        
        # Sites per patient ratio
        if features["enrollment"] > 0:
            features["sites_per_100_patients"] = (features["num_locations"] / features["enrollment"]) * 100
        else:
            features["sites_per_100_patients"] = 0
        
        return features
    
    def create_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target labels for training."""
        labels = pd.DataFrame(index=df.index)
        
        # Termination label (clear signal)
        labels["terminated"] = df["status"].isin(["TERMINATED", "WITHDRAWN", "SUSPENDED"]).astype(int)
        
        # Completed successfully
        labels["completed"] = (df["status"] == "COMPLETED").astype(int)
        
        # Has results (proxy for success)
        labels["has_results"] = df["has_results"].fillna(0).astype(int)
        
        return labels


class AmendmentPredictor:
    """Predict probability of protocol amendments."""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_extractor = TrialFeatureExtractor()
        self.feature_names = None
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from dataframe."""
        features_list = []
        
        for _, row in df.iterrows():
            features = self.feature_extractor.extract_trial_features(row)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        self.feature_names = features_df.columns.tolist()
        
        # Fill NaN
        features_df = features_df.fillna(0)
        
        return features_df.values
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the amendment prediction model."""
        print("Preparing features...")
        X = self.prepare_features(df)
        
        # Create proxy label: terminated OR very short trials often had issues
        # This is a proxy since we don't have actual amendment data
        y = (
            (df["status"].isin(["TERMINATED", "WITHDRAWN"])) |
            ((df["status"] == "COMPLETED") & (df["has_results"] == 0))
        ).astype(int).values
        
        print(f"Dataset: {len(X)} samples, {sum(y)} positive")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        results = {
            "accuracy": (y_pred == y_test).mean(),
            "auc_roc": roc_auc_score(y_test, y_prob),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)),
        }
        
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"AUC-ROC: {results['auc_roc']:.3f}")
        
        return results
    
    def predict(self, trial_data: Dict) -> Dict[str, Any]:
        """Predict amendment probability for a new trial."""
        features = self.feature_extractor.extract_trial_features(pd.Series(trial_data))
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0, 1]
        
        # Get top risk factors
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        feature_values = dict(zip(self.feature_names, X[0]))
        
        risk_factors = []
        for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
            if feature_values[feat] > 0:
                risk_factors.append({
                    "feature": feat,
                    "importance": imp,
                    "value": feature_values[feat]
                })
        
        return {
            "amendment_probability": float(prob),
            "risk_level": "high" if prob > 0.6 else "medium" if prob > 0.3 else "low",
            "top_risk_factors": risk_factors
        }
    
    def save(self, path: str):
        """Save model to file."""
        with open(path, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AmendmentPredictor':
        """Load model from file."""
        predictor = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        predictor.model = data["model"]
        predictor.scaler = data["scaler"]
        predictor.feature_names = data["feature_names"]
        return predictor


class TerminationPredictor:
    """Predict probability of early trial termination."""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight="balanced"  # Handle imbalanced data
        )
        self.scaler = StandardScaler()
        self.feature_extractor = TrialFeatureExtractor()
        self.feature_names = None
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix."""
        features_list = []
        for _, row in df.iterrows():
            features = self.feature_extractor.extract_trial_features(row)
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list).fillna(0)
        self.feature_names = features_df.columns.tolist()
        return features_df.values
    
    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train the termination prediction model."""
        # Filter to only terminated or completed trials
        df_filtered = df[df["status"].isin([
            "TERMINATED", "WITHDRAWN", "SUSPENDED", "COMPLETED"
        ])].copy()
        
        print(f"Training on {len(df_filtered)} trials with known outcomes")
        
        X = self.prepare_features(df_filtered)
        y = df_filtered["status"].isin(["TERMINATED", "WITHDRAWN", "SUSPENDED"]).astype(int).values
        
        print(f"Positive samples (terminated): {sum(y)}")
        print(f"Negative samples (completed): {len(y) - sum(y)}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        results = {
            "accuracy": (y_pred == y_test).mean(),
            "auc_roc": roc_auc_score(y_test, y_prob),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
            "feature_importance": dict(zip(self.feature_names, self.model.feature_importances_)),
        }
        
        print(f"Accuracy: {results['accuracy']:.3f}")
        print(f"AUC-ROC: {results['auc_roc']:.3f}")
        
        # Top features
        print("\nTop 10 predictive features:")
        sorted_features = sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        for feat, imp in sorted_features[:10]:
            print(f"  {feat}: {imp:.4f}")
        
        return results
    
    def predict(self, trial_data: Dict) -> Dict[str, Any]:
        """Predict termination probability."""
        features = self.feature_extractor.extract_trial_features(pd.Series(trial_data))
        X = np.array([[features.get(f, 0) for f in self.feature_names]])
        X_scaled = self.scaler.transform(X)
        
        prob = self.model.predict_proba(X_scaled)[0, 1]
        
        return {
            "termination_probability": float(prob),
            "risk_level": "high" if prob > 0.4 else "medium" if prob > 0.2 else "low",
        }
    
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_names": self.feature_names,
            }, f)
    
    @classmethod
    def load(cls, path: str) -> 'TerminationPredictor':
        predictor = cls()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        predictor.model = data["model"]
        predictor.scaler = data["scaler"]
        predictor.feature_names = data["feature_names"]
        return predictor


def load_data(db_path: str) -> pd.DataFrame:
    """Load trial data from SQLite database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM trials", conn)
    conn.close()
    print(f"Loaded {len(df)} trials from database")
    return df


def main():
    parser = argparse.ArgumentParser(description="Train ML models for TrialIntel")
    parser.add_argument("--db", required=True, help="Path to trials database")
    parser.add_argument("--output", default="./models", help="Output directory for models")
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print("="*60)
    print("LOADING DATA")
    print("="*60)
    df = load_data(args.db)
    
    # Train Amendment Predictor
    print("\n" + "="*60)
    print("TRAINING AMENDMENT PREDICTOR")
    print("="*60)
    amendment_model = AmendmentPredictor()
    amendment_results = amendment_model.train(df)
    amendment_model.save(os.path.join(args.output, "amendment_predictor.pkl"))
    
    # Train Termination Predictor
    print("\n" + "="*60)
    print("TRAINING TERMINATION PREDICTOR")
    print("="*60)
    termination_model = TerminationPredictor()
    termination_results = termination_model.train(df)
    termination_model.save(os.path.join(args.output, "termination_predictor.pkl"))
    
    # Save results summary
    results_summary = {
        "training_date": datetime.now().isoformat(),
        "total_trials": len(df),
        "amendment_model": {
            "accuracy": amendment_results["accuracy"],
            "auc_roc": amendment_results["auc_roc"],
        },
        "termination_model": {
            "accuracy": termination_results["accuracy"],
            "auc_roc": termination_results["auc_roc"],
        },
    }
    
    with open(os.path.join(args.output, "training_results.json"), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Models saved to: {args.output}")
    print(f"Amendment Predictor AUC: {amendment_results['auc_roc']:.3f}")
    print(f"Termination Predictor AUC: {termination_results['auc_roc']:.3f}")


if __name__ == "__main__":
    main()
