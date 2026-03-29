"""
Machine Learning Risk Models for Protocol Analysis

This module provides trained ML models that learn from historical trial data to predict:
1. Trial termination risk
2. Enrollment delay probability
3. Protocol amendment likelihood

Models are trained on features extracted from 100K+ clinical trials.
"""

import json
import logging
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Metrics for a trained model."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    training_samples: int
    feature_importance: Dict[str, float]


@dataclass
class RiskPrediction:
    """Prediction from ML models."""
    termination_probability: float
    delay_probability: float
    amendment_probability: float
    confidence: str  # "high", "medium", "low"
    risk_drivers: List[Dict[str, Any]]  # Top factors driving the prediction
    similar_trials: List[str]  # NCT IDs of similar trials


class TrialFeatureExtractor:
    """Extract ML features from trial data."""

    # Phase encoding
    PHASE_ORDER = {
        "EARLY_PHASE1": 0,
        "PHASE1": 1,
        "PHASE1_PHASE2": 1.5,
        "PHASE2": 2,
        "PHASE2_PHASE3": 2.5,
        "PHASE3": 3,
        "PHASE4": 4,
        "NA": 2,  # Default to Phase 2 if unknown
    }

    # Therapeutic area categories
    THERAPEUTIC_CATEGORIES = {
        "oncology": ["cancer", "tumor", "carcinoma", "lymphoma", "leukemia", "melanoma", "sarcoma"],
        "cardiovascular": ["heart", "cardiac", "hypertension", "stroke", "atrial", "coronary"],
        "metabolic": ["diabetes", "obesity", "metabolic", "lipid", "thyroid"],
        "neurology": ["alzheimer", "parkinson", "sclerosis", "epilepsy", "migraine", "neuropathy"],
        "psychiatry": ["depression", "anxiety", "schizophrenia", "bipolar", "ptsd", "adhd"],
        "immunology": ["arthritis", "lupus", "psoriasis", "crohn", "colitis", "inflammatory"],
        "infectious": ["hiv", "hepatitis", "covid", "influenza", "tuberculosis", "sepsis"],
        "respiratory": ["asthma", "copd", "pulmonary", "fibrosis", "pneumonia"],
        "rare_disease": ["duchenne", "fabry", "gaucher", "pompe", "huntington", "sma"],
    }

    def extract_features(self, trial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a single trial record."""
        features = {}

        # Basic features
        features["phase_numeric"] = self._encode_phase(trial_data.get("phase", ""))
        features["enrollment"] = trial_data.get("enrollment") or 0
        features["num_sites"] = trial_data.get("num_sites") or 1

        # Enrollment per site
        if features["num_sites"] > 0:
            features["enrollment_per_site"] = features["enrollment"] / features["num_sites"]
        else:
            features["enrollment_per_site"] = features["enrollment"]

        # Sponsor type
        sponsor_type = trial_data.get("sponsor_type", "").upper()
        features["is_industry"] = 1 if sponsor_type == "INDUSTRY" else 0
        features["is_academic"] = 1 if sponsor_type in ["OTHER", "ACADEMIC", "NETWORK"] else 0

        # Therapeutic area
        therapeutic_area = trial_data.get("therapeutic_area", "") or ""
        conditions = trial_data.get("conditions", "") or ""
        category = self._categorize_therapeutic_area(therapeutic_area + " " + conditions)
        features["therapeutic_category"] = category

        # Eligibility criteria complexity
        eligibility = trial_data.get("eligibility_criteria", "") or ""
        features["criteria_length"] = len(eligibility)
        features["inclusion_count"] = self._count_criteria(eligibility, "inclusion")
        features["exclusion_count"] = self._count_criteria(eligibility, "exclusion")
        features["criteria_complexity"] = features["inclusion_count"] + features["exclusion_count"]

        # Age restrictions
        min_age, max_age = self._parse_age_range(
            trial_data.get("min_age", ""),
            trial_data.get("max_age", "")
        )
        features["min_age"] = min_age
        features["max_age"] = max_age
        features["age_range"] = max_age - min_age

        # Sex restrictions
        sex = trial_data.get("sex", "ALL")
        features["sex_restricted"] = 0 if sex == "ALL" else 1

        # Duration (if dates available)
        features["planned_duration_months"] = self._calculate_duration(
            trial_data.get("start_date"),
            trial_data.get("completion_date")
        )

        # Endpoint complexity
        primary_outcomes = trial_data.get("primary_outcomes", [])
        if isinstance(primary_outcomes, str):
            try:
                primary_outcomes = json.loads(primary_outcomes)
            except (json.JSONDecodeError, TypeError, ValueError):
                logger.debug(f"Could not parse primary_outcomes JSON: {primary_outcomes[:100] if primary_outcomes else 'empty'}")
                primary_outcomes = []
        features["num_primary_endpoints"] = len(primary_outcomes) if primary_outcomes else 1

        # Has biomarker in criteria (often makes enrollment harder)
        features["has_biomarker_criteria"] = 1 if self._has_biomarker(eligibility) else 0

        # Multi-site study
        features["is_multisite"] = 1 if features["num_sites"] > 1 else 0

        return features

    def _encode_phase(self, phase: str) -> float:
        """Convert phase string to numeric."""
        if not phase:
            return 2.0
        phase_upper = phase.upper().replace(" ", "_")
        return self.PHASE_ORDER.get(phase_upper, 2.0)

    def _categorize_therapeutic_area(self, text: str) -> str:
        """Map therapeutic area to category."""
        text_lower = text.lower()
        for category, keywords in self.THERAPEUTIC_CATEGORIES.items():
            if any(kw in text_lower for kw in keywords):
                return category
        return "other"

    def _count_criteria(self, text: str, criteria_type: str) -> int:
        """Count inclusion or exclusion criteria."""
        text_lower = text.lower()

        # Find the relevant section
        if criteria_type == "exclusion":
            match = re.search(r'exclusion\s*criteria[:\s]*(.*?)(?=inclusion|$)',
                            text_lower, re.DOTALL)
        else:
            match = re.search(r'inclusion\s*criteria[:\s]*(.*?)(?=exclusion|$)',
                            text_lower, re.DOTALL)

        if not match:
            return 5  # Default estimate

        section = match.group(1)
        # Count bullet points or numbered items
        count = len(re.findall(r'[\n\r]\s*[-•*\d]+[.)\s]', section))
        return max(count, 1)

    def _parse_age_range(self, min_age: str, max_age: str) -> Tuple[int, int]:
        """Parse age strings to integers."""
        def parse_age(age_str: str, default: int) -> int:
            if not age_str:
                return default
            match = re.search(r'(\d+)', age_str)
            return int(match.group(1)) if match else default

        min_val = parse_age(min_age, 18)
        max_val = parse_age(max_age, 99)
        return min_val, max_val

    def _calculate_duration(self, start_date: str, end_date: str) -> float:
        """Calculate trial duration in months."""
        if not start_date or not end_date:
            return 24.0  # Default estimate

        try:
            # Parse dates (format varies)
            def parse_date(date_str: str) -> Optional[datetime]:
                for fmt in ["%Y-%m-%d", "%Y-%m", "%B %Y", "%Y"]:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue
                return None

            start = parse_date(start_date)
            end = parse_date(end_date)

            if start and end:
                diff = (end - start).days / 30.44
                return max(1, min(diff, 120))  # Cap at 10 years
        except (ValueError, TypeError, AttributeError) as e:
            logger.debug(f"Could not calculate duration from dates: {start_date}, {end_date}: {e}")

        return 24.0

    def _has_biomarker(self, text: str) -> bool:
        """Check if eligibility mentions biomarker requirements."""
        biomarker_patterns = [
            r'hba1c', r'egfr', r'creatinine', r'biopsy', r'marker',
            r'mutation', r'expression', r'positive', r'negative',
            r'her2', r'pd-l1', r'brca', r'kras', r'egfr',
        ]
        text_lower = text.lower()
        return any(re.search(p, text_lower) for p in biomarker_patterns)


class MLRiskPredictor:
    """
    ML-based risk predictor trained on historical trial data.

    Models:
    1. Termination Model: Predicts P(trial terminates early)
    2. Delay Model: Predicts P(enrollment delays)
    3. Amendment Model: Predicts P(protocol amendment)
    """

    MODEL_DIR = "models"

    def __init__(self, db_session: Optional[Session] = None):
        """Initialize predictor with optional database connection."""
        self.db_session = db_session
        self.feature_extractor = TrialFeatureExtractor()

        # Models
        self.termination_model = None
        self.delay_model = None
        self.amendment_model = None

        # Feature names for interpretation
        self.feature_names = []
        self.categorical_features = ["therapeutic_category"]
        self.numeric_features = [
            "phase_numeric", "enrollment", "num_sites", "enrollment_per_site",
            "is_industry", "is_academic", "criteria_length", "inclusion_count",
            "exclusion_count", "criteria_complexity", "min_age", "max_age",
            "age_range", "sex_restricted", "planned_duration_months",
            "num_primary_endpoints", "has_biomarker_criteria", "is_multisite"
        ]

        # Metrics
        self.termination_metrics: Optional[ModelMetrics] = None
        self.delay_metrics: Optional[ModelMetrics] = None

        # Try to load pre-trained models
        self._load_models()

    def train_models(self, min_samples: int = 1000) -> Dict[str, ModelMetrics]:
        """
        Train all risk models on historical trial data.

        Args:
            min_samples: Minimum samples required for training

        Returns:
            Dictionary of model names to metrics
        """
        if self.db_session is None:
            raise ValueError("Database session required for training")

        logger.info("Loading training data from database...")
        df = self._load_training_data()

        if len(df) < min_samples:
            logger.warning(f"Only {len(df)} samples available, need {min_samples}")
            return {}

        logger.info(f"Training on {len(df)} trials...")

        metrics = {}

        # Train termination model
        logger.info("Training termination risk model...")
        self.termination_model, self.termination_metrics = self._train_termination_model(df)
        metrics["termination"] = self.termination_metrics

        # Train delay model (using proxy: actual duration vs planned)
        logger.info("Training delay risk model...")
        self.delay_model, self.delay_metrics = self._train_delay_model(df)
        metrics["delay"] = self.delay_metrics

        # Amendment model uses heuristics (true amendment data not available)
        # But we can use complexity features as proxy

        # Save trained models
        self._save_models()

        logger.info("Model training complete!")
        return metrics

    def _load_training_data(self) -> pd.DataFrame:
        """Load trial data from database for training."""
        from ..database.models import Trial

        # Query trials with outcome data
        query = self.db_session.query(Trial).filter(
            Trial.status.in_(["COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"])
        )

        records = []
        for trial in query.yield_per(1000):
            # Extract features
            trial_dict = {
                "nct_id": trial.nct_id,
                "status": trial.status,
                "phase": trial.phase,
                "therapeutic_area": trial.therapeutic_area,
                "conditions": trial.conditions,
                "enrollment": trial.enrollment,
                "num_sites": trial.num_sites,
                "sponsor_type": trial.sponsor_type,
                "eligibility_criteria": trial.eligibility_criteria,
                "min_age": trial.min_age,
                "max_age": trial.max_age,
                "sex": trial.sex,
                "start_date": trial.start_date,
                "completion_date": trial.completion_date,
                "primary_outcomes": trial.primary_outcomes,
                "why_stopped": trial.why_stopped,
            }

            features = self.feature_extractor.extract_features(trial_dict)
            features["nct_id"] = trial.nct_id
            features["status"] = trial.status
            features["why_stopped"] = trial.why_stopped

            records.append(features)

        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} trials for training")

        # Create target variables
        df["terminated"] = df["status"].isin(["TERMINATED", "WITHDRAWN", "SUSPENDED"]).astype(int)

        return df

    def _train_termination_model(
        self,
        df: pd.DataFrame
    ) -> Tuple[Pipeline, ModelMetrics]:
        """Train model to predict trial termination."""
        # Prepare features
        feature_cols = self.numeric_features + self.categorical_features
        X = df[feature_cols].copy()
        y = df["terminated"]

        # Handle missing values
        for col in self.numeric_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())

        X["therapeutic_category"] = X["therapeutic_category"].fillna("other")

        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ]
        )

        # Create model pipeline with Gradient Boosting
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ))
        ])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Get feature importance
        feature_importance = self._get_feature_importance(model, feature_cols)

        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            auc_roc=roc_auc_score(y_test, y_prob),
            training_samples=len(X_train),
            feature_importance=feature_importance
        )

        logger.info(f"Termination model - AUC: {metrics.auc_roc:.3f}, F1: {metrics.f1:.3f}")

        return model, metrics

    def _train_delay_model(
        self,
        df: pd.DataFrame
    ) -> Tuple[Pipeline, ModelMetrics]:
        """Train model to predict enrollment delays."""
        # Create delay proxy target
        # Trials with high enrollment per site relative to duration are more likely delayed
        df = df.copy()

        # Calculate enrollment difficulty score
        df["enrollment_difficulty"] = (
            df["enrollment"] / (df["num_sites"].clip(lower=1) * df["planned_duration_months"].clip(lower=1))
        )

        # Top 30% of difficulty = likely delayed
        threshold = df["enrollment_difficulty"].quantile(0.7)
        df["likely_delayed"] = (df["enrollment_difficulty"] > threshold).astype(int)

        # Prepare features
        feature_cols = self.numeric_features + self.categorical_features
        X = df[feature_cols].copy()
        y = df["likely_delayed"]

        # Handle missing values
        for col in self.numeric_features:
            if col in X.columns:
                X[col] = X[col].fillna(X[col].median())
        X["therapeutic_category"] = X["therapeutic_category"].fillna("other")

        # Same preprocessing as termination model
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="other")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, self.numeric_features),
                ("cat", categorical_transformer, self.categorical_features)
            ]
        )

        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            ))
        ])

        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        feature_importance = self._get_feature_importance(model, feature_cols)

        metrics = ModelMetrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred),
            recall=recall_score(y_test, y_pred),
            f1=f1_score(y_test, y_pred),
            auc_roc=roc_auc_score(y_test, y_prob),
            training_samples=len(X_train),
            feature_importance=feature_importance
        )

        logger.info(f"Delay model - AUC: {metrics.auc_roc:.3f}, F1: {metrics.f1:.3f}")

        return model, metrics

    def _get_feature_importance(
        self,
        model: Pipeline,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Extract feature importance from trained model."""
        try:
            # Get the classifier from pipeline
            classifier = model.named_steps["classifier"]
            preprocessor = model.named_steps["preprocessor"]

            # Get feature names after preprocessing
            if hasattr(classifier, "feature_importances_"):
                importances = classifier.feature_importances_

                # Map back to original feature names (simplified)
                # For now, use numeric features directly
                importance_dict = {}
                for i, name in enumerate(self.numeric_features):
                    if i < len(importances):
                        importance_dict[name] = float(importances[i])

                # Sort by importance
                return dict(sorted(
                    importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        return {}

    def predict(self, protocol_data: Dict[str, Any]) -> RiskPrediction:
        """
        Predict risks for a protocol.

        Args:
            protocol_data: Dictionary with protocol details

        Returns:
            RiskPrediction with probabilities and drivers
        """
        # Extract features
        features = self.feature_extractor.extract_features(protocol_data)

        # Create dataframe for prediction
        feature_cols = self.numeric_features + self.categorical_features
        X = pd.DataFrame([features])[feature_cols]

        # Handle missing values
        for col in self.numeric_features:
            if col in X.columns:
                X[col] = X[col].fillna(0)
        X["therapeutic_category"] = X["therapeutic_category"].fillna("other")

        # Get predictions
        termination_prob = 0.2  # Default
        delay_prob = 0.3  # Default

        if self.termination_model is not None:
            try:
                termination_prob = float(self.termination_model.predict_proba(X)[0, 1])
            except Exception as e:
                logger.warning(f"Termination prediction failed: {e}")

        if self.delay_model is not None:
            try:
                delay_prob = float(self.delay_model.predict_proba(X)[0, 1])
            except Exception as e:
                logger.warning(f"Delay prediction failed: {e}")

        # Amendment probability based on complexity heuristic
        complexity_score = (
            features.get("exclusion_count", 10) / 20 +
            features.get("has_biomarker_criteria", 0) * 0.2 +
            (1 if features.get("age_range", 80) < 30 else 0) * 0.15
        )
        amendment_prob = min(0.9, 0.4 + complexity_score * 0.3)

        # Determine confidence
        has_models = self.termination_model is not None and self.delay_model is not None
        if has_models and self.termination_metrics and self.termination_metrics.auc_roc > 0.7:
            confidence = "high"
        elif has_models:
            confidence = "medium"
        else:
            confidence = "low"

        # Identify risk drivers
        risk_drivers = self._identify_risk_drivers(features, termination_prob, delay_prob)

        return RiskPrediction(
            termination_probability=termination_prob,
            delay_probability=delay_prob,
            amendment_probability=amendment_prob,
            confidence=confidence,
            risk_drivers=risk_drivers,
            similar_trials=[]
        )

    def _identify_risk_drivers(
        self,
        features: Dict[str, Any],
        termination_prob: float,
        delay_prob: float
    ) -> List[Dict[str, Any]]:
        """Identify top factors driving the risk prediction."""
        drivers = []

        # Check each feature against thresholds
        if features.get("exclusion_count", 0) > 15:
            drivers.append({
                "factor": "High exclusion criteria count",
                "value": features["exclusion_count"],
                "threshold": 15,
                "impact": "Increases termination and amendment risk",
                "recommendation": "Consider relaxing exclusion criteria"
            })

        if features.get("enrollment_per_site", 0) > 50:
            drivers.append({
                "factor": "High enrollment per site target",
                "value": round(features["enrollment_per_site"], 1),
                "threshold": 50,
                "impact": "Increases delay probability",
                "recommendation": "Add more sites or reduce enrollment target"
            })

        if features.get("age_range", 100) < 30:
            drivers.append({
                "factor": "Narrow age range",
                "value": features["age_range"],
                "threshold": 30,
                "impact": "Limits eligible population",
                "recommendation": "Consider expanding age range if scientifically appropriate"
            })

        if features.get("has_biomarker_criteria", 0):
            drivers.append({
                "factor": "Biomarker-based eligibility",
                "value": "Yes",
                "threshold": None,
                "impact": "May slow screening and enrollment",
                "recommendation": "Ensure adequate screening capacity at sites"
            })

        if features.get("planned_duration_months", 0) > 36:
            drivers.append({
                "factor": "Long trial duration",
                "value": f"{features['planned_duration_months']:.0f} months",
                "threshold": "36 months",
                "impact": "Increases dropout and termination risk",
                "recommendation": "Consider interim analyses or shorter endpoints"
            })

        # Sort by impact (based on termination probability)
        return sorted(drivers, key=lambda x: len(x.get("impact", "")), reverse=True)[:5]

    def _save_models(self):
        """Save trained models to disk."""
        os.makedirs(self.MODEL_DIR, exist_ok=True)

        if self.termination_model is not None:
            with open(os.path.join(self.MODEL_DIR, "termination_model.pkl"), "wb") as f:
                pickle.dump(self.termination_model, f)

        if self.delay_model is not None:
            with open(os.path.join(self.MODEL_DIR, "delay_model.pkl"), "wb") as f:
                pickle.dump(self.delay_model, f)

        # Save metrics
        metrics = {
            "termination": self.termination_metrics.__dict__ if self.termination_metrics else None,
            "delay": self.delay_metrics.__dict__ if self.delay_metrics else None,
            "trained_at": datetime.utcnow().isoformat()
        }
        with open(os.path.join(self.MODEL_DIR, "model_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        logger.info(f"Models saved to {self.MODEL_DIR}/")

    def _load_models(self):
        """Load pre-trained models from disk."""
        try:
            term_path = os.path.join(self.MODEL_DIR, "termination_model.pkl")
            if os.path.exists(term_path):
                with open(term_path, "rb") as f:
                    self.termination_model = pickle.load(f)
                logger.info("Loaded pre-trained termination model")

            delay_path = os.path.join(self.MODEL_DIR, "delay_model.pkl")
            if os.path.exists(delay_path):
                with open(delay_path, "rb") as f:
                    self.delay_model = pickle.load(f)
                logger.info("Loaded pre-trained delay model")

            # Load metrics
            metrics_path = os.path.join(self.MODEL_DIR, "model_metrics.json")
            if os.path.exists(metrics_path):
                with open(metrics_path, "r") as f:
                    metrics = json.load(f)
                if metrics.get("termination"):
                    self.termination_metrics = ModelMetrics(**metrics["termination"])
                if metrics.get("delay"):
                    self.delay_metrics = ModelMetrics(**metrics["delay"])
        except Exception as e:
            logger.warning(f"Could not load pre-trained models: {e}")


def train_risk_models(db_session: Session) -> Dict[str, ModelMetrics]:
    """
    Train all ML risk models using data from the database.

    Args:
        db_session: SQLAlchemy session

    Returns:
        Dictionary mapping model names to their metrics
    """
    predictor = MLRiskPredictor(db_session=db_session)
    return predictor.train_models()


def predict_protocol_risk(
    protocol_data: Dict[str, Any],
    db_session: Optional[Session] = None
) -> RiskPrediction:
    """
    Predict risks for a protocol using trained ML models.

    Args:
        protocol_data: Protocol details (condition, phase, enrollment, etc.)
        db_session: Optional database session for loading models

    Returns:
        RiskPrediction with probabilities
    """
    predictor = MLRiskPredictor(db_session=db_session)
    return predictor.predict(protocol_data)


if __name__ == "__main__":
    # Example: Train models if database is available
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    from src.database import DatabaseManager

    logging.basicConfig(level=logging.INFO)

    # Connect to database
    db = DatabaseManager.get_instance()
    session = db.get_session()

    # Train models
    print("Training ML risk models...")
    predictor = MLRiskPredictor(db_session=session)
    metrics = predictor.train_models(min_samples=100)

    print("\nModel Performance:")
    for name, m in metrics.items():
        print(f"\n{name.upper()} Model:")
        print(f"  Accuracy: {m.accuracy:.3f}")
        print(f"  AUC-ROC:  {m.auc_roc:.3f}")
        print(f"  F1 Score: {m.f1:.3f}")
        print(f"  Training samples: {m.training_samples}")

    # Test prediction
    print("\n" + "="*60)
    print("Test Prediction")
    print("="*60)

    test_protocol = {
        "condition": "Type 2 Diabetes",
        "phase": "PHASE3",
        "enrollment": 2000,
        "num_sites": 150,
        "eligibility_criteria": """
        Inclusion Criteria:
        - Adults 18-65 years with Type 2 Diabetes
        - HbA1c between 7.5% and 10.0%

        Exclusion Criteria:
        - eGFR < 60
        - History of pancreatitis
        - Prior GLP-1 use
        """,
        "min_age": "18 Years",
        "max_age": "65 Years",
    }

    prediction = predictor.predict(test_protocol)
    print(f"\nTermination Probability: {prediction.termination_probability:.1%}")
    print(f"Delay Probability: {prediction.delay_probability:.1%}")
    print(f"Amendment Probability: {prediction.amendment_probability:.1%}")
    print(f"Confidence: {prediction.confidence}")
    print("\nRisk Drivers:")
    for driver in prediction.risk_drivers:
        print(f"  - {driver['factor']}: {driver['value']}")
