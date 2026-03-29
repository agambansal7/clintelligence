"""
Medical Ontology and Synonym Mapping

Maps medical terms to standardized forms and provides synonyms
for better trial matching.
"""

import re
from typing import List, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class MedicalConcept:
    """A standardized medical concept with synonyms."""
    canonical_name: str
    synonyms: Set[str]
    mesh_id: Optional[str] = None
    icd10_codes: List[str] = None
    therapeutic_area: str = ""

    def matches(self, text: str) -> bool:
        """Check if text matches this concept."""
        text_lower = text.lower()
        if self.canonical_name.lower() in text_lower:
            return True
        return any(syn.lower() in text_lower for syn in self.synonyms)

    def get_all_terms(self) -> Set[str]:
        """Get all terms including canonical name and synonyms."""
        terms = {self.canonical_name.lower()}
        terms.update(s.lower() for s in self.synonyms)
        return terms


class MedicalOntology:
    """
    Medical ontology for condition and intervention normalization.
    """

    def __init__(self):
        self.conditions = self._build_condition_ontology()
        self.interventions = self._build_intervention_ontology()
        self.therapeutic_areas = self._build_therapeutic_areas()
        self.endpoint_types = self._build_endpoint_types()

    def _build_condition_ontology(self) -> Dict[str, MedicalConcept]:
        """Build condition synonym mappings."""
        conditions = {}

        # Diabetes
        conditions["type_2_diabetes"] = MedicalConcept(
            canonical_name="Type 2 Diabetes Mellitus",
            synonyms={"T2DM", "T2D", "type 2 diabetes", "type II diabetes", "NIDDM",
                     "non-insulin dependent diabetes", "adult-onset diabetes",
                     "diabetes mellitus type 2", "DM2", "maturity onset diabetes"},
            mesh_id="D003924",
            icd10_codes=["E11"],
            therapeutic_area="endocrinology"
        )

        conditions["type_1_diabetes"] = MedicalConcept(
            canonical_name="Type 1 Diabetes Mellitus",
            synonyms={"T1DM", "T1D", "type 1 diabetes", "type I diabetes", "IDDM",
                     "insulin dependent diabetes", "juvenile diabetes", "DM1"},
            mesh_id="D003922",
            icd10_codes=["E10"],
            therapeutic_area="endocrinology"
        )

        # Cardiovascular
        conditions["heart_failure"] = MedicalConcept(
            canonical_name="Heart Failure",
            synonyms={"CHF", "congestive heart failure", "cardiac failure", "HF",
                     "HFrEF", "HFpEF", "systolic heart failure", "diastolic heart failure",
                     "left ventricular failure", "right heart failure"},
            mesh_id="D006333",
            icd10_codes=["I50"],
            therapeutic_area="cardiology"
        )

        conditions["atrial_fibrillation"] = MedicalConcept(
            canonical_name="Atrial Fibrillation",
            synonyms={"AFib", "AF", "A-fib", "auricular fibrillation", "atrial fib"},
            mesh_id="D001281",
            icd10_codes=["I48"],
            therapeutic_area="cardiology"
        )

        conditions["coronary_artery_disease"] = MedicalConcept(
            canonical_name="Coronary Artery Disease",
            synonyms={"CAD", "coronary heart disease", "CHD", "ischemic heart disease",
                     "IHD", "coronary atherosclerosis", "ASCVD"},
            mesh_id="D003324",
            icd10_codes=["I25"],
            therapeutic_area="cardiology"
        )

        conditions["myocardial_infarction"] = MedicalConcept(
            canonical_name="Myocardial Infarction",
            synonyms={"MI", "heart attack", "STEMI", "NSTEMI", "acute MI", "AMI",
                     "acute coronary syndrome", "ACS"},
            mesh_id="D009203",
            icd10_codes=["I21", "I22"],
            therapeutic_area="cardiology"
        )

        conditions["hypertension"] = MedicalConcept(
            canonical_name="Hypertension",
            synonyms={"HTN", "high blood pressure", "elevated blood pressure",
                     "essential hypertension", "primary hypertension"},
            mesh_id="D006973",
            icd10_codes=["I10"],
            therapeutic_area="cardiology"
        )

        conditions["aortic_stenosis"] = MedicalConcept(
            canonical_name="Aortic Stenosis",
            synonyms={"AS", "aortic valve stenosis", "AVS", "calcific aortic stenosis",
                     "severe aortic stenosis", "aortic valve disease"},
            mesh_id="D001024",
            icd10_codes=["I35.0"],
            therapeutic_area="cardiology"
        )

        # Oncology
        conditions["nsclc"] = MedicalConcept(
            canonical_name="Non-Small Cell Lung Cancer",
            synonyms={"NSCLC", "non-small cell lung carcinoma", "lung adenocarcinoma",
                     "squamous cell lung cancer", "large cell lung cancer",
                     "stage IV NSCLC", "advanced NSCLC", "metastatic NSCLC"},
            mesh_id="D002289",
            icd10_codes=["C34"],
            therapeutic_area="oncology"
        )

        conditions["breast_cancer"] = MedicalConcept(
            canonical_name="Breast Cancer",
            synonyms={"breast carcinoma", "breast neoplasm", "HER2+ breast cancer",
                     "triple negative breast cancer", "TNBC", "ER+ breast cancer",
                     "metastatic breast cancer", "mBC", "early breast cancer"},
            mesh_id="D001943",
            icd10_codes=["C50"],
            therapeutic_area="oncology"
        )

        conditions["colorectal_cancer"] = MedicalConcept(
            canonical_name="Colorectal Cancer",
            synonyms={"CRC", "colon cancer", "rectal cancer", "bowel cancer",
                     "colorectal carcinoma", "metastatic CRC", "mCRC"},
            mesh_id="D015179",
            icd10_codes=["C18", "C19", "C20"],
            therapeutic_area="oncology"
        )

        conditions["melanoma"] = MedicalConcept(
            canonical_name="Melanoma",
            synonyms={"malignant melanoma", "cutaneous melanoma", "skin melanoma",
                     "metastatic melanoma", "advanced melanoma", "uveal melanoma"},
            mesh_id="D008545",
            icd10_codes=["C43"],
            therapeutic_area="oncology"
        )

        # Neurology
        conditions["alzheimers"] = MedicalConcept(
            canonical_name="Alzheimer's Disease",
            synonyms={"AD", "Alzheimer disease", "Alzheimers", "senile dementia",
                     "early onset Alzheimer", "mild cognitive impairment", "MCI"},
            mesh_id="D000544",
            icd10_codes=["G30"],
            therapeutic_area="neurology"
        )

        conditions["parkinsons"] = MedicalConcept(
            canonical_name="Parkinson's Disease",
            synonyms={"PD", "Parkinson disease", "Parkinsons", "parkinsonism",
                     "idiopathic Parkinson"},
            mesh_id="D010300",
            icd10_codes=["G20"],
            therapeutic_area="neurology"
        )

        conditions["multiple_sclerosis"] = MedicalConcept(
            canonical_name="Multiple Sclerosis",
            synonyms={"MS", "relapsing remitting MS", "RRMS", "progressive MS",
                     "PPMS", "SPMS", "secondary progressive MS"},
            mesh_id="D009103",
            icd10_codes=["G35"],
            therapeutic_area="neurology"
        )

        # Immunology/Rheumatology
        conditions["rheumatoid_arthritis"] = MedicalConcept(
            canonical_name="Rheumatoid Arthritis",
            synonyms={"RA", "rheumatoid", "inflammatory arthritis", "autoimmune arthritis"},
            mesh_id="D001172",
            icd10_codes=["M05", "M06"],
            therapeutic_area="rheumatology"
        )

        conditions["psoriasis"] = MedicalConcept(
            canonical_name="Psoriasis",
            synonyms={"plaque psoriasis", "psoriatic disease", "moderate to severe psoriasis"},
            mesh_id="D011565",
            icd10_codes=["L40"],
            therapeutic_area="dermatology"
        )

        conditions["crohns"] = MedicalConcept(
            canonical_name="Crohn's Disease",
            synonyms={"Crohn disease", "Crohns", "regional enteritis", "inflammatory bowel disease",
                     "IBD", "CD"},
            mesh_id="D003424",
            icd10_codes=["K50"],
            therapeutic_area="gastroenterology"
        )

        conditions["ulcerative_colitis"] = MedicalConcept(
            canonical_name="Ulcerative Colitis",
            synonyms={"UC", "colitis ulcerosa", "inflammatory bowel disease", "IBD"},
            mesh_id="D003093",
            icd10_codes=["K51"],
            therapeutic_area="gastroenterology"
        )

        # Respiratory
        conditions["asthma"] = MedicalConcept(
            canonical_name="Asthma",
            synonyms={"bronchial asthma", "allergic asthma", "severe asthma",
                     "uncontrolled asthma", "eosinophilic asthma"},
            mesh_id="D001249",
            icd10_codes=["J45"],
            therapeutic_area="pulmonology"
        )

        conditions["copd"] = MedicalConcept(
            canonical_name="Chronic Obstructive Pulmonary Disease",
            synonyms={"COPD", "chronic bronchitis", "emphysema", "chronic airway obstruction",
                     "chronic obstructive lung disease", "COLD"},
            mesh_id="D029424",
            icd10_codes=["J44"],
            therapeutic_area="pulmonology"
        )

        # Infectious Disease
        conditions["hepatitis_c"] = MedicalConcept(
            canonical_name="Hepatitis C",
            synonyms={"HCV", "hep C", "chronic hepatitis C", "hepatitis C virus infection"},
            mesh_id="D006526",
            icd10_codes=["B17.1", "B18.2"],
            therapeutic_area="infectious_disease"
        )

        conditions["hiv"] = MedicalConcept(
            canonical_name="HIV Infection",
            synonyms={"HIV", "HIV/AIDS", "AIDS", "human immunodeficiency virus",
                     "HIV-1", "HIV positive"},
            mesh_id="D015658",
            icd10_codes=["B20", "B24"],
            therapeutic_area="infectious_disease"
        )

        # Obesity/Metabolic
        conditions["obesity"] = MedicalConcept(
            canonical_name="Obesity",
            synonyms={"morbid obesity", "severe obesity", "class III obesity",
                     "overweight", "BMI >30", "adiposity"},
            mesh_id="D009765",
            icd10_codes=["E66"],
            therapeutic_area="endocrinology"
        )

        conditions["nash"] = MedicalConcept(
            canonical_name="Non-Alcoholic Steatohepatitis",
            synonyms={"NASH", "NAFLD", "fatty liver disease", "non-alcoholic fatty liver",
                     "metabolic associated fatty liver", "MAFLD"},
            mesh_id="D065626",
            icd10_codes=["K75.81"],
            therapeutic_area="hepatology"
        )

        return conditions

    def _build_intervention_ontology(self) -> Dict[str, MedicalConcept]:
        """Build intervention/drug class mappings."""
        interventions = {}

        # GLP-1 Receptor Agonists
        interventions["glp1_agonist"] = MedicalConcept(
            canonical_name="GLP-1 Receptor Agonist",
            synonyms={"GLP-1 RA", "GLP1 agonist", "incretin mimetic", "glucagon-like peptide-1",
                     "semaglutide", "liraglutide", "dulaglutide", "exenatide", "tirzepatide",
                     "Ozempic", "Wegovy", "Victoza", "Trulicity", "Byetta", "Mounjaro"},
            therapeutic_area="endocrinology"
        )

        # SGLT2 Inhibitors
        interventions["sglt2_inhibitor"] = MedicalConcept(
            canonical_name="SGLT2 Inhibitor",
            synonyms={"SGLT2i", "sodium glucose co-transporter 2", "gliflozin",
                     "empagliflozin", "dapagliflozin", "canagliflozin",
                     "Jardiance", "Farxiga", "Invokana"},
            therapeutic_area="endocrinology"
        )

        # DPP-4 Inhibitors
        interventions["dpp4_inhibitor"] = MedicalConcept(
            canonical_name="DPP-4 Inhibitor",
            synonyms={"DPP4i", "dipeptidyl peptidase-4", "gliptin",
                     "sitagliptin", "saxagliptin", "linagliptin",
                     "Januvia", "Onglyza", "Tradjenta"},
            therapeutic_area="endocrinology"
        )

        # PD-1/PD-L1 Inhibitors
        interventions["pd1_inhibitor"] = MedicalConcept(
            canonical_name="PD-1/PD-L1 Inhibitor",
            synonyms={"PD-1 inhibitor", "PD-L1 inhibitor", "checkpoint inhibitor",
                     "immune checkpoint inhibitor", "ICI", "anti-PD-1", "anti-PD-L1",
                     "pembrolizumab", "nivolumab", "atezolizumab", "durvalumab",
                     "Keytruda", "Opdivo", "Tecentriq", "Imfinzi"},
            therapeutic_area="oncology"
        )

        # CTLA-4 Inhibitors
        interventions["ctla4_inhibitor"] = MedicalConcept(
            canonical_name="CTLA-4 Inhibitor",
            synonyms={"CTLA4 inhibitor", "anti-CTLA-4", "ipilimumab", "Yervoy"},
            therapeutic_area="oncology"
        )

        # TNF Inhibitors
        interventions["tnf_inhibitor"] = MedicalConcept(
            canonical_name="TNF Inhibitor",
            synonyms={"TNF blocker", "anti-TNF", "TNF alpha inhibitor",
                     "adalimumab", "infliximab", "etanercept", "certolizumab", "golimumab",
                     "Humira", "Remicade", "Enbrel", "Cimzia", "Simponi"},
            therapeutic_area="immunology"
        )

        # JAK Inhibitors
        interventions["jak_inhibitor"] = MedicalConcept(
            canonical_name="JAK Inhibitor",
            synonyms={"JAKi", "Janus kinase inhibitor",
                     "tofacitinib", "baricitinib", "upadacitinib", "ruxolitinib",
                     "Xeljanz", "Olumiant", "Rinvoq", "Jakafi"},
            therapeutic_area="immunology"
        )

        # IL-17 Inhibitors
        interventions["il17_inhibitor"] = MedicalConcept(
            canonical_name="IL-17 Inhibitor",
            synonyms={"IL-17 blocker", "anti-IL-17",
                     "secukinumab", "ixekizumab", "brodalumab",
                     "Cosentyx", "Taltz", "Siliq"},
            therapeutic_area="immunology"
        )

        # PCSK9 Inhibitors
        interventions["pcsk9_inhibitor"] = MedicalConcept(
            canonical_name="PCSK9 Inhibitor",
            synonyms={"PCSK9i", "anti-PCSK9",
                     "evolocumab", "alirocumab", "inclisiran",
                     "Repatha", "Praluent", "Leqvio"},
            therapeutic_area="cardiology"
        )

        # CAR-T
        interventions["car_t"] = MedicalConcept(
            canonical_name="CAR-T Cell Therapy",
            synonyms={"CAR-T", "chimeric antigen receptor", "CAR T-cell",
                     "axicabtagene ciloleucel", "tisagenlecleucel",
                     "Yescarta", "Kymriah"},
            therapeutic_area="oncology"
        )

        # Bispecific Antibodies
        interventions["bispecific"] = MedicalConcept(
            canonical_name="Bispecific Antibody",
            synonyms={"bispecific", "BiTE", "bispecific T-cell engager",
                     "blinatumomab", "Blincyto"},
            therapeutic_area="oncology"
        )

        # ADC
        interventions["adc"] = MedicalConcept(
            canonical_name="Antibody-Drug Conjugate",
            synonyms={"ADC", "antibody drug conjugate",
                     "trastuzumab deruxtecan", "enfortumab vedotin",
                     "Enhertu", "Padcev"},
            therapeutic_area="oncology"
        )

        return interventions

    def _build_therapeutic_areas(self) -> Dict[str, Set[str]]:
        """Map therapeutic area synonyms."""
        return {
            "oncology": {"cancer", "tumor", "neoplasm", "malignancy", "carcinoma", "oncologic"},
            "cardiology": {"cardiovascular", "cardiac", "heart", "coronary", "vascular"},
            "endocrinology": {"diabetes", "metabolic", "thyroid", "hormonal", "endocrine"},
            "neurology": {"neurological", "brain", "CNS", "nervous system", "neuro"},
            "immunology": {"autoimmune", "immune", "inflammatory", "rheumatology"},
            "pulmonology": {"respiratory", "lung", "pulmonary", "airway"},
            "gastroenterology": {"GI", "gastrointestinal", "digestive", "hepatic", "liver"},
            "infectious_disease": {"infection", "viral", "bacterial", "infectious"},
            "dermatology": {"skin", "dermal", "cutaneous"},
            "hematology": {"blood", "hematologic", "coagulation"},
            "nephrology": {"kidney", "renal", "nephro"},
            "ophthalmology": {"eye", "ocular", "ophthalmic", "retinal"},
        }

    def _build_endpoint_types(self) -> Dict[str, Set[str]]:
        """Map endpoint type keywords."""
        return {
            "efficacy": {"response", "remission", "reduction", "improvement", "change from baseline"},
            "survival": {"overall survival", "OS", "progression-free survival", "PFS", "DFS", "mortality", "death"},
            "biomarker": {"HbA1c", "LDL", "blood pressure", "tumor marker", "PSA", "CRP", "eGFR"},
            "patient_reported": {"quality of life", "QoL", "PRO", "patient reported", "symptom", "pain score"},
            "safety": {"adverse event", "AE", "safety", "tolerability", "side effect"},
            "pharmacokinetic": {"PK", "pharmacokinetic", "Cmax", "AUC", "half-life", "clearance"},
        }

    def normalize_condition(self, text: str) -> tuple[str, List[str]]:
        """
        Normalize a condition to canonical form and get all search terms.

        Returns:
            Tuple of (canonical_name, list_of_all_search_terms)
        """
        text_lower = text.lower()

        for key, concept in self.conditions.items():
            if concept.matches(text):
                return concept.canonical_name, list(concept.get_all_terms())

        # No match found - return original with basic variations
        terms = [text_lower]
        # Add individual words
        for word in text_lower.split():
            if len(word) >= 4:
                terms.append(word)

        return text, terms

    def normalize_intervention(self, text: str) -> tuple[str, List[str]]:
        """Normalize an intervention to canonical form."""
        text_lower = text.lower()

        for key, concept in self.interventions.items():
            if concept.matches(text):
                return concept.canonical_name, list(concept.get_all_terms())

        return text, [text_lower]

    def get_therapeutic_area(self, condition: str) -> str:
        """Get therapeutic area for a condition."""
        condition_lower = condition.lower()

        for key, concept in self.conditions.items():
            if concept.matches(condition):
                return concept.therapeutic_area

        # Try to infer from keywords
        for area, keywords in self.therapeutic_areas.items():
            if any(kw in condition_lower for kw in keywords):
                return area

        return "general"

    def expand_search_terms(self, text: str) -> List[str]:
        """Expand a search term into all related terms."""
        terms = set()
        text_lower = text.lower()
        terms.add(text_lower)

        # Check conditions
        for key, concept in self.conditions.items():
            if concept.matches(text):
                terms.update(concept.get_all_terms())

        # Check interventions
        for key, concept in self.interventions.items():
            if concept.matches(text):
                terms.update(concept.get_all_terms())

        # Add therapeutic area terms
        for area, keywords in self.therapeutic_areas.items():
            if any(kw in text_lower for kw in keywords):
                terms.update(keywords)

        return list(terms)


# Singleton instance
_ontology = None

def get_ontology() -> MedicalOntology:
    """Get singleton ontology instance."""
    global _ontology
    if _ontology is None:
        _ontology = MedicalOntology()
    return _ontology
