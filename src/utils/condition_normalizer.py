"""
Condition Normalizer for TrialIntel.

Provides centralized condition/therapeutic area normalization for consistent
querying across the database. Maps various condition names to standardized
categories for better matching.
"""

from typing import List, Set, Optional
import re


# Mapping of normalized categories to their synonyms/variants
CONDITION_MAPPINGS = {
    # Oncology
    "breast cancer": ["breast cancer", "breast neoplasm", "breast carcinoma", "breast tumor", "mammary cancer"],
    "lung cancer": ["lung cancer", "lung neoplasm", "lung carcinoma", "nsclc", "sclc", "non-small cell lung", "small cell lung", "pulmonary cancer"],
    "prostate cancer": ["prostate cancer", "prostate neoplasm", "prostate carcinoma", "prostatic cancer"],
    "colorectal cancer": ["colorectal cancer", "colon cancer", "rectal cancer", "bowel cancer", "colorectal neoplasm", "colorectal carcinoma"],
    "melanoma": ["melanoma", "skin cancer", "cutaneous melanoma", "malignant melanoma"],
    "ovarian cancer": ["ovarian cancer", "ovarian neoplasm", "ovarian carcinoma", "ovary cancer"],
    "pancreatic cancer": ["pancreatic cancer", "pancreas cancer", "pancreatic neoplasm", "pancreatic carcinoma"],
    "liver cancer": ["liver cancer", "hepatocellular carcinoma", "hcc", "hepatic cancer", "liver neoplasm"],
    "gastric cancer": ["gastric cancer", "stomach cancer", "gastric carcinoma", "gastric neoplasm"],
    "bladder cancer": ["bladder cancer", "bladder neoplasm", "bladder carcinoma", "urothelial cancer"],
    "kidney cancer": ["kidney cancer", "renal cancer", "renal cell carcinoma", "rcc", "kidney neoplasm"],
    "head and neck cancer": ["head and neck cancer", "head neck cancer", "oral cancer", "throat cancer", "laryngeal cancer", "pharyngeal cancer"],
    "brain tumor": ["brain tumor", "brain cancer", "brain neoplasm", "cns tumor", "central nervous system tumor"],
    "glioblastoma": ["glioblastoma", "gbm", "glioblastoma multiforme", "glioma"],
    "lymphoma": ["lymphoma", "hodgkin", "non-hodgkin", "nhl", "b-cell lymphoma", "t-cell lymphoma"],
    "leukemia": ["leukemia", "leukaemia", "aml", "all", "cml", "cll", "acute leukemia", "chronic leukemia"],
    "multiple myeloma": ["multiple myeloma", "myeloma", "plasma cell myeloma"],
    "sarcoma": ["sarcoma", "soft tissue sarcoma", "bone sarcoma", "osteosarcoma"],

    # Cardiovascular
    "heart failure": ["heart failure", "cardiac failure", "chf", "congestive heart failure", "hfref", "hfpef"],
    "hypertension": ["hypertension", "high blood pressure", "htn", "elevated blood pressure"],
    "atrial fibrillation": ["atrial fibrillation", "afib", "af", "a-fib", "auricular fibrillation"],
    "coronary artery disease": ["coronary artery disease", "cad", "coronary heart disease", "chd", "ischemic heart disease", "ihd"],
    "myocardial infarction": ["myocardial infarction", "heart attack", "mi", "stemi", "nstemi", "acute coronary syndrome", "acs"],
    "stroke": ["stroke", "cerebrovascular accident", "cva", "ischemic stroke", "hemorrhagic stroke", "cerebral infarction"],
    "peripheral artery disease": ["peripheral artery disease", "pad", "peripheral vascular disease", "pvd", "claudication"],
    "pulmonary hypertension": ["pulmonary hypertension", "pah", "pulmonary arterial hypertension", "ph"],
    "cardiomyopathy": ["cardiomyopathy", "dilated cardiomyopathy", "hypertrophic cardiomyopathy", "dcm", "hcm"],

    # Metabolic/Endocrine
    "diabetes": ["diabetes", "diabetes mellitus", "type 2 diabetes", "type 1 diabetes", "t2dm", "t1dm", "dm", "diabetic"],
    "type 2 diabetes": ["type 2 diabetes", "t2dm", "type ii diabetes", "non-insulin dependent diabetes", "niddm", "adult onset diabetes"],
    "type 1 diabetes": ["type 1 diabetes", "t1dm", "type i diabetes", "insulin dependent diabetes", "iddm", "juvenile diabetes"],
    "obesity": ["obesity", "obese", "overweight", "morbid obesity", "severe obesity"],
    "dyslipidemia": ["dyslipidemia", "hyperlipidemia", "hypercholesterolemia", "high cholesterol", "lipid disorder"],
    "metabolic syndrome": ["metabolic syndrome", "syndrome x", "insulin resistance syndrome"],
    "thyroid disease": ["thyroid disease", "hypothyroidism", "hyperthyroidism", "thyroid disorder", "thyroid"],
    "osteoporosis": ["osteoporosis", "bone loss", "osteopenia", "low bone density"],

    # Neurology
    "alzheimer": ["alzheimer", "alzheimer's disease", "alzheimers", "ad", "alzheimer disease"],
    "parkinson": ["parkinson", "parkinson's disease", "parkinsons", "pd", "parkinson disease"],
    "multiple sclerosis": ["multiple sclerosis", "ms", "relapsing remitting ms", "rrms", "progressive ms"],
    "epilepsy": ["epilepsy", "seizure disorder", "seizures", "convulsions"],
    "migraine": ["migraine", "headache", "chronic migraine", "episodic migraine"],
    "amyotrophic lateral sclerosis": ["amyotrophic lateral sclerosis", "als", "lou gehrig", "motor neuron disease", "mnd"],
    "huntington disease": ["huntington disease", "huntington's disease", "huntingtons", "hd"],
    "neuropathy": ["neuropathy", "peripheral neuropathy", "diabetic neuropathy", "nerve damage"],
    "spinal muscular atrophy": ["spinal muscular atrophy", "sma"],

    # Psychiatry
    "depression": ["depression", "major depression", "major depressive disorder", "mdd", "depressive disorder", "clinical depression"],
    "major depressive disorder": ["major depressive disorder", "mdd", "major depression", "clinical depression"],
    "anxiety": ["anxiety", "anxiety disorder", "generalized anxiety", "gad", "panic disorder", "social anxiety"],
    "schizophrenia": ["schizophrenia", "schizoaffective", "psychosis", "psychotic disorder"],
    "bipolar disorder": ["bipolar disorder", "bipolar", "manic depression", "bipolar i", "bipolar ii"],
    "ptsd": ["ptsd", "post traumatic stress", "post-traumatic stress disorder", "trauma"],
    "adhd": ["adhd", "attention deficit", "add", "attention deficit hyperactivity disorder"],
    "autism": ["autism", "autism spectrum", "asd", "autistic", "asperger"],
    "substance use disorder": ["substance use disorder", "addiction", "substance abuse", "drug abuse", "alcohol use disorder"],

    # Immunology/Rheumatology
    "rheumatoid arthritis": ["rheumatoid arthritis", "ra", "rheumatoid", "inflammatory arthritis"],
    "psoriasis": ["psoriasis", "plaque psoriasis", "psoriatic", "skin psoriasis"],
    "psoriatic arthritis": ["psoriatic arthritis", "psa"],
    "lupus": ["lupus", "systemic lupus erythematosus", "sle", "lupus erythematosus"],
    "inflammatory bowel disease": ["inflammatory bowel disease", "ibd"],
    "crohn disease": ["crohn disease", "crohn's disease", "crohns", "regional enteritis"],
    "ulcerative colitis": ["ulcerative colitis", "uc", "colitis"],
    "ankylosing spondylitis": ["ankylosing spondylitis", "as", "axial spondyloarthritis"],
    "sjogren syndrome": ["sjogren syndrome", "sjogren's syndrome", "sjogrens", "sicca syndrome"],
    "dermatomyositis": ["dermatomyositis", "polymyositis", "inflammatory myopathy"],

    # Respiratory
    "asthma": ["asthma", "bronchial asthma", "allergic asthma", "severe asthma"],
    "copd": ["copd", "chronic obstructive pulmonary disease", "chronic bronchitis", "emphysema"],
    "pulmonary fibrosis": ["pulmonary fibrosis", "ipf", "idiopathic pulmonary fibrosis", "lung fibrosis"],
    "cystic fibrosis": ["cystic fibrosis", "cf", "mucoviscidosis"],
    "pneumonia": ["pneumonia", "lung infection", "respiratory infection"],
    "acute respiratory distress syndrome": ["acute respiratory distress syndrome", "ards", "respiratory distress"],

    # Infectious Disease
    "hiv": ["hiv", "human immunodeficiency virus", "aids", "hiv/aids", "hiv-1", "hiv infection"],
    "hepatitis b": ["hepatitis b", "hbv", "hep b", "chronic hepatitis b"],
    "hepatitis c": ["hepatitis c", "hcv", "hep c", "chronic hepatitis c"],
    "influenza": ["influenza", "flu", "seasonal flu", "influenza virus"],
    "covid-19": ["covid-19", "covid", "coronavirus", "sars-cov-2", "covid19"],
    "tuberculosis": ["tuberculosis", "tb", "mycobacterium tuberculosis"],
    "sepsis": ["sepsis", "septic shock", "severe sepsis", "bloodstream infection"],
    "bacterial infection": ["bacterial infection", "bacteremia", "bacterial"],

    # Nephrology
    "chronic kidney disease": ["chronic kidney disease", "ckd", "chronic renal disease", "renal insufficiency", "kidney disease"],
    "diabetic nephropathy": ["diabetic nephropathy", "diabetic kidney disease", "dkd"],
    "glomerulonephritis": ["glomerulonephritis", "gn", "nephritis"],
    "polycystic kidney disease": ["polycystic kidney disease", "pkd", "adpkd"],

    # Gastroenterology/Hepatology
    "non-alcoholic fatty liver disease": ["non-alcoholic fatty liver disease", "nafld", "nash", "fatty liver", "non-alcoholic steatohepatitis"],
    "cirrhosis": ["cirrhosis", "liver cirrhosis", "hepatic cirrhosis"],
    "gastroparesis": ["gastroparesis", "delayed gastric emptying"],
    "irritable bowel syndrome": ["irritable bowel syndrome", "ibs"],

    # Hematology
    "anemia": ["anemia", "anaemia", "low hemoglobin", "iron deficiency anemia"],
    "sickle cell disease": ["sickle cell disease", "sickle cell anemia", "scd"],
    "hemophilia": ["hemophilia", "haemophilia", "bleeding disorder"],
    "thrombocytopenia": ["thrombocytopenia", "low platelets", "itp", "immune thrombocytopenia"],
    "myelodysplastic syndrome": ["myelodysplastic syndrome", "mds", "myelodysplasia"],

    # Ophthalmology
    "macular degeneration": ["macular degeneration", "amd", "age-related macular degeneration", "wet amd", "dry amd"],
    "glaucoma": ["glaucoma", "open angle glaucoma", "closed angle glaucoma"],
    "diabetic retinopathy": ["diabetic retinopathy", "dr", "diabetic eye disease", "diabetic macular edema", "dme"],
    "dry eye disease": ["dry eye disease", "dry eye", "dry eye syndrome", "keratoconjunctivitis sicca"],

    # Dermatology
    "atopic dermatitis": ["atopic dermatitis", "eczema", "atopic eczema", "ad"],
    "eczema": ["eczema", "dermatitis", "atopic dermatitis"],
    "alopecia": ["alopecia", "hair loss", "alopecia areata", "androgenetic alopecia"],
    "vitiligo": ["vitiligo", "skin depigmentation"],

    # Rare/Genetic Diseases
    "duchenne muscular dystrophy": ["duchenne muscular dystrophy", "dmd", "duchenne", "muscular dystrophy"],
    "fabry disease": ["fabry disease", "fabry", "alpha-galactosidase a deficiency"],
    "gaucher disease": ["gaucher disease", "gaucher", "glucocerebrosidase deficiency"],
    "pompe disease": ["pompe disease", "pompe", "glycogen storage disease type ii"],
    "hemoglobin disorders": ["hemoglobin disorders", "thalassemia", "hemoglobinopathy"],

    # Women's Health
    "endometriosis": ["endometriosis", "endometrial"],
    "uterine fibroids": ["uterine fibroids", "fibroids", "leiomyoma", "myoma"],
    "polycystic ovary syndrome": ["polycystic ovary syndrome", "pcos", "polycystic ovarian syndrome"],
    "menopause": ["menopause", "menopausal", "postmenopausal", "hot flashes"],

    # Pain/Musculoskeletal
    "chronic pain": ["chronic pain", "pain", "neuropathic pain", "chronic pain syndrome"],
    "osteoarthritis": ["osteoarthritis", "oa", "degenerative joint disease", "arthritis"],
    "fibromyalgia": ["fibromyalgia", "fibromyalgia syndrome", "fms"],
    "gout": ["gout", "gouty arthritis", "hyperuricemia"],

    # Transplant/Immunosuppression
    "transplant rejection": ["transplant rejection", "organ rejection", "graft rejection", "transplant"],
    "graft versus host disease": ["graft versus host disease", "gvhd", "graft-versus-host"],
}

# Build reverse lookup for fast searching
_SYNONYM_TO_NORMALIZED = {}
for normalized, synonyms in CONDITION_MAPPINGS.items():
    for synonym in synonyms:
        _SYNONYM_TO_NORMALIZED[synonym.lower()] = normalized


def normalize_condition(condition: str) -> str:
    """
    Normalize a condition string to a standard category.

    Args:
        condition: Raw condition string (e.g., "Type 2 Diabetes Mellitus")

    Returns:
        Normalized condition string (e.g., "diabetes")
    """
    if not condition:
        return ""

    condition_lower = condition.lower().strip()

    # Direct match
    if condition_lower in _SYNONYM_TO_NORMALIZED:
        return _SYNONYM_TO_NORMALIZED[condition_lower]

    # Substring match - find best match
    best_match = None
    best_length = 0

    for synonym, normalized in _SYNONYM_TO_NORMALIZED.items():
        if synonym in condition_lower or condition_lower in synonym:
            # Prefer longer matches (more specific)
            if len(synonym) > best_length:
                best_match = normalized
                best_length = len(synonym)

    if best_match:
        return best_match

    # Fallback: return original lowercased
    return condition_lower


def get_condition_variants(condition: str) -> List[str]:
    """
    Get all variants/synonyms for a condition.

    Args:
        condition: Condition to look up (can be normalized or raw)

    Returns:
        List of all known variants for this condition
    """
    normalized = normalize_condition(condition)

    if normalized in CONDITION_MAPPINGS:
        return CONDITION_MAPPINGS[normalized]

    # Return the condition itself if no mappings found
    return [condition.lower()]


def build_condition_query_pattern(condition: str) -> str:
    """
    Build a SQL LIKE pattern that matches all variants of a condition.

    Args:
        condition: Condition to search for

    Returns:
        Pattern string for use in SQL queries
    """
    variants = get_condition_variants(condition)
    # Return the most specific variant for LIKE matching
    # The query should use multiple OR conditions
    return normalize_condition(condition)


def get_search_terms(condition: str) -> Set[str]:
    """
    Get a set of search terms for a condition (for flexible matching).

    Args:
        condition: Condition to search for

    Returns:
        Set of search terms to use in queries
    """
    variants = get_condition_variants(condition)
    search_terms = set()

    for variant in variants:
        # Add the full variant
        search_terms.add(variant)
        # Add individual words (for partial matching)
        words = variant.split()
        for word in words:
            if len(word) > 3:  # Skip short words
                search_terms.add(word)

    return search_terms


def conditions_match(condition1: str, condition2: str) -> bool:
    """
    Check if two condition strings refer to the same condition.

    Args:
        condition1: First condition string
        condition2: Second condition string

    Returns:
        True if conditions match
    """
    return normalize_condition(condition1) == normalize_condition(condition2)


# High-level categories for grouping
THERAPEUTIC_CATEGORIES = {
    "oncology": [
        "breast cancer", "lung cancer", "prostate cancer", "colorectal cancer",
        "melanoma", "ovarian cancer", "pancreatic cancer", "liver cancer",
        "gastric cancer", "bladder cancer", "kidney cancer", "head and neck cancer",
        "brain tumor", "glioblastoma", "lymphoma", "leukemia", "multiple myeloma", "sarcoma"
    ],
    "cardiovascular": [
        "heart failure", "hypertension", "atrial fibrillation", "coronary artery disease",
        "myocardial infarction", "stroke", "peripheral artery disease",
        "pulmonary hypertension", "cardiomyopathy"
    ],
    "metabolic": [
        "diabetes", "type 2 diabetes", "type 1 diabetes", "obesity",
        "dyslipidemia", "metabolic syndrome", "thyroid disease", "osteoporosis"
    ],
    "neurology": [
        "alzheimer", "parkinson", "multiple sclerosis", "epilepsy", "migraine",
        "amyotrophic lateral sclerosis", "huntington disease", "neuropathy",
        "spinal muscular atrophy"
    ],
    "psychiatry": [
        "depression", "major depressive disorder", "anxiety", "schizophrenia",
        "bipolar disorder", "ptsd", "adhd", "autism", "substance use disorder"
    ],
    "immunology": [
        "rheumatoid arthritis", "psoriasis", "psoriatic arthritis", "lupus",
        "inflammatory bowel disease", "crohn disease", "ulcerative colitis",
        "ankylosing spondylitis", "sjogren syndrome", "dermatomyositis"
    ],
    "respiratory": [
        "asthma", "copd", "pulmonary fibrosis", "cystic fibrosis",
        "pneumonia", "acute respiratory distress syndrome"
    ],
    "infectious": [
        "hiv", "hepatitis b", "hepatitis c", "influenza", "covid-19",
        "tuberculosis", "sepsis", "bacterial infection"
    ],
    "nephrology": [
        "chronic kidney disease", "diabetic nephropathy", "glomerulonephritis",
        "polycystic kidney disease"
    ],
    "hematology": [
        "anemia", "sickle cell disease", "hemophilia", "thrombocytopenia",
        "myelodysplastic syndrome"
    ],
    "rare_disease": [
        "duchenne muscular dystrophy", "fabry disease", "gaucher disease",
        "pompe disease", "hemoglobin disorders"
    ],
}


def get_therapeutic_category(condition: str) -> Optional[str]:
    """
    Get the high-level therapeutic category for a condition.

    Args:
        condition: Condition string

    Returns:
        Category name or None
    """
    normalized = normalize_condition(condition)

    for category, conditions in THERAPEUTIC_CATEGORIES.items():
        if normalized in conditions:
            return category

    return None
