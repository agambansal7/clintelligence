#!/usr/bin/env python3
"""
CASE STUDY GENERATOR - Day 3 Sprint

Find real terminated trials and create compelling case studies showing
how TrialIntel could have predicted (and prevented) their failure.

This is CRUCIAL for sales. Nothing sells like "Look at this $50M trial that
failed - we could have warned them on Day 1."

Usage:
    python day3_case_studies.py --db ./data/trials.db --output ./case_studies
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Dict, Any
import argparse
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class CaseStudyGenerator:
    """Generate compelling case studies from terminated trials."""
    
    # High-value therapeutic areas (expensive to fail)
    HIGH_VALUE_AREAS = ["cancer", "diabetes", "alzheimer", "heart"]
    
    # Interesting failure reasons to highlight
    INTERESTING_FAILURES = [
        "enrollment",
        "recruitment",
        "futility",
        "efficacy",
        "safety",
        "sponsor decision",
        "lack of funding",
    ]
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def find_terminated_trials(self, limit: int = 100) -> List[Dict]:
        """Find interesting terminated trials."""
        cursor = self.conn.cursor()
        
        # Find terminated trials with reasons
        cursor.execute("""
            SELECT * FROM trials 
            WHERE status IN ('TERMINATED', 'WITHDRAWN')
            AND why_stopped IS NOT NULL 
            AND why_stopped != ''
            AND enrollment > 50
            AND phase LIKE '%PHASE3%' OR phase LIKE '%PHASE2%'
            ORDER BY enrollment DESC
            LIMIT ?
        """, (limit,))
        
        trials = [dict(row) for row in cursor.fetchall()]
        return trials
    
    def analyze_trial_failure(self, trial: Dict) -> Dict[str, Any]:
        """Analyze why a trial failed and how we could have predicted it."""
        analysis = {
            "nct_id": trial["nct_id"],
            "title": trial["title"],
            "sponsor": trial["sponsor"],
            "phase": trial["phase"],
            "therapeutic_area": trial["therapeutic_area"],
            "target_enrollment": trial["enrollment"],
            "why_stopped": trial["why_stopped"],
            "status": trial["status"],
        }
        
        # Categorize failure reason
        why_stopped_lower = (trial["why_stopped"] or "").lower()
        
        if any(term in why_stopped_lower for term in ["enroll", "recruit", "accrual"]):
            analysis["failure_category"] = "Enrollment Failure"
            analysis["failure_cost_estimate"] = "$10-50M"
            analysis["preventable"] = True
            analysis["how_we_could_help"] = [
                "Our Site Intelligence would have identified enrollment challenges",
                "Protocol Risk Scorer would have flagged restrictive eligibility criteria",
                "Enrollment velocity prediction would have shown unrealistic targets"
            ]
        elif any(term in why_stopped_lower for term in ["efficacy", "futility", "endpoint"]):
            analysis["failure_category"] = "Efficacy Failure"
            analysis["failure_cost_estimate"] = "$50-200M"
            analysis["preventable"] = "Partially"
            analysis["how_we_could_help"] = [
                "Endpoint Benchmarking would have shown historical success rates",
                "Competitive analysis would have revealed similar failed trials",
                "Risk scoring could have suggested alternative endpoints"
            ]
        elif any(term in why_stopped_lower for term in ["safety", "adverse", "toxicity"]):
            analysis["failure_category"] = "Safety Issue"
            analysis["failure_cost_estimate"] = "$20-100M"
            analysis["preventable"] = False
            analysis["how_we_could_help"] = [
                "Not directly preventable, but faster identification possible",
                "Competitive radar tracks similar safety signals in related trials"
            ]
        elif any(term in why_stopped_lower for term in ["sponsor", "business", "strategic", "funding"]):
            analysis["failure_category"] = "Business Decision"
            analysis["failure_cost_estimate"] = "$5-30M"
            analysis["preventable"] = "Partially"
            analysis["how_we_could_help"] = [
                "Competitive analysis shows market dynamics",
                "Risk scoring helps prioritize portfolio decisions"
            ]
        else:
            analysis["failure_category"] = "Other"
            analysis["failure_cost_estimate"] = "$10-50M"
            analysis["preventable"] = "Unknown"
            analysis["how_we_could_help"] = [
                "Comprehensive risk assessment helps identify multiple risk factors"
            ]
        
        # Analyze eligibility criteria issues
        criteria = trial.get("eligibility_criteria", "")
        if criteria:
            criteria_issues = self._analyze_eligibility_issues(criteria)
            analysis["eligibility_issues"] = criteria_issues
        
        # Analyze site/location issues
        locations = json.loads(trial.get("locations", "[]") or "[]")
        analysis["num_sites"] = len(locations)
        
        if trial["enrollment"] and len(locations) > 0:
            sites_per_100 = (len(locations) / trial["enrollment"]) * 100
            if sites_per_100 < 2:
                analysis["site_issue"] = f"Low site density: {sites_per_100:.1f} sites per 100 patients"
        
        return analysis
    
    def _analyze_eligibility_issues(self, criteria: str) -> List[str]:
        """Identify potential issues in eligibility criteria."""
        issues = []
        criteria_lower = criteria.lower()
        
        # Count exclusion criteria
        exclusion_count = criteria_lower.count("exclusion") + len(
            [line for line in criteria.split('\n') 
             if line.strip().startswith('-') or line.strip().startswith('•')]
        )
        
        if exclusion_count > 20:
            issues.append(f"High number of criteria ({exclusion_count}+) - likely enrollment challenges")
        
        # Check for restrictive patterns
        restrictive_patterns = [
            (r"no prior", "Excludes patients with prior treatment"),
            (r"treatment.?naive", "Requires treatment-naive patients"),
            (r"< \d+ years", "Restrictive age requirement"),
            (r"> \d+ years", "Restrictive age requirement"),
            (r"hba1c.*[<>]", "Specific HbA1c threshold"),
            (r"egfr.*[<>]", "Specific eGFR threshold"),
            (r"within \d+ days", "Strict timing requirements"),
        ]
        
        import re
        for pattern, issue in restrictive_patterns:
            if re.search(pattern, criteria_lower):
                issues.append(issue)
        
        return issues[:5]  # Top 5 issues
    
    def generate_case_study(self, trial: Dict) -> str:
        """Generate a formatted case study document."""
        analysis = self.analyze_trial_failure(trial)
        
        case_study = f"""
================================================================================
CASE STUDY: {analysis['nct_id']}
================================================================================

TRIAL OVERVIEW
--------------
Title: {analysis['title'][:100]}...
Sponsor: {analysis['sponsor']}
Phase: {analysis['phase']}
Therapeutic Area: {analysis['therapeutic_area']}
Target Enrollment: {analysis['target_enrollment']} patients
Number of Sites: {analysis['num_sites']}

WHAT HAPPENED
-------------
Status: {analysis['status']}
Reason: {analysis['why_stopped']}

FAILURE ANALYSIS
----------------
Category: {analysis['failure_category']}
Estimated Cost: {analysis['failure_cost_estimate']}
Was This Preventable? {analysis['preventable']}

ELIGIBILITY CRITERIA ISSUES IDENTIFIED
--------------------------------------
"""
        for issue in analysis.get('eligibility_issues', []):
            case_study += f"• {issue}\n"
        
        if analysis.get('site_issue'):
            case_study += f"\nSITE DENSITY ISSUE\n------------------\n{analysis['site_issue']}\n"
        
        case_study += """
HOW TRIALINTEL COULD HAVE HELPED
--------------------------------
"""
        for point in analysis['how_we_could_help']:
            case_study += f"✓ {point}\n"
        
        case_study += f"""
BOTTOM LINE
-----------
This {analysis['failure_cost_estimate']} failure could have been 
{"PREDICTED AND POTENTIALLY PREVENTED" if analysis['preventable'] == True else "identified earlier"}
with TrialIntel's pre-protocol intelligence.

View on ClinicalTrials.gov: https://clinicaltrials.gov/study/{analysis['nct_id']}

================================================================================
"""
        return case_study, analysis
    
    def generate_all_case_studies(self, output_dir: str, num_studies: int = 10):
        """Generate multiple case studies."""
        os.makedirs(output_dir, exist_ok=True)
        
        terminated_trials = self.find_terminated_trials(limit=50)
        print(f"Found {len(terminated_trials)} terminated trials")
        
        # Score and rank trials by "interestingness"
        scored_trials = []
        for trial in terminated_trials:
            score = 0
            
            # Prefer high enrollment (expensive failures)
            if trial.get("enrollment", 0) > 500:
                score += 3
            elif trial.get("enrollment", 0) > 200:
                score += 2
            
            # Prefer Phase 3 (most expensive)
            if "PHASE3" in str(trial.get("phase", "")):
                score += 3
            
            # Prefer enrollment-related failures (we can help most)
            why_stopped = (trial.get("why_stopped") or "").lower()
            if any(term in why_stopped for term in ["enroll", "recruit"]):
                score += 5
            
            # Prefer big pharma sponsors (credibility)
            sponsor = (trial.get("sponsor") or "").lower()
            if any(p in sponsor for p in ["pfizer", "novartis", "merck", "lilly", "roche"]):
                score += 2
            
            scored_trials.append((score, trial))
        
        # Sort by score
        scored_trials.sort(key=lambda x: x[0], reverse=True)
        
        # Generate top case studies
        all_analyses = []
        combined_output = "TRIALINTEL CASE STUDIES\n" + "="*80 + "\n\n"
        combined_output += f"Generated: {datetime.now().strftime('%Y-%m-%d')}\n"
        combined_output += f"These are real clinical trials that failed. TrialIntel could have helped.\n\n"
        
        for i, (score, trial) in enumerate(scored_trials[:num_studies]):
            print(f"\nGenerating case study {i+1}: {trial['nct_id']}")
            case_study, analysis = self.generate_case_study(trial)
            
            # Save individual case study
            filename = f"{output_dir}/case_study_{i+1}_{trial['nct_id']}.txt"
            with open(filename, 'w') as f:
                f.write(case_study)
            
            combined_output += case_study + "\n\n"
            all_analyses.append(analysis)
        
        # Save combined document
        with open(f"{output_dir}/ALL_CASE_STUDIES.txt", 'w') as f:
            f.write(combined_output)
        
        # Save analyses as JSON
        with open(f"{output_dir}/case_study_analyses.json", 'w') as f:
            json.dump(all_analyses, f, indent=2)
        
        # Generate summary
        self._generate_summary(all_analyses, output_dir)
        
        print(f"\n{'='*60}")
        print(f"Generated {len(all_analyses)} case studies")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}")
    
    def _generate_summary(self, analyses: List[Dict], output_dir: str):
        """Generate executive summary of case studies."""
        total_cost = 0
        preventable_count = 0
        
        for a in analyses:
            # Parse cost estimate (take middle of range)
            cost_str = a.get("failure_cost_estimate", "$10M")
            cost_match = cost_str.replace("$", "").replace("M", "").split("-")
            avg_cost = sum(float(c) for c in cost_match) / len(cost_match)
            total_cost += avg_cost
            
            if a.get("preventable") == True:
                preventable_count += 1
        
        summary = f"""
TRIALINTEL CASE STUDY EXECUTIVE SUMMARY
=======================================

Analysis of {len(analyses)} Failed Clinical Trials

KEY FINDINGS
------------
• Total estimated cost of failures: ${total_cost:.0f}M
• Preventable failures: {preventable_count}/{len(analyses)} ({preventable_count/len(analyses)*100:.0f}%)
• Most common failure: Enrollment challenges

FAILURE CATEGORIES
------------------
"""
        categories = {}
        for a in analyses:
            cat = a.get("failure_category", "Other")
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            summary += f"• {cat}: {count} trials\n"
        
        summary += f"""
VALUE PROPOSITION
-----------------
If TrialIntel prevented just 20% of these failures, that's:
${total_cost * 0.2:.0f}M in savings for these {len(analyses)} trials alone.

With 50,000+ trials in our database, the total preventable waste is in the BILLIONS.
"""
        
        with open(f"{output_dir}/EXECUTIVE_SUMMARY.txt", 'w') as f:
            f.write(summary)
        
        print(summary)


def main():
    parser = argparse.ArgumentParser(description="Generate case studies from failed trials")
    parser.add_argument("--db", required=True, help="Path to trials database")
    parser.add_argument("--output", default="./case_studies", help="Output directory")
    parser.add_argument("--num", type=int, default=10, help="Number of case studies")
    args = parser.parse_args()
    
    generator = CaseStudyGenerator(args.db)
    generator.generate_all_case_studies(args.output, args.num)


if __name__ == "__main__":
    main()
