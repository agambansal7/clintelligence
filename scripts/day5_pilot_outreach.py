"""
PILOT OUTREACH STRATEGY - Day 5 Sprint

Your goal: Get 2-3 companies to agree to a FREE pilot within 48 hours.

This file contains:
1. Target company list (20 small biotechs)
2. Email templates
3. LinkedIn message templates
4. Follow-up sequence
5. Discovery call script

The key insight: Small biotechs (Series A/B, 20-100 employees) are PERFECT because:
- They're about to run trials and need help
- They don't have internal infrastructure
- Decision-makers are accessible
- They can say "yes" quickly
- They become case studies AND potential customers
"""

# ==============================================================================
# TARGET COMPANY LIST
# ==============================================================================

TARGET_COMPANIES = """
TIER 1: IDEAL TARGETS (Recently funded, about to run trials)
------------------------------------------------------------

1. Centessa Pharmaceuticals
   - Focus: Multiple programs
   - Stage: Public (raised $250M+)
   - Why: Large pipeline, needs efficiency
   - Contact: Find on LinkedIn - Head of Clinical Operations

2. Disc Medicine
   - Focus: Hematology
   - Stage: Series C ($140M)
   - Why: Building clinical team, open to tools
   - Contact: Chief Medical Officer

3. Boundless Bio
   - Focus: Oncology (ecDNA)
   - Stage: Series C ($105M)
   - Why: Novel approach, needs strong trial design
   - Contact: VP Clinical Development

4. Adagene
   - Focus: Oncology
   - Stage: Public
   - Why: Multiple Phase 1/2 trials
   - Contact: Clinical Operations team

5. ArriVent Biopharma
   - Focus: Oncology
   - Stage: Series D ($200M)
   - Why: Aggressive clinical timeline
   - Contact: COO or CMO


TIER 2: GOOD TARGETS (Active in clinical development)
-----------------------------------------------------

6. Eliem Therapeutics
   - Focus: Neurology/Pain
   - Stage: Public
   - Contact: VP Clinical Operations

7. Arcus Biosciences
   - Focus: Oncology
   - Stage: Public
   - Contact: Head of Clinical Development

8. Day One Biopharmaceuticals
   - Focus: Pediatric Oncology
   - Stage: Public
   - Contact: CMO

9. ORIC Pharmaceuticals
   - Focus: Oncology
   - Stage: Public
   - Contact: VP Clinical Operations

10. Bicycle Therapeutics
    - Focus: Oncology
    - Stage: Public
    - Contact: Clinical team


TIER 3: EMERGING COMPANIES (Series A/B, building teams)
-------------------------------------------------------

11-20: Search for these on LinkedIn/Crunchbase:
    - Recently funded (6-12 months ago)
    - Raised $30-100M
    - "Clinical" or "Phase 1" mentioned in news
    - Has hired clinical operations roles recently


HOW TO FIND DECISION MAKERS
---------------------------
1. LinkedIn Sales Navigator (free trial)
   - Search: "Clinical Operations" + company name
   - Search: "CMO" OR "Chief Medical Officer" + company name
   - Search: "VP Clinical Development" + company name

2. Company website → Leadership page

3. ClinicalTrials.gov → Search company → Find PI names

4. Press releases → Often mention clinical leadership
"""


# ==============================================================================
# EMAIL TEMPLATES
# ==============================================================================

EMAIL_COLD_OUTREACH = """
Subject: Reducing trial amendments at {company_name} (data from 50K+ trials)

Hi {first_name},

Quick question: do you know your protocol amendment probability before you start a trial?

I built a tool that analyzes 50,000+ historical trials from ClinicalTrials.gov to predict:
- Amendment likelihood (avg trial has 2-3 amendments at $150K+ each)
- Enrollment delay risk (85% of trials are delayed 30+ days)
- Site performance by therapeutic area

I'm offering **free pilots** to 3 biotech companies this month.

What you'd get:
- Risk score on your current/planned protocol
- Site recommendations based on historical performance
- Endpoint benchmarking vs similar trials

Takes 30 minutes. No commitment. You keep the analysis.

Would you have 15 minutes this week to see if it's useful?

Best,
{your_name}

P.S. I analyzed {example_terminated_trial} - a {therapeutic_area} trial terminated for enrollment issues. Our tool would have flagged {specific_issue} before Day 1.
"""


EMAIL_FOLLOW_UP_1 = """
Subject: Re: Reducing trial amendments at {company_name}

Hi {first_name},

Following up - I know you're busy.

One data point: I analyzed 847 terminated Phase 3 trials. 34% cited enrollment challenges that were predictable from the protocol design.

Happy to show you the analysis on any of your planned/active trials. Free, takes 30 min.

Would Thursday or Friday work?

{your_name}
"""


EMAIL_FOLLOW_UP_2 = """
Subject: Re: Reducing trial amendments at {company_name}

Hi {first_name},

Last note from me. 

If protocol optimization isn't a priority right now, I totally understand. But if you're planning a trial in the next 6 months, the analysis is more valuable earlier.

Let me know if timing is better in Q2.

{your_name}
"""


EMAIL_WARM_INTRO_REQUEST = """
Subject: Quick favor - intro to {target_name}?

Hi {connector_name},

I saw you're connected to {target_name} at {company_name} on LinkedIn.

I built a clinical trial intelligence tool that's getting traction - it predicts protocol amendments and enrollment issues using data from 50K trials. Offering free pilots to biotechs.

Would you be open to a quick intro? Happy to make it easy:

"Hey {target_first_name}, my friend {your_name} built something interesting for clinical ops teams - tool that predicts trial risks. Worth 15 min if you're planning any trials."

No pressure either way. Appreciate you!

{your_name}
"""


# ==============================================================================
# LINKEDIN TEMPLATES
# ==============================================================================

LINKEDIN_CONNECTION_REQUEST = """
Hi {first_name} - I'm building tools for clinical trial optimization using ClinicalTrials.gov data. Would love to connect and share what I'm learning about protocol risk factors.
"""


LINKEDIN_MESSAGE_AFTER_CONNECT = """
Thanks for connecting, {first_name}!

I've been analyzing terminated trials to understand why they fail. Found that 34% of Phase 3 terminations cite enrollment issues that were often predictable from protocol design.

I built a tool that scores protocols against 50K+ historical trials to predict:
- Amendment probability
- Enrollment delay risk
- Site performance

Offering free pilots to a few biotech companies this month. Takes 30 min, you keep the analysis.

Would {company_name} be interested?
"""


LINKEDIN_INMAIL = """
Hi {first_name},

I noticed {company_name} is running/planning trials in {therapeutic_area}.

I built a tool that predicts protocol amendment risk using data from 50K+ trials. Average Phase 3 has 2-3 amendments at $150K+ each - often for eligibility criteria issues we can identify before Day 1.

Offering free pilots to 3 biotechs this month. You'd get:
✓ Risk score on your protocol
✓ Site recommendations for your indication
✓ Endpoint benchmarking

Would 15 minutes this week work to see if it's useful?

{your_name}
"""


# ==============================================================================
# DISCOVERY CALL SCRIPT
# ==============================================================================

DISCOVERY_CALL_SCRIPT = """
TRIALINTEL PILOT DISCOVERY CALL
===============================
Duration: 15-20 minutes
Goal: Qualify interest, schedule pilot analysis

OPENING (2 min)
---------------
"Thanks for taking the time, {name}. I'll keep this brief.

Quick background: I've been analyzing ClinicalTrials.gov data - 500K+ trials - to understand why trials fail and how to predict issues earlier.

Built a tool that scores protocols for risk before you start. Offering free pilots to validate it with real users.

Before I show you anything - can you tell me a bit about what {company} has in the pipeline?"


DISCOVERY QUESTIONS (5 min)
---------------------------
1. "What trials are you planning or currently running?"
   (Listen for: therapeutic area, phase, timeline)

2. "What's your biggest operational challenge right now?"
   (Listen for: enrollment, sites, timeline, budget)

3. "Have you experienced protocol amendments in past trials?"
   (Listen for: pain points, costs, frustration)

4. "How do you currently select sites for trials?"
   (Listen for: manual process, consultants, guessing)

5. "Who else would be involved in a decision to use a tool like this?"
   (Listen for: CMO, Head of Ops, procurement)


DEMO OFFER (3 min)
------------------
"Based on what you shared, here's what I can offer for a pilot:

1. Risk Score: I'll analyze your [planned/current] protocol and show you:
   - Amendment probability vs similar trials
   - Specific eligibility criteria that might cause issues
   - Benchmark against {N} similar trials in {indication}

2. Site Intelligence: For your indication, I'll show you:
   - Top-performing sites from historical data
   - Enrollment velocity predictions
   - Diversity access scores

3. Endpoint Analysis: What endpoints work in {indication}:
   - Success/failure rates by endpoint type
   - Regulatory precedent

Takes about 30 minutes to walk through. You keep all the analysis.

Would next week work to do this? I just need your draft protocol or eligibility criteria."


CLOSE (2 min)
-------------
If yes:
"Great. I'll send a calendar invite. Can you share your draft protocol or eligibility criteria before the call? Even a rough draft works."

If hesitant:
"I understand. What would make this more valuable for you?"

If no:
"No problem. Mind if I check back in [3 months] when you're further along?"


FOLLOW-UP EMAIL AFTER CALL
--------------------------
Subject: TrialIntel pilot for {company} - next steps

Hi {name},

Great speaking with you. As discussed, here's what I'll prepare for our pilot session on [DATE]:

1. Protocol Risk Score for your {indication} trial
2. Site recommendations based on historical {indication} performance  
3. Endpoint benchmarking vs similar completed trials

To prepare, please share:
- Draft protocol or eligibility criteria
- Target enrollment number
- Planned number of sites

I'll have the analysis ready for [DATE]. Looking forward to it.

Best,
{your_name}
"""


# ==============================================================================
# SUCCESS METRICS
# ==============================================================================

SUCCESS_METRICS = """
DAY 5 SUCCESS METRICS
=====================

OUTREACH TARGETS:
- LinkedIn connection requests sent: 50
- Cold emails sent: 20
- Warm intro requests sent: 5
- InMails sent: 10

RESPONSE TARGETS (within 48 hours):
- Replies received: 5-10
- Calls scheduled: 3-5
- Pilots committed: 2-3

PILOT SUCCESS CRITERIA:
- Completed analysis delivered: Yes
- Client found it valuable: Yes (NPS > 7)
- Quote/testimonial received: Yes
- Referral to colleague: Bonus

FOLLOW-UP CADENCE:
- Day 0: Initial outreach
- Day 2: First follow-up
- Day 5: Second follow-up
- Day 14: Check back
"""


# ==============================================================================
# TRACKING TEMPLATE
# ==============================================================================

TRACKING_TEMPLATE = """
OUTREACH TRACKING SPREADSHEET
============================

| Company | Contact | Title | Email | LinkedIn | Status | Next Step | Notes |
|---------|---------|-------|-------|----------|--------|-----------|-------|
|         |         |       |       |          |        |           |       |

STATUS OPTIONS:
- Researching
- Contacted (date)
- Replied
- Call Scheduled (date)
- Pilot Committed
- Pilot Complete
- Not Interested
- No Response

Create this in Google Sheets or Notion for real-time tracking.
"""


def print_outreach_guide():
    """Print the complete outreach guide."""
    print("="*70)
    print("TRIALINTEL PILOT OUTREACH GUIDE")
    print("="*70)
    print("\n1. TARGET COMPANIES")
    print(TARGET_COMPANIES)
    print("\n2. EMAIL TEMPLATES")
    print(EMAIL_COLD_OUTREACH)
    print("\n3. LINKEDIN TEMPLATES")
    print(LINKEDIN_CONNECTION_REQUEST)
    print(LINKEDIN_MESSAGE_AFTER_CONNECT)
    print("\n4. DISCOVERY CALL SCRIPT")
    print(DISCOVERY_CALL_SCRIPT)
    print("\n5. SUCCESS METRICS")
    print(SUCCESS_METRICS)


if __name__ == "__main__":
    print_outreach_guide()
