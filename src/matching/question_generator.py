"""
Question Generator Module

Uses Claude AI to generate smart, patient-friendly questions
based on the eligibility criteria of matching trials.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import anthropic

logger = logging.getLogger(__name__)


@dataclass
class ScreeningQuestion:
    """A question to ask the patient."""
    id: str
    question: str
    type: str  # 'number', 'text', 'single_choice', 'multi_choice', 'boolean'
    options: List[str] = field(default_factory=list)
    required: bool = True
    help_text: str = ''
    criteria_coverage: int = 0  # How many trials this question helps filter


@dataclass
class QuestionSet:
    """Set of questions for a patient to answer."""
    condition: str
    trial_count: int
    questions: List[ScreeningQuestion]


class QuestionGenerator:
    """
    Generates smart screening questions using Claude AI.

    Analyzes eligibility criteria from multiple trials to find
    the most impactful questions to ask patients.
    """

    def __init__(self):
        """Initialize with Anthropic client."""
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )
        self.model = "claude-sonnet-4-20250514"

    def generate_questions(
        self,
        condition: str,
        trials: List[Any],
        max_questions: int = 8
    ) -> QuestionSet:
        """
        Generate screening questions based on trial eligibility criteria.

        Args:
            condition: The condition/disease being searched
            trials: List of matching trials with eligibility criteria
            max_questions: Maximum number of questions to generate

        Returns:
            QuestionSet with personalized questions
        """

        if not trials:
            return self._get_basic_questions(condition)

        # Extract all eligibility criteria
        criteria_texts = []
        for trial in trials[:50]:  # Limit to first 50 trials for prompt size
            if hasattr(trial, 'eligibility_criteria') and trial.eligibility_criteria:
                criteria_texts.append(f"Trial {trial.nct_id}:\n{trial.eligibility_criteria[:2000]}")

        if not criteria_texts:
            return self._get_basic_questions(condition)

        combined_criteria = "\n\n---\n\n".join(criteria_texts[:20])  # Further limit

        prompt = f"""You are helping patients find clinical trials they may qualify for.

Condition being searched: {condition}
Number of recruiting trials found: {len(trials)}

Here are eligibility criteria from some of these trials:

{combined_criteria}

Based on these criteria, generate {max_questions} screening questions that will help determine which trials a patient may be eligible for.

IMPORTANT GUIDELINES:
1. Use simple, patient-friendly language (no medical jargon)
2. Ask the most impactful questions first (ones that appear in most trials)
3. Always include: age (location is already collected separately)
4. Include questions about common inclusion/exclusion criteria you see
5. Provide multiple choice options where appropriate
6. Allow "I don't know" or "Not sure" as an option for medical questions

Return your response as a JSON array with this exact structure:
[
  {{
    "id": "age",
    "question": "What is your age?",
    "type": "number",
    "required": true,
    "help_text": "Most trials have age requirements"
  }},
  {{
    "id": "example_choice",
    "question": "Have you been diagnosed with X?",
    "type": "single_choice",
    "options": ["Yes", "No", "Not sure"],
    "required": true,
    "help_text": "Some explanation"
  }},
  {{
    "id": "example_multi",
    "question": "Which medications are you currently taking?",
    "type": "multi_choice",
    "options": ["Med A", "Med B", "Med C", "None of these", "Not sure"],
    "required": true,
    "help_text": "Select all that apply"
  }}
]

Valid types: "number", "text", "single_choice", "multi_choice", "boolean"

Return ONLY the JSON array, no other text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            # Parse the response
            response_text = response.content[0].text.strip()

            # Try to extract JSON from the response
            if response_text.startswith('['):
                questions_data = json.loads(response_text)
            else:
                # Try to find JSON array in response
                import re
                json_match = re.search(r'\[[\s\S]*\]', response_text)
                if json_match:
                    questions_data = json.loads(json_match.group())
                else:
                    logger.error(f"Could not parse questions from response: {response_text[:500]}")
                    return self._get_basic_questions(condition)

            # Convert to ScreeningQuestion objects
            questions = []
            for q in questions_data:
                questions.append(ScreeningQuestion(
                    id=q.get('id', f'q{len(questions)}'),
                    question=q.get('question', ''),
                    type=q.get('type', 'text'),
                    options=q.get('options', []),
                    required=q.get('required', True),
                    help_text=q.get('help_text', '')
                ))

            return QuestionSet(
                condition=condition,
                trial_count=len(trials),
                questions=questions
            )

        except Exception as e:
            logger.error(f"Error generating questions: {e}")
            return self._get_basic_questions(condition)

    def _get_basic_questions(self, condition: str) -> QuestionSet:
        """Return basic default questions when AI generation fails."""

        questions = [
            ScreeningQuestion(
                id="age",
                question="What is your age?",
                type="number",
                required=True,
                help_text="Most clinical trials have age requirements"
            ),
            ScreeningQuestion(
                id="sex",
                question="What is your biological sex?",
                type="single_choice",
                options=["Male", "Female", "Prefer not to say"],
                required=True,
                help_text="Some trials are specific to one sex"
            ),
            ScreeningQuestion(
                id="diagnosis_confirmed",
                question=f"Has a doctor diagnosed you with {condition}?",
                type="single_choice",
                options=["Yes", "No", "Awaiting diagnosis"],
                required=True,
                help_text="Most trials require a confirmed diagnosis"
            ),
            ScreeningQuestion(
                id="current_treatment",
                question="Are you currently receiving treatment for this condition?",
                type="single_choice",
                options=["Yes", "No", "Not sure"],
                required=True,
                help_text="Some trials require or exclude current treatments"
            ),
            ScreeningQuestion(
                id="other_conditions",
                question="Do you have any other major health conditions?",
                type="text",
                required=False,
                help_text="Some conditions may affect eligibility (optional)"
            )
        ]

        return QuestionSet(
            condition=condition,
            trial_count=0,
            questions=questions
        )

    def generate_followup_questions(
        self,
        condition: str,
        initial_answers: Dict[str, Any],
        remaining_trials: List[Any],
        max_questions: int = 4
    ) -> List[ScreeningQuestion]:
        """
        Generate follow-up questions to further narrow down trials.

        Called when initial screening leaves multiple possible matches
        and more specific questions could help narrow down.
        """

        if not remaining_trials or len(remaining_trials) <= 5:
            return []

        # Extract criteria from remaining trials
        criteria_texts = []
        for trial in remaining_trials[:20]:
            if hasattr(trial, 'eligibility_criteria') and trial.eligibility_criteria:
                criteria_texts.append(f"Trial {trial.nct_id}:\n{trial.eligibility_criteria[:1500]}")

        if not criteria_texts:
            return []

        prompt = f"""A patient is searching for {condition} clinical trials.

They have already answered these questions:
{json.dumps(initial_answers, indent=2)}

There are still {len(remaining_trials)} trials that might match.

Here are the eligibility criteria from some remaining trials:
{chr(10).join(criteria_texts[:10])}

Generate {max_questions} additional questions that would help narrow down which specific trials this patient qualifies for. Focus on criteria that differ between the remaining trials.

Use patient-friendly language. Return as JSON array:
[
  {{"id": "...", "question": "...", "type": "...", "options": [...], "help_text": "..."}}
]

Return ONLY the JSON array."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            questions_data = json.loads(response_text) if response_text.startswith('[') else []

            return [
                ScreeningQuestion(
                    id=q.get('id', f'followup_{i}'),
                    question=q.get('question', ''),
                    type=q.get('type', 'text'),
                    options=q.get('options', []),
                    required=q.get('required', False),
                    help_text=q.get('help_text', '')
                )
                for i, q in enumerate(questions_data)
            ]

        except Exception as e:
            logger.error(f"Error generating followup questions: {e}")
            return []
