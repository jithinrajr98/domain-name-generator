from groq import Groq
from typing import List, Dict, Any, Tuple
import json
from dotenv import load_dotenv
import os
import re
load_dotenv()

class LLMJudgeEvaluator:
    """LLM-based evaluation system for domain name quality"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):

        api_key = None
        if 'GROQ_API_KEY' in os.environ:
            api_key = os.environ['GROQ_API_KEY']
        
        elif not api_key:
            raise ValueError("GROQ_API_KEY not found in Streamlit secrets or environment variables.")
        self.groq_client = Groq(api_key=self.api_key)  
        self.GROQ_MODEL = "llama-3.3-70b-versatile"	
        
    def evaluate_single_domain(self, domain: str, business_description: str) -> Dict[str, Any]:
        """Evaluate a single domain name"""
        prompt = self._create_evaluation_prompt(domain, business_description)
        
        try:
            response = self.groq_client.chat.completions.create( messages=[
                  {"role": "user", "content": prompt}],
                  model=self.GROQ_MODEL )
            
            result_text = response.choices[0].message.content.strip()
            return self._parse_evaluation_result(result_text, domain)
            
        except Exception as e:
            return self._create_fallback_evaluation(domain, str(e))
    
    def evaluate_domain_list(self, domains: List[str], business_description: str) -> Dict[str, Any]:
        """Evaluate multiple domains and rank them"""
        if not domains:
            return {"overall_score": 0, "domains": [], "error": "No domains to evaluate"}
        
        evaluations = []
        for domain in domains:
            evaluation = self.evaluate_single_domain(domain, business_description)
            evaluations.append(evaluation)
        
        # Sort by overall score
        evaluations.sort(key=lambda x: x.get('overall_score', 0), reverse=True)
        
        return {
            "overall_score": self._calculate_average_score(evaluations),
            "top_domain": evaluations[0] if evaluations else None,
            "domains": evaluations,
            "total_evaluated": len(evaluations)
        }
    
    def _create_evaluation_prompt(self, domain: str, business_description: str) -> str:
        return f"""
        Evaluate this domain name for the business described below. Provide a comprehensive evaluation.

        BUSINESS DESCRIPTION: "{business_description}"
        DOMAIN NAME: "{domain}"

        Please evaluate based on these criteria (score each 1-10):
        1. RELEVANCE: How well does the domain relate to the business?
        2. MEMORABILITY: Is it easy to remember and type?
        3. BRANDABILITY: Could this be a strong brand name?
        4. TECHNICAL QUALITY: Consider length, hyphens, numbers, spelling
        5. CREATIVITY: How original and unique is it?

        Provide your response in this exact JSON format:
        {{
            "domain": "{domain}",
            "scores": {{
                "relevance": 0,
                "memorability": 0,
                "brandability": 0,
                "technical_quality": 0,
                "creativity": 0
            }},
            "overall_score": 0,
  
        }}

        Calculate overall_score as the average of the 5 criteria scores.
        Be honest and critical in your evaluation.
        """
    
    def _parse_evaluation_result(self, result_text: str, domain: str) -> Dict[str, Any]:
        """Parse the LLM evaluation result"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
                # Ensure all required fields are present
                evaluation.setdefault('domain', domain)
                evaluation.setdefault('scores', {})
                evaluation.setdefault('overall_score', 0)
      
                return evaluation
        except json.JSONDecodeError:
            pass
        
        # Fallback if JSON parsing fails
        return self._create_fallback_evaluation(domain, "Failed to parse evaluation")
    
    def _create_fallback_evaluation(self, domain: str, error: str = None) -> Dict[str, Any]:
        """Create a fallback evaluation when LLM fails"""
        return {
            "domain": domain,
            "scores": {
                "relevance": 5,
                "memorability": 5,
                "brandability": 5,
                "technical_quality": 5,
                "creativity": 5
            },
            "overall_score": 5,
            "error": error
        }
    
    def _calculate_average_score(self, evaluations: List[Dict]) -> float:
        """Calculate average overall score from evaluations"""
        if not evaluations:
            return 0
        total = sum(eval.get('overall_score', 0) for eval in evaluations)
        return total / len(evaluations)