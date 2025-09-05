
from typing import List, Dict, Any


class SafetyFilter:
    def __init__(self):
        """Initialize safety filter with inappropriate content patterns"""
        self.inappropriate_keywords = [
            'adult', 'porn', 'xxx', 'sex', 'nude', 'naked', 'explicit',
            'gambling', 'casino', 'bet', 'drug', 'illegal', 'weapon',
            'hate', 'violence', 'terror', 'scam', 'fraud', 'kill', 'murder'
        ]
        
    def is_safe(self, business_description: str) -> Dict[str, Any]:
        """
        Check if a business description is safe to process
        
        Args:
            business_description: The business description to check
            
        Returns:
            Dictionary with safety status and details
        """
        description_lower = business_description.lower()
        
        for keyword in self.inappropriate_keywords:
            if keyword in description_lower:
                return {
                    'is_safe': False,
                    'reason': f'Contains inappropriate keyword: {keyword}',
                    'blocked_content': keyword
                }
        
        return {'is_safe': True, 'reason': 'Content is safe'}