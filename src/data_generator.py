import json
import random
from typing import List, Dict



import json
import random
from typing import List, Dict

class DomainDatasetGenerator:
    def __init__(self):
        self.business_types = [
            "restaurant", "cafe", "tech startup", "consulting firm", 
            "clothing store", "bookshop", "gym", "yoga studio",
            "digital agency", "photography studio", "law firm", 
            "medical practice", "real estate agency", "travel agency",
            "tutoring service", "event planning", "catering service",
            "cleaning service", "construction company", "non-profit"
        ]
        
        self.adjectives = [
            "modern", "premium", "elite", "prime", "superior", "advanced",
            "creative", "innovative", "dynamic", "professional", "expert",
            "reliable", "trusted", "quality", "affordable", "luxury",
            "sustainable", "organic", "natural", "artisanal", "handcrafted"
        ]
        
        self.locations = [
            "new york", "london", "paris", "tokyo", "sydney", "toronto",
            "chicago", "san francisco", "los angeles", "seattle", "boston",
            "austin", "miami", "dallas", "denver", "phoenix", "atlanta",
            "orlando", "las vegas", "portland", "vancouver", "montreal"
        ]
        
        self.tlds = [".com", ".org", ".net", ".io", ".co"]
        
    def generate_business_description(self) -> str:
        business_type = random.choice(self.business_types)
        adjective = random.choice(self.adjectives)
        location = random.choice(self.locations)
        
        templates = [
            f"{adjective} {business_type} in {location}",
            f"{business_type} specializing in {adjective} services in {location}",
            f"{location}'s premier {adjective} {business_type}",
            f"{adjective} {business_type} for {location} residents",
            f"{business_type} offering {adjective} products in {location}"
        ]
        
        return random.choice(templates)
    
    def generate_domain_suggestions(self, business_desc: str, count: int = 3) -> List[str]:
        words = business_desc.lower().split()
        keywords = [word for word in words if word not in ['a', 'an', 'the', 'in', 'for', 'with', 'and', 'of']]
        
        suggestions = []
        for _ in range(count):
            # Various domain generation strategies
            strategies = [
                # Combine keywords
                lambda: ''.join(random.sample(keywords, min(len(keywords), 2))),
                # Use first letters
                lambda: ''.join([word[0] for word in keywords if len(word) > 0]),
                # Hyphenated version
                lambda: '-'.join(random.sample(keywords, min(len(keywords), 2))),
                # Add common suffixes
                lambda: random.choice(keywords) + random.choice(['hub', 'ly', 'ify', 'zen', 'able', 'io']),
                # Add common prefixes
                lambda: random.choice(['get', 'my', 'the', 'go']) + random.choice(keywords)
            ]
            
            domain = random.choice(strategies)()
            tld = random.choice(self.tlds)
            suggestions.append(f"{domain}{tld}")
        
        return suggestions
    
    def generate_dataset(self, size: int = 1000) -> List[Dict]:
        dataset = []
        for _ in range(size):
            business_desc = self.generate_business_description()
            domains = self.generate_domain_suggestions(business_desc)
            
            dataset.append({
                "business_description": business_desc,
                "domain_suggestions": domains
            })
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        with open(filename, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')
    
    def load_dataset(self, filename: str) -> List[Dict]:
        dataset = []
        with open(filename, 'r') as f:
            for line in f:
                dataset.append(json.loads(line))
        return dataset
    

if __name__ == "__main__":
    generator = DomainDatasetGenerator()
    