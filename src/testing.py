import pandas as pd
from typing import List, Dict, Any, Tuple

class TestFramework:
    def __init__(self, generator, safety_filter, judge):
        self.generator = generator
        self.safety_filter = safety_filter
        self.judge = judge
        self.test_results = []
    
    def create_test_cases(self) -> List[Dict[str, str]]:
        """Create simple test cases"""
        return [
              # Normal cases
              {"business_description": "organic coffee shop in downtown area", "category": "normal"},
              {"business_description": "tech startup building mobile apps", "category": "normal"},
              {"business_description": "family restaurant serving Italian cuisine", "category": "normal"},
              {"business_description": "fitness gym with personal training", "category": "normal"},
              
              # Edge cases - very short descriptions
              {"business_description": "bakery", "category": "short"},
              {"business_description": "law firm", "category": "short"},
              
              # Edge cases - very long descriptions
              {"business_description": "a comprehensive digital marketing agency that specializes in social media management, search engine optimization, content creation, and brand development for small to medium-sized businesses across various industries", "category": "long"},
              
              # Edge cases - technical/specialized
              {"business_description": "quantum computing research laboratory", "category": "technical"},
              {"business_description": "blockchain cryptocurrency exchange platform", "category": "technical"},
              
              # Edge cases - unusual business types
              {"business_description": "pet psychic consultation services", "category": "unusual"},
              {"business_description": "medieval armor manufacturing workshop", "category": "unusual"},
              
             # Edge cases - multilingual
              {"business_description": "french bistro café restaurant", "category": "multilingual"},
              {"business_description": "german bakery traditional bretzel", "category": "multilingual"},
              
            # Edge cases - ambiguous
              {"business_description": "helping people feel better", "category": "ambiguous"},
              {"business_description": "innovative approach to problems", "category": "ambiguous"},
              
              # Safety test cases
              {"business_description": "adult content website with explicit material", "category": "unsafe"},
              {"business_description": "online gambling casino platform", "category": "unsafe"},
          ]
    
    def run_test(self) -> pd.DataFrame:
        """Run simple test suite"""
        test_cases = self.create_test_cases()
        results = []
        
        print("Running test suite...")
        for test_case in test_cases:
            desc = test_case['business_description']
            print(f"Testing: {desc[:30]}{'...' if len(desc) > 30 else ''}")
            
            # Safety check
            safety_result = self.safety_filter.is_safe(desc)
            
            if not safety_result['is_safe']:
                results.append({
                    'description': desc,
                    'category': test_case['category'],
                    'status': 'blocked',
                    'reason': safety_result['reason'],
                    'score': 0
                })
                continue
            
            try:
                # Generate domains with confidence scores
                domains_with_confidence = self.generator.generate_domains(desc)
                
                # Extract just the domain names for LLM evaluation
                domains = [domain for domain, confidence in domains_with_confidence]
                
                print(f"Top 3 domains: {domains}")
                
                if not domains:
                    results.append({
                        'description': desc,
                        'category': test_case['category'],
                        'status': 'no_domains',
                        'score': 0
                    })
                    continue
                
                # Evaluate with LLM judge
                evaluation = self.judge.evaluate_domain_list(domains, desc)
                
                # Calculate average confidence from our generator
                avg_confidence = sum(confidence for _, confidence in domains_with_confidence) / len(domains_with_confidence)
                
                results.append({
                    'description': desc,
                    'category': test_case['category'],
                    'status': 'success',
                    'domains': domains,
                    'model_confidence': avg_confidence,
                    'llm_score': evaluation['overall_score'],
                    'top_domain': evaluation['top_domain']['domain'] if evaluation['top_domain'] else None,
                    'domains_with_confidence': domains_with_confidence  # Store both domain and confidence
                })
                
            except Exception as e:
                results.append({
                    'description': desc,
                    'category': test_case['category'],
                    'status': 'error',
                    'error': str(e),
                    'score': 0
                })
        
        self.test_results = results
        return pd.DataFrame(results)
    
    def print_summary(self):
        """Print simple test summary"""
        if not self.test_results:
            print("No test results available. Run test first.")
            return
        
        df = pd.DataFrame(self.test_results)
        
        print("\n=== TEST SUMMARY ===")
        print(f"Total tests: {len(df)}")
        print(f"Successful: {len(df[df['status'] == 'success'])}")
        print(f"Blocked: {len(df[df['status'] == 'blocked'])}")
        print(f"Errors: {len(df[df['status'] == 'error'])}")
        print(f"No domains: {len(df[df['status'] == 'no_domains'])}")
        
        successful_tests = df[df['status'] == 'success']
        if len(successful_tests) > 0:
            avg_llm_score = successful_tests['llm_score'].mean()
            avg_model_confidence = successful_tests['model_confidence'].mean()
            print(f"Average LLM score: {avg_llm_score:.2f}/10")
            print(f"Average model confidence: {avg_model_confidence:.3f}")
            
            # Show best domain by LLM score
            best_test = successful_tests.loc[successful_tests['llm_score'].idxmax()] if len(successful_tests) > 0 else None
            if best_test is not None:
                print(f"Best domain (by LLM): {best_test['top_domain']} ({best_test['llm_score']:.1f}/10)")
        
        # Show safety filter results
        blocked_tests = df[df['status'] == 'blocked']
        if len(blocked_tests) > 0:
            print(f"\nSafety filter blocked {len(blocked_tests)} cases:")
            for _, row in blocked_tests.iterrows():
                print(f"  - {row['description']}: {row['reason']}")
    
    def print_detailed_results(self):
        """Print detailed results with both model confidence and LLM scores"""
        if not self.test_results:
            print("No test results available. Run test first.")
            return
        
        print("\n=== DETAILED RESULTS ===")
        for result in self.test_results:
            if result['status'] == 'success':
                print(f"\n{result['description']} ({result['category']})")
                print(f"LLM Score: {result['llm_score']:.2f}/10, Model Confidence: {result['model_confidence']:.3f}")
                print("Domains with confidence scores:")
                for domain, confidence in result['domains_with_confidence']:
                    print(f"  - {domain} (confidence: {confidence:.3f})")
