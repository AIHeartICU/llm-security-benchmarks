"""
Basic test script to demonstrate the framework functionality
"""

import sys
import json
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.evaluation_framework import SecurityEvaluator

class MockModelInterface:
    """Mock model interface for testing"""
    
    def __init__(self, refusal_rate=0.7):
        self.refusal_rate = refusal_rate
        self.calls = 0
        
    def process_query(self, query):
        """Process a mock query"""
        import random
        self.calls += 1
        
        # Randomly determine if query contains concerning content
        is_concerning = random.random() < self.refusal_rate
        
        if is_concerning:
            return {
                "content": [{"text": "I cannot provide information on that topic."}],
                "stop_reason": "refusal",
                "metadata": {
                    "input_classifier_score": 0.85,
                    "output_classifier_score": 0.92
                }
            }
        else:
            return {
                "content": [{"text": f"Here is information about: {query}"}],
                "stop_reason": "end_turn",
                "metadata": {
                    "input_classifier_score": 0.15,
                    "output_classifier_score": 0.08
                }
            }

def main():
    """Run a basic test of the framework"""
    print("Starting basic test of security evaluation framework...")
    
    # Create output directory if it doesn't exist
    os.makedirs("test_outputs/results", exist_ok=True)
    
    # Initialize the security evaluator
    evaluator = SecurityEvaluator()
    
    # Create mock model
    mock_model = MockModelInterface()
    
    # Set up test parameters
    test_models = [
        {"name": "model_safe", "refusal_rate": 0.9},
        {"name": "model_medium", "refusal_rate": 0.7},
        {"name": "model_weak", "refusal_rate": 0.4}
    ]
    
    # Run tests on different mock models
    for model_config in test_models:
        print(f"\nTesting model: {model_config['name']}")
        
        # Configure mock model
        mock_model.refusal_rate = model_config["refusal_rate"]
        mock_model.calls = 0
        
        # Run evaluation
        results = evaluator.evaluate_model(
            model_id=model_config["name"],
            assessment_level=2,
            domains=["general", "specialized"]
        )
        
        # Print summary
        security_score = results["metrics"].get("overall_security_score", 0)
        print(f"Security Score: {security_score:.2f}")
        print(f"API Calls: {mock_model.calls}")
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_outputs/results/{model_config['name']}_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    # Generate consolidated report
    report = evaluator.generate_report()
    report_path = f"test_outputs/results/consolidated_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nConsolidated report saved to {report_path}")
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()