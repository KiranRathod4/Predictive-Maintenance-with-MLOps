# tests/performance/load_test.py
"""
Load testing for API performance validation
"""
import time
import json
import requests
import statistics
import concurrent.futures
import os
from datetime import datetime

class APILoadTester:
    """Load tester for predictive maintenance API"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.sample_data = {
            "setting1": 42.0,
            "setting2": 0.84,
            "setting3": 100.0,
            "sensor_2": 642.35,
            "sensor_3": 1589.70,
            "sensor_4": 1400.60,
            "sensor_6": 21.61,
            "sensor_7": 554.36,
            "sensor_8": 2388.06,
            "sensor_9": 9046.19,
            "sensor_11": 47.47,
            "sensor_12": 521.66,
            "sensor_13": 2388.02,
            "sensor_14": 8138.62,
            "sensor_15": 8.4195,
            "sensor_17": 8.4195,
            "sensor_20": 0.03,
            "sensor_21": 0.02,
            "engine_id": 1,
            "cycle": 150
        }
        self.results = []
    
    def health_check(self):
        """Check if API is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return data.get('status') == 'healthy'
            return False
        except:
            return False
    
    def single_prediction_request(self):
        """Make a single prediction request and measure response time"""
        start_time = time.time()
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json=self.sample_data,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            success = response.status_code == 200
            
            result = {
                'response_time': response_time,
                'status_code': response.status_code,
                'success': success,
                'timestamp': datetime.now().isoformat()
            }
            
            if success:
                try:
                    data = response.json()
                    result['prediction'] = data.get('rul_prediction')
                except:
                    result['success'] = False
            
            return result
            
        except Exception as e:
            end_time = time.time()
            return {
                'response_time': end_time - start_time,
                'status_code': 0,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_load_test(self, num_requests=50, max_workers=10):
        """Run load test with concurrent requests"""
        print(f"🚀 Starting load test: {num_requests} requests, {max_workers} workers")
        
        # Health check first
        if not self.health_check():
            print("❌ API health check failed - skipping load test")
            return False
        
        # Run concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.single_prediction_request) 
                      for _ in range(num_requests)]
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                self.results.append(result)
        
        # Analyze results
        self.analyze_results(results)
        return True
    
    def analyze_results(self, results):
        """Analyze load test results"""
        successful_requests = [r for r in results if r['success']]
        failed_requests = [r for r in results if not r['success']]
        
        total_requests = len(results)
        success_count = len(successful_requests)
        failure_count = len(failed_requests)
        success_rate = (success_count / total_requests) * 100
        
        print(f"\n📊 Load Test Results:")
        print(f"Total Requests: {total_requests}")
        print(f"Successful: {success_count}")
        print(f"Failed: {failure_count}")
        print(f"Success Rate: {success_rate:.2f}%")
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            
            print(f"\n⏱️  Response Time Statistics:")
            print(f"Average: {statistics.mean(response_times):.3f}s")
            print(f"Median: {statistics.median(response_times):.3f}s")
            print(f"Min: {min(response_times):.3f}s")
            print(f"Max: {max(response_times):.3f}s")
            
            if len(response_times) > 1:
                print(f"Std Dev: {statistics.stdev(response_times):.3f}s")
            
            # Percentiles
            sorted_times = sorted(response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p99_idx = int(len(sorted_times) * 0.99)
            
            print(f"95th Percentile: {sorted_times[p95_idx]:.3f}s")
            print(f"99th Percentile: {sorted_times[p99_idx]:.3f}s")
            
            # Performance assertions
            avg_response_time = statistics.mean(response_times)
            p95_response_time = sorted_times[p95_idx]
            
            # Save results to file
            results_data = {
                'total_requests': total_requests,
                'success_rate': success_rate,
                'avg_response_time': avg_response_time,
                'p95_response_time': p95_response_time,
                'timestamp': datetime.now().isoformat(),
                'detailed_results': results
            }
            
            with open('performance_results.json', 'w') as f:
                json.dump(results_data, f, indent=2)
            
            # Performance thresholds
            assert success_rate >= 95, f"Success rate {success_rate:.2f}% below 95%"
            assert avg_response_time <= 2.0, f"Average response time {avg_response_time:.3f}s above 2s"
            assert p95_response_time <= 5.0, f"95th percentile {p95_response_time:.3f}s above 5s"
            
            print("\n✅ Performance test passed!")
        else:
            print("\n❌ No successful requests - performance test failed")
            raise Exception("All requests failed")

def main():
    """Main function for running load test"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run API load test')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--requests', type=int, default=50, help='Number of requests')
    parser.add_argument('--workers', type=int, default=10, help='Number of workers')
    
    args = parser.parse_args()
    
    # Use environment variables if available
    base_url = os.getenv('API_BASE_URL', args.url)
    num_requests = int(os.getenv('LOAD_TEST_REQUESTS', args.requests))
    max_workers = int(os.getenv('LOAD_TEST_WORKERS', args.workers))
    
    tester = APILoadTester(base_url)
    success = tester.run_load_test(num_requests, max_workers)
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()