#!/usr/bin/env python3
"""
SCAFAD Delta Test Suite Runner

A comprehensive test runner for the SCAFAD Delta system that provides:
- Multiple test execution modes
- Performance benchmarking
- Coverage reporting
- Test result analysis
- Custom test filtering
"""

import argparse
import sys
import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test configuration
TEST_CATEGORIES = {
    'unit': {
        'path': 'tests/unit',
        'description': 'Unit tests for individual components',
        'markers': ['unit']
    },
    'integration': {
        'path': 'tests/integration',
        'description': 'Integration tests for component interactions',
        'markers': ['integration']
    },
    'performance': {
        'path': 'tests/performance',
        'description': 'Performance and benchmark tests',
        'markers': ['performance', 'benchmark']
    },
    'preservation': {
        'path': 'tests/preservation',
        'description': 'Data preservation specific tests',
        'markers': ['preservation']
    },
    'privacy': {
        'path': 'tests/privacy',
        'description': 'Privacy and compliance tests',
        'markers': ['privacy']
    },
    'schema': {
        'path': 'tests/schema',
        'description': 'Schema evolution tests',
        'markers': ['schema']
    },
    'validation': {
        'path': 'tests/validation',
        'description': 'Validation system tests',
        'markers': ['validation']
    },
    'sanitization': {
        'path': 'tests/sanitization',
        'description': 'Data sanitization tests',
        'markers': ['sanitization']
    },
    'hashing': {
        'path': 'tests/hashing',
        'description': 'Hashing functionality tests',
        'markers': ['hashing']
    }
}

# Test execution modes
EXECUTION_MODES = {
    'fast': {
        'description': 'Fast execution with minimal coverage',
        'options': ['--tb=short', '--maxfail=3', '--durations=5']
    },
    'standard': {
        'description': 'Standard execution with good coverage',
        'options': ['--tb=short', '--maxfail=5', '--durations=10']
    },
    'thorough': {
        'description': 'Thorough execution with maximum coverage',
        'options': ['--tb=short', '--maxfail=10', '--durations=20']
    },
    'performance': {
        'description': 'Performance-focused execution',
        'options': ['--benchmark-only', '--durations=50']
    },
    'debug': {
        'description': 'Debug mode with verbose output',
        'options': ['--tb=long', '--verbose', '--capture=no']
    }
}


class TestRunner:
    """Main test runner class."""
    
    def __init__(self, args):
        self.args = args
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        required_packages = ['pytest', 'pytest-cov', 'pytest-benchmark']
        missing_packages = []
        
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All required packages are installed")
        return True
    
    def discover_tests(self):
        """Discover available test files."""
        test_files = {}
        
        for category, config in TEST_CATEGORIES.items():
            category_path = Path(config['path'])
            if category_path.exists():
                test_files[category] = []
                for test_file in category_path.glob('test_*.py'):
                    test_files[category].append(test_file.name)
        
        return test_files
    
    def print_test_discovery(self):
        """Print discovered test files."""
        print("\n🔍 Test Discovery Results:")
        print("=" * 50)
        
        test_files = self.discover_tests()
        
        for category, files in test_files.items():
            if files:
                print(f"\n📁 {category.upper()} Tests ({len(files)} files):")
                for file in files:
                    print(f"   • {file}")
            else:
                print(f"\n📁 {category.upper()} Tests: No test files found")
    
    def build_pytest_command(self, categories=None, mode='standard', additional_options=None):
        """Build pytest command with appropriate options."""
        base_cmd = ['python', '-m', 'pytest']
        
        # Add execution mode options
        if mode in EXECUTION_MODES:
            base_cmd.extend(EXECUTION_MODES[mode]['options'])
        
        # Add category-specific paths
        if categories:
            for category in categories:
                if category in TEST_CATEGORIES:
                    base_cmd.append(TEST_CATEGORIES[category]['path'])
        else:
            base_cmd.append('tests/')
        
        # Add coverage options if requested
        if self.args.coverage:
            base_cmd.extend([
                '--cov=core',
                '--cov=subsystems',
                '--cov=utils',
                '--cov-report=term-missing',
                '--cov-report=html:htmlcov',
                '--cov-fail-under=80'
            ])
        
        # Add benchmark options if in performance mode
        if mode == 'performance':
            base_cmd.extend([
                '--benchmark-only',
                '--benchmark-skip',
                '--benchmark-min-rounds=5'
            ])
        
        # Add additional options
        if additional_options:
            base_cmd.extend(additional_options)
        
        # Add verbose output if requested
        if self.args.verbose:
            base_cmd.append('-v')
        
        # Add parallel execution if requested
        if self.args.parallel and self.args.parallel > 1:
            base_cmd.extend(['-n', str(self.args.parallel)])
        
        return base_cmd
    
    def run_tests(self, categories=None, mode='standard'):
        """Run tests with specified configuration."""
        print(f"\n🚀 Running tests in {mode.upper()} mode...")
        if categories:
            print(f"📋 Categories: {', '.join(categories)}")
        
        # Build command
        cmd = self.build_pytest_command(categories, mode)
        
        print(f"🔧 Command: {' '.join(cmd)}")
        print("-" * 50)
        
        # Run tests
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.args.verbose,
                text=True,
                cwd=project_root
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Store results
            self.results[mode] = {
                'return_code': result.return_code,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'duration': duration,
                'success': result.return_code == 0
            }
            
            # Print results
            if result.return_code == 0:
                print(f"✅ Tests completed successfully in {duration:.2f}s")
            else:
                print(f"❌ Tests failed with return code {result.return_code}")
                if result.stderr:
                    print(f"Error output:\n{result.stderr}")
            
            return result.return_code == 0
            
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            return False
    
    def run_all_tests(self):
        """Run all test categories."""
        print("\n🎯 Running all test categories...")
        
        success = True
        for category in TEST_CATEGORIES.keys():
            print(f"\n📁 Testing {category}...")
            if not self.run_tests([category], 'standard'):
                success = False
        
        return success
    
    def run_performance_benchmarks(self):
        """Run performance benchmarks."""
        print("\n⚡ Running performance benchmarks...")
        
        return self.run_tests(mode='performance')
    
    def generate_report(self):
        """Generate test execution report."""
        if not self.results:
            print("No test results to report")
            return
        
        print("\n📊 Test Execution Report")
        print("=" * 50)
        
        for mode, result in self.results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            print(f"\n{mode.upper()} Mode: {status}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Return Code: {result['return_code']}")
            
            if result['stderr']:
                print(f"Errors: {result['stderr'][:200]}...")
        
        # Summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"\n📈 Summary:")
        print(f"Total Test Runs: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    def save_results(self, filename=None):
        """Save test results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"test_results_{timestamp}.json"
        
        # Prepare results for JSON serialization
        serializable_results = {}
        for mode, result in self.results.items():
            serializable_results[mode] = {
                'return_code': result['return_code'],
                'duration': result['duration'],
                'success': result['success'],
                'timestamp': datetime.now().isoformat()
            }
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"💾 Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    def run(self):
        """Main execution method."""
        print("🧪 SCAFAD Delta Test Suite Runner")
        print("=" * 50)
        
        # Check dependencies
        if not self.check_dependencies():
            return 1
        
        # Discover tests
        if self.args.discover:
            self.print_test_discovery()
            return 0
        
        # Run tests based on arguments
        success = True
        
        if self.args.all:
            success = self.run_all_tests()
        elif self.args.performance:
            success = self.run_performance_benchmarks()
        elif self.args.categories:
            success = self.run_tests(self.args.categories, self.args.mode)
        else:
            # Default: run all tests
            success = self.run_all_tests()
        
        # Generate report
        self.generate_report()
        
        # Save results if requested
        if self.args.save_results:
            self.save_results(self.args.output_file)
        
        return 0 if success else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SCAFAD Delta Test Suite Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests
  python run_tests.py --all
  
  # Run specific categories
  python run_tests.py --categories unit privacy validation
  
  # Run performance benchmarks
  python run_tests.py --performance
  
  # Run with custom mode
  python run_tests.py --categories unit --mode thorough
  
  # Discover available tests
  python run_tests.py --discover
  
  # Run with coverage and save results
  python run_tests.py --all --coverage --save-results
        """
    )
    
    # Test execution options
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all test categories'
    )
    
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=list(TEST_CATEGORIES.keys()),
        help='Specific test categories to run'
    )
    
    parser.add_argument(
        '--performance',
        action='store_true',
        help='Run performance benchmarks only'
    )
    
    parser.add_argument(
        '--discover',
        action='store_true',
        help='Discover and list available tests'
    )
    
    # Execution mode options
    parser.add_argument(
        '--mode',
        choices=list(EXECUTION_MODES.keys()),
        default='standard',
        help='Test execution mode (default: standard)'
    )
    
    # Additional options
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        metavar='N',
        help='Run tests in parallel with N workers'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save test results to file'
    )
    
    parser.add_argument(
        '--output-file',
        metavar='FILE',
        help='Output file for results (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.all, args.categories, args.performance, args.discover]):
        args.all = True  # Default to running all tests
    
    # Create and run test runner
    runner = TestRunner(args)
    return runner.run()


if __name__ == '__main__':
    sys.exit(main())
