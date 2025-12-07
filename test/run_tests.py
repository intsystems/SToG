#!/usr/bin/env python
"""
Cross-platform test runner for Stochastic Gating

Usage: python run_tests.py [command]
"""

import sys
import os
import subprocess
import shutil
from pathlib import Path


class Colors:
    """ANSI color codes for pretty output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  
    
    @classmethod
    def disable(cls):
        """Disable colors for Windows without ANSI support."""
        cls.RED = cls.GREEN = cls.YELLOW = cls.BLUE = cls.NC = ''


if os.name == 'nt' and not os.environ.get('ANSICON'):
    Colors.disable()


def print_header(text):
    """Print header."""
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN} {text}{Colors.NC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW} {text}{Colors.NC}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED} {text}{Colors.NC}")


def run_command(cmd, check=True):
    """Run command with error handling."""
    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        result = subprocess.run(cmd, check=check)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print_error(f"Command failed with error: {e}")
        return False
    except FileNotFoundError:
        print_error(f"Command not found: {cmd[0]}")
        return False


def clean_artifacts():
    """Clean test artifacts."""
    print_header("Cleaning test artifacts...")
    
    paths_to_clean = [
        '.pytest_cache',
        '__pycache__',
    ]
    
    for path in paths_to_clean:
        if Path(path).exists():
            if Path(path).is_dir():
                shutil.rmtree(path)
            else:
                Path(path).unlink()
            print_success(f"Removed: {path}")
    
    
    for pycache in Path('.').rglob('__pycache__'):
        shutil.rmtree(pycache, ignore_errors=True)
    
    
    for pyc in Path('.').rglob('*.pyc'):
        pyc.unlink()
    
    print_success("Artifacts cleaned")


def run_all_tests():
    """Run all tests."""
    print_header("Running all tests...")
    
    cmd = [
        'pytest',
        'test/test_stochastic_gating.py',
        '-v'
    ]
    
    if run_command(cmd):
        print_success("All tests completed!")
    else:
        print_error("Tests completed with errors")
        sys.exit(1)


def run_specific(test_name):
    """Run specific test."""
    print_header(f"Running test: {test_name}")
    
    cmd = ['pytest', test_name, '-v']
    
    if run_command(cmd):
        print_success(f"Test {test_name} completed!")


def show_help():
    """Show help."""
    help_text = """
Stochastic Gating Testing Script

Usage: python run_tests.py [command]

Commands:
  all          - Run all tests (default)
  specific <name> - Run specific test
  clean        - Clean test artifacts
  help         - Show this help
  check        - Check dependencies

Examples:
  python run_tests.py all
  python run_tests.py specific test/test_stochastic_gating.py::TestBasicSelectors
  python run_tests.py clean
"""
    print(help_text)


def check_dependencies():
    """Check installed dependencies."""
    print_header("Checking dependencies...")
    
    required = ['pytest', 'torch', 'sklearn', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
            print_success(f"{pkg} is installed")
        except ImportError:
            print_error(f"{pkg} is NOT installed")
            missing.append(pkg)
    
    if missing:
        print_warning("\nInstall missing packages:")
        print(f"pip install {' '.join(missing)}")
        print("or")
        print("pip install -r requirements.txt")
        sys.exit(1)
    else:
        print_success("\nAll dependencies are installed!")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        command = 'all'
    else:
        command = sys.argv[1].lower()
    
    commands = {
        'all': run_all_tests,
        'clean': clean_artifacts,
        'help': show_help,
        '--help': show_help,
        '-h': show_help,
        'check': check_dependencies,
    }
    
    if command == 'specific':
        if len(sys.argv) < 3:
            print_error("Specify test name")
            print("Usage: python run_tests.py specific <test_name>")
            sys.exit(1)
        run_specific(sys.argv[2])
    elif command in commands:
        commands[command]()
    else:
        print_error(f"Unknown command: {command}")
        print()
        show_help()
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)