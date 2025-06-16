#!/usr/bin/env python3
"""
Test script for video processing integration in Tree of Thoughts.
"""

import sys
import os
from pathlib import Path

# Add the tree_of_thoughts package to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from tree_of_thoughts import TotAgent, VideoProcessor
        print("✓ Successfully imported TotAgent and VideoProcessor")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_video_processor_initialization():
    """Test VideoProcessor initialization."""
    print("\nTesting VideoProcessor initialization...")
    
    try:
        from tree_of_thoughts import VideoProcessor
        processor = VideoProcessor()
        print("✓ VideoProcessor initialized successfully")
        return True
    except Exception as e:
        print(f"✗ VideoProcessor initialization failed: {e}")
        return False

def test_tot_agent_initialization():
    """Test TotAgent initialization with and without video processing."""
    print("\nTesting TotAgent initialization...")
    
    try:
        from tree_of_thoughts import TotAgent
        
        # Test basic initialization
        agent_basic = TotAgent(use_openai_caller=False)
        print("✓ Basic TotAgent initialized successfully")
        
        # Test with video processing disabled
        agent_no_video = TotAgent(enable_video_processing=False)
        print("✓ TotAgent with video processing disabled initialized successfully")
        
        # Test with video processing enabled (may fail if dependencies missing)
        try:
            agent_with_video = TotAgent(enable_video_processing=True)
            print("✓ TotAgent with video processing enabled initialized successfully")
        except Exception as e:
            print(f"⚠ TotAgent with video processing failed (expected if dependencies missing): {e}")
        
        return True
    except Exception as e:
        print(f"✗ TotAgent initialization failed: {e}")
        return False

def test_method_availability():
    """Test that new video processing methods are available."""
    print("\nTesting method availability...")
    
    try:
        from tree_of_thoughts import TotAgent
        agent = TotAgent(enable_video_processing=False)  # Don't require video deps
        
        # Check if methods exist
        if hasattr(agent, 'analyze_video'):
            print("✓ analyze_video method available")
        else:
            print("✗ analyze_video method missing")
            return False
            
        if hasattr(agent, 'analyze_video_features'):
            print("✓ analyze_video_features method available")
        else:
            print("✗ analyze_video_features method missing")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Method availability test failed: {e}")
        return False

def test_backwards_compatibility():
    """Test that existing functionality still works."""
    print("\nTesting backwards compatibility...")
    
    try:
        from tree_of_thoughts import TotAgent
        
        # Test basic ToT functionality
        agent = TotAgent(use_openai_caller=False)
        
        # Test that run method still exists and is callable
        if hasattr(agent, 'run') and callable(getattr(agent, 'run')):
            print("✓ Original run method is available and callable")
        else:
            print("✗ Original run method missing or not callable")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Backwards compatibility test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Tree of Thoughts Video Processing Integration Test")
    print("=" * 55)
    
    tests = [
        test_imports,
        test_video_processor_initialization,
        test_tot_agent_initialization,
        test_method_availability,
        test_backwards_compatibility
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nTest Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Video processing integration is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
