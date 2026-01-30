#!/usr/bin/env python3
"""
Test script for dares_peft/dares compatibility patch.

This script verifies that the compatibility wrappers work correctly
and can switch between dares_peft and dares based on the OLD_DARES_ARCH
environment variable.
"""

import os
import sys

def test_default_import():
    """Test that default import uses dares_peft"""
    print("Test 1: Default import (should use dares_peft)...")
    
    # Make sure OLD_DARES_ARCH is not set
    if 'OLD_DARES_ARCH' in os.environ:
        del os.environ['OLD_DARES_ARCH']
    
    # Force reload of modules - clear all related modules from cache
    modules_to_clear = [
        'DARES.networks.dares_compat',
        'DARES.networks.dares',
        'DARES.networks.dares_peft',
        'networks.dares_compat',
        'networks.dares',
        'networks.dares_peft'
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    
    try:
        from DARES.networks.dares_compat import DARES
        print("  ✓ Successfully imported DARES from compatibility wrapper")
        
        # Check that it's from dares_peft by inspecting module
        module_name = DARES.__module__
        if 'dares_peft' in module_name:
            print(f"  ✓ Correctly using dares_peft (module: {module_name})")
            return True
        else:
            print(f"  ✗ Expected dares_peft but got: {module_name}")
            return False
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_old_arch_import():
    """Test that OLD_DARES_ARCH=1 switches to dares"""
    print("\nTest 2: Import with OLD_DARES_ARCH=1 (should use dares)...")
    
    # Set OLD_DARES_ARCH
    os.environ['OLD_DARES_ARCH'] = '1'
    
    # Force reload of modules - clear all related modules from cache
    modules_to_clear = [
        'DARES.networks.dares_compat',
        'DARES.networks.dares',
        'DARES.networks.dares_peft',
        'networks.dares_compat',
        'networks.dares',
        'networks.dares_peft'
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    
    try:
        from DARES.networks.dares_compat import DARES
        print("  ✓ Successfully imported DARES from compatibility wrapper")
        
        # Check that it's from dares by inspecting module
        module_name = DARES.__module__
        if 'dares_peft' not in module_name and 'dares' in module_name:
            print(f"  ✓ Correctly using dares (module: {module_name})")
            return True
        else:
            print(f"  ✗ Expected dares but got: {module_name}")
            return False
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False
    finally:
        # Clean up
        if 'OLD_DARES_ARCH' in os.environ:
            del os.environ['OLD_DARES_ARCH']

def test_dares_mh_compat():
    """Test that DARES_MH compatibility wrapper works"""
    print("\nTest 3: DARES_MH compatibility wrapper...")
    
    # Make sure OLD_DARES_ARCH is not set
    if 'OLD_DARES_ARCH' in os.environ:
        del os.environ['OLD_DARES_ARCH']
    
    # Force reload of modules - clear all related modules from cache
    modules_to_clear = [
        'DARES.networks.dares_mh_compat',
        'DARES.networks.dares_peft_MH',
        'networks.dares_mh_compat',
        'networks.dares_peft_MH'
    ]
    for mod in modules_to_clear:
        if mod in sys.modules:
            del sys.modules[mod]
    
    try:
        from DARES.networks.dares_mh_compat import DARES_MH
        print("  ✓ Successfully imported DARES_MH from compatibility wrapper")
        
        # DARES_MH should always come from dares_peft_MH
        module_name = DARES_MH.__module__
        if 'dares_peft_MH' in module_name:
            print(f"  ✓ Correctly using dares_peft_MH (module: {module_name})")
            return True
        else:
            print(f"  ✗ Expected dares_peft_MH but got: {module_name}")
            return False
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

def test_syntax():
    """Test that all compatibility wrappers have valid syntax"""
    print("\nTest 4: Syntax validation...")
    
    import py_compile
    
    files_to_check = [
        'DARES/networks/dares_compat.py',
        'DARES/networks/dares_mh_compat.py'
    ]
    
    all_valid = True
    for file_path in files_to_check:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"  ✓ {file_path} syntax is valid")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {file_path} has syntax errors: {e}")
            all_valid = False
    
    return all_valid

def main():
    print("=" * 60)
    print("Testing dares_peft/dares Compatibility Patch")
    print("=" * 60)
    
    results = []
    
    # Run syntax test first (doesn't require dependencies)
    results.append(("Syntax validation", test_syntax()))
    
    # Try to run import tests (may fail if dependencies not installed)
    try:
        results.append(("Default import", test_default_import()))
        results.append(("OLD_DARES_ARCH import", test_old_arch_import()))
        results.append(("DARES_MH compatibility", test_dares_mh_compat()))
    except ModuleNotFoundError as e:
        print(f"\n⚠ Skipping import tests due to missing dependencies: {e}")
        print("  This is expected if running without a full environment setup.")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())
