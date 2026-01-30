"""
Compatibility wrapper for dares_peft and dares modules.

When environment variable OLD_DARES_ARCH=1 is set, this module will
import from dares.py instead of dares_peft.py for backward compatibility.
"""
import os

# Check environment variable
USE_OLD_ARCH = os.environ.get('OLD_DARES_ARCH', '0') == '1'

if USE_OLD_ARCH:
    # Use the old dares architecture
    try:
        from DARES.networks.dares import DARES
    except ImportError:
        from networks.dares import DARES
else:
    # Use the new dares_peft architecture (default)
    try:
        from DARES.networks.dares_peft import DARES
    except ImportError:
        from networks.dares_peft import DARES

__all__ = ['DARES']
