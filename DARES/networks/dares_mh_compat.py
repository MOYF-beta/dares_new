"""
Compatibility wrapper for dares_peft_MH module.

When environment variable OLD_DARES_ARCH=1 is set, this module will
still use dares_peft_MH since there is no old architecture equivalent.
This ensures the code doesn't break when switching architectures.
"""
import os

# DARES_MH only exists in the PEFT version, so we always use it
# regardless of the OLD_DARES_ARCH setting
try:
    from DARES.networks.dares_peft_MH import DARES_MH
except ImportError:
    from networks.dares_peft_MH import DARES_MH

__all__ = ['DARES_MH']
