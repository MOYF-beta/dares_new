# Compatibility Patch for dares_peft

## Overview

This compatibility patch allows switching between the `dares_peft` (PEFT-based) and `dares` (original) architectures using an environment variable. This is useful for backward compatibility or when you need to use the original architecture for specific use cases.

## Usage

### Default Behavior (Using dares_peft)

By default, all imports will use the `dares_peft` architecture:

```python
from DARES.networks.dares_compat import DARES
# This imports from dares_peft.py
```

### Switching to Old Architecture (Using dares)

To use the original `dares` architecture, set the environment variable `OLD_DARES_ARCH=1`:

```bash
# In bash/shell
export OLD_DARES_ARCH=1
python your_script.py

# Or inline
OLD_DARES_ARCH=1 python your_script.py
```

```python
# In Python (must be set before imports)
import os
os.environ['OLD_DARES_ARCH'] = '1'

from DARES.networks.dares_compat import DARES
# This now imports from dares.py
```

## Implementation Details

### Files Added

1. **`DARES/networks/dares_compat.py`**: Compatibility wrapper for DARES class
   - Checks `OLD_DARES_ARCH` environment variable
   - Imports from `dares_peft.py` by default
   - Imports from `dares.py` when `OLD_DARES_ARCH=1`

2. **`DARES/networks/dares_mh_compat.py`**: Compatibility wrapper for DARES_MH class
   - Always imports from `dares_peft_MH.py` (no old architecture equivalent exists)
   - Provided for consistency in import patterns

### Files Modified

The following files have been updated to use the compatibility wrappers:

1. `exps/attn_encoder_dora/trainer_attn_encoder.py`
2. `exps/no_pose_encoder/trainer_no_pose.py`
3. `exps/DSfm/trainer_dsfm.py`
4. `exps/mh/trainer_mh.py`
5. `exps/trainerMH_abc.py`
6. `exps/load_other_models.py` (respects `OLD_DARES_ARCH` in `load_DARES()` function with its own environment check logic)
7. `simple_3d_reconstruction.py`

### Import Changes

**Before:**
```python
from DARES.networks.dares_peft import DARES
from DARES.networks.dares_peft_MH import DARES_MH
```

**After:**
```python
from DARES.networks.dares_compat import DARES
from DARES.networks.dares_mh_compat import DARES_MH
```

## Testing

A test script is provided to verify the compatibility patch:

```bash
python3 test_compatibility_patch.py
```

This will run syntax validation tests on the compatibility wrappers. Import tests require all dependencies to be installed.

## Notes

- The environment variable must be set **before** importing the modules
- DARES_MH only exists in the PEFT architecture, so it will always use `dares_peft_MH` regardless of the environment variable
- The compatibility wrappers handle both absolute imports (`from DARES.networks...`) and relative imports (`from networks...`)
- Setting `OLD_DARES_ARCH=0` or unsetting it entirely will use the default PEFT architecture
- `exps/load_other_models.py` implements its own environment variable checking logic rather than using the compatibility wrapper, but achieves the same result


## Example

```python
# Example 1: Using default PEFT architecture
from DARES.networks.dares_compat import DARES
model = DARES()  # Uses dares_peft implementation

# Example 2: Using old architecture
import os
os.environ['OLD_DARES_ARCH'] = '1'
# Must reload modules if already imported
import sys
if 'DARES.networks.dares_compat' in sys.modules:
    del sys.modules['DARES.networks.dares_compat']

from DARES.networks.dares_compat import DARES
model = DARES()  # Uses dares implementation
```

## Troubleshooting

**Q: Changes don't take effect after setting the environment variable**

A: Make sure to set the environment variable before importing the modules. If the modules are already imported, you need to reload them or restart your Python process.

**Q: Getting import errors**

A: Make sure all required dependencies are installed. The compatibility wrappers themselves have minimal dependencies, but the underlying `dares` and `dares_peft` modules require various packages.
