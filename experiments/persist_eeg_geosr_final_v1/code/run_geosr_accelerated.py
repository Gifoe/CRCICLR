"""Original full protocol with an explicit CUDA index and intact cache hashes.

Same arguments as run_geosr.py. This launcher does not authorize extending a
RAPID_TRIAGE STOP; use it only after the locked continuation gate passes.
"""
import sys
import torch
import run_geosr


if __name__ == '__main__':
    if '--device' not in sys.argv and torch.cuda.is_available():
        sys.argv.extend(['--device', f'cuda:{torch.cuda.current_device()}'])
    elif '--device' in sys.argv:
        i = sys.argv.index('--device') + 1
        if i < len(sys.argv) and sys.argv[i] == 'cuda':
            sys.argv[i] = f'cuda:{torch.cuda.current_device()}'
    run_geosr.main()
