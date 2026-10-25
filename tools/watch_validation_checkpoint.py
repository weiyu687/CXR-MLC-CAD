"""Preserve official-valid peaks from an already-running older training process.

Read-only with respect to the training checkpoint/history. Only writes the
separate official_valid_selected/ candidate. A process handle is observed,
never signalled. No image inference or GPU computation is performed here.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time
import torch
from src.training import save_validation_candidate


def preserve_available(run):
    run = Path(run)
    history = [json.loads(line) for line in (run/'history.jsonl').read_text(encoding='utf-8').splitlines()]
    if not history:
        return
    previous = run/'official_valid_selected'/'metrics.json'
    retained = json.loads(previous.read_text(encoding='utf-8'))['macro_auc'] if previous.exists() else -1
    for filename, resume_state in [('best.pt', False), ('last.pt', True)]:
        if not (run/filename).exists():
            continue
        # Training writes history before the resume file. Wait for the correct
        # epoch instead of associating new metrics with an old model state.
        if max(record['macro_auc'] for record in history) <= retained:
            return
        checkpoint = torch.load(run/filename, map_location='cpu', weights_only=False)
        epoch = checkpoint['epoch'] + int(resume_state)
        record = next((record for record in history if record['epoch'] == epoch), None)
        if record is not None and record['macro_auc'] > retained:
            if save_validation_candidate(run, checkpoint, record, resume_state):
                retained = record['macro_auc']
                print(json.dumps({'epoch': epoch, 'macro_auc': retained, 'source': filename}), flush=True)
        del checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--process', type=int, required=True)
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('This process observer is for the current Windows training host')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000, False, args.process)
    if not handle:
        raise OSError(ctypes.get_last_error(), 'Cannot observe requested training PID')
    try:
        while True:
            terminal = kernel.WaitForSingleObject(handle, 0)
            if terminal not in [0, 258]:
                raise OSError(ctypes.get_last_error(), 'Cannot poll training process')
            try:
                preserve_available(args.run)
            except (OSError, EOFError, RuntimeError, json.JSONDecodeError) as error:
                if terminal == 0:
                    raise
                print('RETRY incomplete training write: ' + type(error).__name__, flush=True)
            if terminal == 0:
                break
            time.sleep(1)
    finally:
        kernel.CloseHandle(handle)
    print('VALIDATION_OBSERVER_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
