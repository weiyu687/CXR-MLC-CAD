"""Check that the finite finisher rejects partial runs and failed queue exits."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import torch
from tools.finish_research_ablation import completed_training, wait_windows_process


def main():
    with tempfile.TemporaryDirectory() as temporary:
        run = Path(temporary)
        config = {'epochs': 6, 'patience': 2}
        (run / 'config.json').write_text(json.dumps(config))
        def save(scores, checkpoint_epoch=None):
            history = [{'epoch': i+1, 'selection_score': value, 'macro_auc': .7}
                       for i, value in enumerate(scores)]
            (run / 'history.jsonl').write_text('\n'.join(map(json.dumps, history)))
            best_epoch = scores.index(max(scores))+1
            torch.save({'epoch': checkpoint_epoch or best_epoch, 'config': config}, run / 'best.pt')
            torch.save({'epoch': len(scores)-1, 'best': max(scores), 'config': config}, run / 'last.pt')
        def rejected():
            try:
                completed_training(run)
            except ValueError:
                return
            raise AssertionError('Partial/corrupt run incorrectly accepted')
        save([.7, .8])
        rejected()
        save([.7, .8, .79, .78])
        record = completed_training(run)
        assert record['selected_epoch'] == 2 and record['epochs_completed'] == 4
        save([.7, .8, .79, .78], checkpoint_epoch=3)
        rejected()
        save([.7, .71, .72, .73, .74, .75])
        assert completed_training(run)['selected_epoch'] == 6
        # Queue-exit handling is tested on disposable children owned here.
        if os.name == 'nt':
            for exit_code in [0, 7]:
                child = subprocess.Popen([sys.executable, '-c',
                    'import sys,time; time.sleep(.5); sys.exit(' + str(exit_code) + ')'])
                try:
                    try:
                        wait_windows_process(child.pid)
                    except RuntimeError:
                        assert exit_code == 7
                    else:
                        assert exit_code == 0
                finally:
                    child.wait(timeout=10)
    print('PASS: partial-run rejection, patience/epoch-limit completion, checkpoint/history binding and queue exit-code handling')


if __name__ == '__main__':
    main()
