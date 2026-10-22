"""Compare four predefined pooling rules using validation predictions only."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.data import frame,aggregate
from src.metrics import evaluate

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--predictions',required=True)
    ap.add_argument('--root',default='raw_data/cxr_first')
    ap.add_argument('--out',default='outputs/aggregation.json')
    args=ap.parse_args()
    p=np.load(args.predictions)['probabilities']
    df=frame(args.root,'valid');scores={}
    for mode in ['mean','max','frontal','weighted']:
        _,probs,y=aggregate(df,p,mode)
        scores[mode]=evaluate(y,probs)
    result={'best':max(scores,key=lambda k:scores[k]['macro_auc']),'validation':scores}
    Path(args.out).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
