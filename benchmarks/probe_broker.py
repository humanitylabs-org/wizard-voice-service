#!/usr/bin/env python3
"""Real HTTP probes against an already-started isolated candidate broker."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import httpx


def main():
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expect-unavailable',action='store_true');a=p.parse_args()
    from three_engines import score
    receipt={'url':a.url,'probes':[]}
    with httpx.Client(trust_env=False,timeout=900) as client:
        health=client.get(a.url+'/health');health.raise_for_status();receipt['health']=health.json()
        for filename in (['short.wav'] if a.expect_unavailable else ['short.wav','long.wav']):
            start=time.monotonic()
            with (a.corpus/filename).open('rb') as audio:
                response=client.post(a.url+'/v1/audio/transcriptions',files={'file':(filename,audio,'audio/wav')},data={'language':'en','mode':'verbatim'})
            row={'file':filename,'status_code':response.status_code,'elapsed_seconds':time.monotonic()-start}
            receipt['probes'].append(row)
            if a.expect_unavailable:
                if response.status_code!=502: raise AssertionError('engine outage did not fail closed with 502')
                row['no_partial_text']=not response.json().get('text')
                if not row['no_partial_text']:raise AssertionError('outage returned partial text')
            else:
                response.raise_for_status();result=response.json();row['response']=result
                row['score']=score((a.corpus/filename).with_suffix('.txt').read_text(),result['text'])
                if result.get('model')!=receipt['health']['model']:raise AssertionError('broker model identity changed')
        a.output.write_text(json.dumps(receipt,indent=2)+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
