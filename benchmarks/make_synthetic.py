#!/usr/bin/env python3
"""Generate only non-private scripted speech on macOS; retain scripts and hashes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from batch_audio import decode, write_wav


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False,mode=0o700)
    short='Do not delete the blue folder. Keep the green notebook. Repeat repeat the final word sunflower.'
    singles={'short':short,'yes':'Yes.','stop':'Stop.'}
    sections=['first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth',
        'eleventh','twelfth','thirteenth','fourteenth','fifteenth','sixteenth','seventeenth','eighteenth',
        'nineteenth','twentieth','twenty first','twenty second','twenty third','twenty fourth',
        'twenty fifth','twenty sixth','twenty seventh','twenty eighth','twenty ninth','thirtieth']
    paragraphs=[f'This is the {s} section of the test recording. The blue notebook is beside the window. '
        'Please keep the green folder and do not delete it. We will review the quiet garden after lunch. '
        'The next sentence must remain in the transcript.' for s in sections]
    singles['long']=' '.join(paragraphs)+' This is the final sentence. Keep every word through the ending purple lantern sunflower.'
    for name,text in singles.items():
        script=a.output/(name+'.txt');script.write_text(text+'\n')
        raw=a.output/(name+'.aiff')
        subprocess.run(['say','-v','Samantha','-r','165','-f',str(script),'-o',str(raw)],check=True,timeout=120)
        samples=decode(raw)
        write_wav(a.output/(name+'.wav'),samples)
        raw.unlink()
        if name=='long' and len(samples)/16000 <360: raise RuntimeError('long speech failed six-minute gate')
    x=decode(a.output/'short.wav')
    rng=np.random.default_rng(42)
    derived={'quiet-speech':x*10**(-28/20),'delayed-speech':np.concatenate([np.zeros(3*16000),x]),
        'noise-mixed':x+rng.normal(0,10**(-32/20),len(x))}
    for name,samples in derived.items():
        write_wav(a.output/(name+'.wav'),samples);(a.output/(name+'.txt')).write_text(short+'\n')
    for name,samples in {'silence':np.zeros(4*16000),'near-silence':rng.normal(0,0.00002,6*16000),
                          'noise-only':rng.normal(0,0.004,4*16000)}.items():
        write_wav(a.output/(name+'.wav'),samples);(a.output/(name+'.txt')).write_text('')
    from three_engines import freeze
    (a.output/'manifest.json').write_text(json.dumps({'origin':'macOS say Samantha, rate165; derived silence/noise seed42',
        'reference_kind':'intended TTS script, not human-verified acoustic truth','files':freeze(a.output)},indent=2)+'\n')


if __name__=='__main__':main()
