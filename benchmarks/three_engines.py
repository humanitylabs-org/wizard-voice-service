#!/usr/bin/env python3
"""Score a frozen audio + same-stem .txt corpus; never silently substitute engines."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
if os.environ.get('WIZARD_BENCH_INSTALLED') != '1':
    sys.path.insert(0, str(ROOT))
ENGINES = ['tinfoil-voxtral', 'parakeet-local', 'voxtral-local']


def words(text):
    return re.findall(r"\w+(?:'\w+)*", text.lower().replace('’', "'"), flags=re.UNICODE)


def score(reference, hypothesis):
    """Exact Levenshtein alignment, deterministic substitution > deletion > insertion ties."""
    a, b = words(reference), words(hypothesis)
    rows = [[(0, '')] * (len(b)+1) for _ in range(len(a)+1)]
    for i in range(1, len(a)+1): rows[i][0] = (i, 'D')
    for j in range(1, len(b)+1): rows[0][j] = (j, 'I')
    for i, left in enumerate(a, 1):
        for j, right in enumerate(b, 1):
            if left == right: rows[i][j] = (rows[i-1][j-1][0], '=')
            else:
                choices = [(rows[i-1][j-1][0]+1, 'S'), (rows[i-1][j][0]+1, 'D'), (rows[i][j-1][0]+1, 'I')]
                rows[i][j] = min(choices, key=lambda item: item[0])
    i, j = len(a), len(b)
    operations = []
    while i or j:
        operation = rows[i][j][1]
        operations.append((operation, a[i-1] if i else None, b[j-1] if j else None))
        if operation in {'=', 'S', 'D'}: i -= 1
        if operation in {'=', 'S', 'I'}: j -= 1
    operations.reverse()
    counts = {key: sum(op == key for op, _, _ in operations) for key in ('S', 'D', 'I')}
    ending = a[-min(12, len(a)):]
    ending_ok = b[-len(ending):] == ending if ending else not b
    trailing_deleted = 0
    for op, _, _ in reversed(operations):
        if op == 'I': continue
        if op != 'D': break
        trailing_deleted += 1
    return dict(reference_words=len(a), hypothesis_words=len(b), substitutions=counts['S'],
        deletions=counts['D'], insertions=counts['I'],
        wer=(sum(counts.values())/len(a)) if a else None,
        deleted_words=[left for op, left, _ in operations if op == 'D'],
        ending_last12_exact=ending_ok, omitted_trailing_words=trailing_deleted,
        hallucinated_words_on_empty_reference=len(b) if not a else 0)


def digest(path):
    with path.open('rb') as source: return hashlib.file_digest(source, 'sha256').hexdigest()


def freeze(folder):
    extensions = {'.wav', '.mp3', '.m4a', '.flac', '.aac', '.ogg', '.webm', '.aiff'}
    result = []
    for audio in sorted(folder.iterdir()):
        if audio.suffix.lower() not in extensions: continue
        reference = audio.with_suffix('.txt')
        if not reference.is_file(): raise ValueError(f'missing same-stem reference for {audio.name}')
        result.append({'id': audio.name, 'audio_sha256': digest(audio), 'reference_sha256': digest(reference)})
    if not result: raise ValueError('no audio + reference pairs')
    return result


def aggregate(output):
    # Read persisted attempts, never count a guessed in-memory subset.
    rows = [json.loads(line) for line in (output/'attempts.jsonl').read_text().splitlines() if line]
    manifest = json.loads((output/'corpus.json').read_text())
    expected = {item['id'] for item in manifest}
    totals = {}
    for engine in ENGINES:
        lane = [r for r in rows if r['engine'] == engine]
        if not lane: continue
        if len({r['file'] for r in lane}) != len(lane): raise ValueError('duplicate engine/file attempts')
        if {r['file'] for r in lane} != expected: raise ValueError('incomplete lane; retain partial attempts, do not claim completion')
        good = [r for r in lane if r['status'] == 'ok']
        n = sum(r['score']['reference_words'] for r in good)
        errors = sum(r['score'][key] for r in good for key in ['substitutions', 'deletions', 'insertions'])
        speech = [r for r in good if r['score']['reference_words']]
        totals[engine] = {'attempted': len(lane), 'successes': len(good), 'failures': len(lane)-len(good),
            'successful_reference_words': n, 'micro_wer_successful_only': errors/n if n else None,
            'speech_micro_wer_successful_only': sum(r['score'][k] for r in speech for k in ['substitutions','deletions','insertions'])/n if n else None,
            'ending_failures': sum(not r['score']['ending_last12_exact'] for r in speech),
            'omitted_words': sum(r['score']['deletions'] for r in speech),
            'omitted_trailing_words': sum(r['score']['omitted_trailing_words'] for r in speech),
            'hallucinated_silent_words': sum(r['score']['hallucinated_words_on_empty_reference'] for r in good),
            'median_latency_seconds': statistics.median(r['elapsed_seconds'] for r in good) if good else None,
            'aggregate_rtf': sum(r['elapsed_seconds'] for r in good)/sum(r['audio_seconds'] for r in good) if good else None,
            'peak_engine_rss_bytes': max((r.get('engine_health_after',{}).get('peak_rss_bytes',0) for r in good), default=0) or None}
    payload = {'reference_contract': 'WER vs supplied reference; generated fixtures use intended macOS say script, not a human-verified acoustic reference',
        'timing_boundary': 'full file via isolated candidate pause30 pipeline; hosted remote resources are unavailable',
        'silence_contract': 'raw engines see silence; no VAD masking; failed empty hosted readings remain failures, not clean silence',
        'corpus': manifest, 'summary': totals, 'attempts': rows}
    (output/'results.json').write_text(json.dumps(payload, indent=2)+'\n')
    lines = ['# Three-engine batch comparison', '', payload['reference_contract'], '',
        '| Engine | Success / attempted | Speech WER* | Omitted words | Ending failures | Silent words | Median seconds | RTF | Peak RSS GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt = lambda n: f'{n:.3f}' if n is not None else 'unmeasured'
    for engine, item in totals.items():
        lines.append(f"| {engine} | {item['successes']}/{item['attempted']} | {fmt(item['speech_micro_wer_successful_only'])} | {item['omitted_words']} | {item['ending_failures']} | {item['hallucinated_silent_words']} | {fmt(item['median_latency_seconds'])} | {fmt(item['aggregate_rtf'])} | {fmt(item['peak_engine_rss_bytes']/2**30 if item['peak_engine_rss_bytes'] else None)} |")
    lines += ['', '*Successful speech files only; inspect failures before ranking. Silence errors are reported separately.', '',
              '| Engine | File | Status | Seconds | RTF | WER | Deleted | Ending exact |', '|---|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        metric = r.get('score', {})
        lines.append(f"| {r['engine']} | {r['file']} | {r['status']} | {r['elapsed_seconds']:.3f} | {r.get('rtf',0):.3f} | {fmt(metric.get('wer'))} | {metric.get('deletions','—')} | {metric.get('ending_last12_exact','—')} |")
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return totals


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--corpus', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engines', default=','.join(ENGINES))
    args = p.parse_args(argv)
    chosen = args.engines.split(',')
    if not chosen or set(chosen)-set(ENGINES) or len(set(chosen)) != len(chosen): p.error('unknown or repeated engine')
    import yaml
    from config import load_provider_config
    from batch_audio import decode
    from providers import NoSpeechError, load_api_key, transcribe, _open_transcription
    import urllib.request
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = freeze(args.corpus)
    frozen = args.output/'corpus.json'
    if frozen.exists() and json.loads(frozen.read_text()) != manifest: raise ValueError('corpus changed since freeze')
    frozen.write_text(json.dumps(manifest, indent=2)+'\n')
    path = args.output/'attempts.jsonl'
    prior = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    root = yaml.safe_load(args.config.read_text())
    for engine in chosen:
        if any(r['engine']==engine for r in prior): raise ValueError('lane already attempted; use a new output directory for retries')
        root['stt']['active_profile'] = engine
        with tempfile.TemporaryDirectory(prefix='voice-benchmark-') as temporary:
            cfg = Path(temporary)/'config.yaml'
            cfg.write_text(yaml.safe_dump(root))
            selected = load_provider_config(cfg)
        selected = replace(selected, postprocess={})  # Acoustic comparison, no personal correction/rules.
        for item in manifest:
            audio = args.corpus/item['id']
            reference = audio.with_suffix('.txt').read_text()
            duration = len(decode(audio))/16000
            row = {'engine': engine, 'file': item['id'], 'audio_sha256': item['audio_sha256'],
                   'model': selected.model, 'audio_seconds': duration, 'fallback': None}
            def health():
                if selected.name != 'local-engine': return {'remote_resources': 'unavailable'}
                url = selected.endpoint.rsplit('/v1/',1)[0]+'/health'
                with _open_transcription(urllib.request.Request(url), timeout=3) as response:
                    return json.load(response)
            start = time.monotonic()
            try:
                row['engine_health_before'] = health()
                key = '' if selected.name == 'local-engine' else load_api_key(selected.api_key_env)
                request_start = time.monotonic()
                try: result = transcribe(audio, selected, api_key=key); text = result.text
                except NoSpeechError: text = ''; row['no_speech'] = True
                row['request_seconds'] = time.monotonic()-request_start
                row.update(status='ok', text=text, score=score(reference, text), engine_health_after=health())
                if selected.name == 'local-engine':
                    row['engine_cpu_seconds'] = row['engine_health_after']['cpu_seconds']-row['engine_health_before']['cpu_seconds']
                    row['engine_average_cpu_cores'] = row['engine_cpu_seconds']/row['request_seconds']
            except Exception as exc:
                row.update(status='failed', error_type=type(exc).__name__, error=str(exc)[:200])
            row['harness_total_seconds'] = time.monotonic()-start
            row['elapsed_seconds'] = row.get('request_seconds', row['harness_total_seconds'])
            row['rtf'] = row['elapsed_seconds']/duration
            with path.open('a') as out:
                os.chmod(path, 0o600)
                out.write(json.dumps(row)+'\n'); out.flush(); os.fsync(out.fileno())
            print(json.dumps({k:row[k] for k in ['engine','file','status','elapsed_seconds']}), flush=True)
    totals = aggregate(args.output)
    print(json.dumps(totals, indent=2))
    return 0 if all(totals[e]['failures']==0 for e in chosen) else 1


if __name__ == '__main__': raise SystemExit(main())
