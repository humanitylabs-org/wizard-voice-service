"""Context preservation is an audio policy, never a phrase correction."""
from pathlib import Path
from unittest.mock import Mock
import urllib.error

import pytest
import providers as p

CONFIG = p.ProviderConfig('tinfoil', 'voxtral-small-24b',
    'http://127.0.0.1:3301/v1/audio/transcriptions', 'TINFOIL_API_KEY', True)


def test_whole_note_ignores_legacy_thirty_second_boundary(monkeypatch, tmp_path):
    normalize = Mock(side_effect=lambda source, out, **kw: out.write_bytes(b'audio'))
    monkeypatch.setattr(p, 'normalize_audio', normalize)
    monkeypatch.setattr(p, '_audio_duration', lambda path: 99.0)
    monkeypatch.setattr(p, '_segment_audio', Mock(side_effect=AssertionError('must remain whole')))
    chunks = p.normalize_audio_chunks(tmp_path/'source', tmp_path, chunk_seconds=30)
    assert len(chunks) == 1
    normalize.assert_called_once()


def test_long_note_uses_bounded_context_segments(monkeypatch, tmp_path):
    monkeypatch.setattr(p, 'normalize_audio', lambda source, out, **kw: out.write_bytes(b'audio'))
    monkeypatch.setattr(p, '_audio_duration', lambda path: 343.0)
    split = Mock(return_value=[tmp_path/'one',tmp_path/'two'])
    monkeypatch.setattr(p, '_segment_audio', split)
    assert len(p.normalize_audio_chunks(tmp_path/'source', tmp_path, chunk_seconds=720)) == 2
    assert split.call_args.kwargs['target_seconds'] == 45


def test_overlap_is_removed_once_without_touching_repetitions():
    assert p._join_overlap('Wait wait. Look at the old poster.',
                          'Look at the old poster, then wait wait.') == 'Wait wait. Look at the old poster. then wait wait.'


@pytest.mark.parametrize('left,right', [
    ('Use the old poster', 'Use the bold poster today'),
    ('Go go go', 'go go go again'),
    ('Say the same thing. Say the same thing.', 'Say the same thing. Continue.'),
    ('End here.', 'Unrelated start.'),
])
def test_conflicting_or_ambiguous_overlap_is_recoverable_not_guessed(left,right):
    with pytest.raises(p.ProviderError, match='transcript_seam_unresolved'):
        p._join_overlap(left,right)


def test_unicode_and_punctuation_do_not_invent_words():
    assert p._join_overlap('Then Café déjà vu!', 'café déjà vu, again.') == 'Then Café déjà vu! again.'


def test_timeout_splits_with_context_and_same_model(monkeypatch, tmp_path):
    whole, left, right = [tmp_path/n for n in ('whole','left','right')]
    monkeypatch.setattr(p, '_audio_duration', lambda path: 99)
    monkeypatch.setattr(p, '_split_segment', lambda path: [left,right])
    monkeypatch.setattr(p, 'detect_no_speech', lambda path: False)
    seen=[]
    def request(audio,cfg,key,prompt=''):
        seen.append((audio,cfg.model,cfg.timeout))
        if audio == whole:
            raise p.ProviderError('tinfoil route unavailable (TimeoutError)')
        return 'Before we look at the old poster' if audio == left else 'look at the old poster and continue'
    monkeypatch.setattr(p,'_transcribe_chunk',request)
    assert p._transcribe_segment(whole,CONFIG,'test') == 'Before we look at the old poster and continue'
    assert len(seen)==3 and all(model==CONFIG.model and timeout<=45 for _,model,timeout in seen)


def test_retry_depth_is_bounded(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'_audio_duration',lambda path:99)
    monkeypatch.setattr(p,'_split_segment',lambda path:[path.with_name(path.name+'x'),path.with_name(path.name+'y')])
    monkeypatch.setattr(p,'detect_no_speech',lambda path:False)
    call=Mock(side_effect=p.ProviderError('tinfoil route unavailable (TimeoutError)'))
    monkeypatch.setattr(p,'_transcribe_chunk',call)
    with pytest.raises(p.ProviderError,match='TimeoutError'):
        p._transcribe_segment(tmp_path/'whole',CONFIG,'test')
    assert call.call_count==3


def test_wrapped_socket_timeout_is_recognized():
    exc=p.ProviderError('tinfoil route unavailable (URLError)')
    exc.__cause__=urllib.error.URLError(TimeoutError('timed out'))
    assert p._is_timeout(exc)


def test_no_partial_success_when_later_segment_fails(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'normalize_audio_chunks',lambda *a:[tmp_path/'left',tmp_path/'right'])
    monkeypatch.setattr(p,'detect_no_speech',lambda path:False)
    def request(audio,*args):
        if audio.name=='right': raise p.ProviderError('HTTP 503')
        return 'A valid first reading'
    monkeypatch.setattr(p,'_transcribe_chunk',request)
    with pytest.raises(p.ProviderError,match='chunk 2/2 failed'):
        p.transcribe(tmp_path/'original',CONFIG,api_key='test')


def test_expired_request_budget_never_calls_provider(monkeypatch,tmp_path):
    call=Mock()
    monkeypatch.setattr(p,'_transcribe_chunk',call)
    with pytest.raises(p.ProviderError,match='deadline'):
        p._transcribe_segment(tmp_path/'original',CONFIG,'test',deadline=0)
    call.assert_not_called()


def test_pause_plan_covers_all_audio_and_overlaps():
    spans=p._segment_spans(99,[28,61],30)
    assert spans[0][0]==0 and spans[-1][1]==99
    assert spans[0][1]==32 and spans[1][0]==24
    assert all(a[1]>b[0] for a,b in zip(spans,spans[1:]))
    assert all(0<=a<b<=99 for a,b in spans)


def test_timeout_split_actually_uses_pause_and_overlap(monkeypatch,tmp_path):
    monkeypatch.setattr(p,'_audio_duration',lambda path:30)
    split=Mock(return_value=[tmp_path/'left',tmp_path/'right'])
    monkeypatch.setattr(p,'_segment_audio',split)
    p._split_segment(tmp_path/'input')
    assert split.call_args.kwargs['target_seconds']==15


def test_verified_speech_marker_survives_overlap():
    text=p._join_overlap(p.VerifiedSpeech('Start with the spoken example'), 'with the spoken example and finish')
    assert isinstance(text,p.VerifiedSpeech)


def test_three_overlaps_keep_entire_middle_and_repeated_nonseam_words():
    texts=['Wait wait. Take the old path', 'take the old path then cross the blue bridge',
           'cross the blue bridge and wait wait.']
    assert p._join_segments(texts)=='Wait wait. Take the old path then cross the blue bridge and wait wait.'


def test_conflict_rereads_audio_not_suggested_words(monkeypatch,tmp_path):
    chunks=p.ContextSegments([tmp_path/'a',tmp_path/'b'],tmp_path/'whole',[(0,34),(26,60)])
    render=Mock(); request=Mock(return_value='This is the complete fresh reading.')
    monkeypatch.setattr(p,'_render_span',render)
    monkeypatch.setattr(p,'_transcribe_chunk',request)
    result=p._reconcile_segments(['a disputed tail','a different head'],chunks,CONFIG,'test','',float('inf'))
    assert result=='This is the complete fresh reading.'
    assert render.call_args.args[1:3]==(0,60)
    assert request.call_args.args[1].model==CONFIG.model
    assert request.call_args.args[3]==''


def test_join_reread_still_must_agree_with_next_neighbour(monkeypatch,tmp_path):
    chunks=p.ContextSegments([tmp_path/n for n in 'abc'],tmp_path/'whole',[(0,34),(26,64),(56,90)])
    request=Mock(side_effect=['Fresh first reading shares the blue bridge',
                              'Entire audio freshly reread'])
    monkeypatch.setattr(p,'_render_span',Mock())
    monkeypatch.setattr(p,'_transcribe_chunk',request)
    result=p._reconcile_segments(['conflicting one','conflicting two','different next words'],
                                 chunks,CONFIG,'test','',float('inf'))
    assert result=='Entire audio freshly reread'
    assert request.call_count==2


def test_join_window_cannot_expand_without_bound(monkeypatch,tmp_path):
    chunks=p.ContextSegments([tmp_path/'a',tmp_path/'b'],tmp_path/'whole',[(0,90),(82,172)])
    request=Mock()
    monkeypatch.setattr(p,'_transcribe_chunk',request)
    with pytest.raises(p.ProviderError,match='safe limit'):
        p._reconcile_segments(['disputed left','disputed right'],chunks,CONFIG,'test','',float('inf'))
    request.assert_not_called()


def test_join_reread_failure_does_not_return_partial_transcript(monkeypatch,tmp_path):
    chunks=p.ContextSegments([tmp_path/'a',tmp_path/'b'],tmp_path/'whole',[(0,34),(26,60)])
    monkeypatch.setattr(p,'_render_span',Mock())
    monkeypatch.setattr(p,'_transcribe_chunk',Mock(side_effect=p.ProviderError('stt_refusal')))
    with pytest.raises(p.ProviderError,match='stt_refusal'):
        p._reconcile_segments(['disputed left','disputed right'],chunks,CONFIG,'test','',float('inf'))


def test_join_reread_count_is_bounded(monkeypatch,tmp_path):
    chunks=p.ContextSegments([tmp_path/str(i) for i in range(6)],tmp_path/'whole',
                             [(i*10,i*10+14) for i in range(6)])
    request=Mock(return_value='Still unrelated words')
    monkeypatch.setattr(p,'_render_span',Mock())
    monkeypatch.setattr(p,'_transcribe_chunk',request)
    with pytest.raises(p.ProviderError,match='transcript_seam_unresolved'):
        p._reconcile_segments([f'part {i}' for i in range(6)],chunks,CONFIG,'test','',float('inf'))
    assert request.call_count==4


def test_unresolved_seam_job_retains_original_without_partial_success(tmp_path):
    import hashlib
    from jobs import VoiceJobStore, VoiceJobWorker
    source=tmp_path/'original.wav'
    source.write_bytes(b'synthetic original audio, unchanged by provider')
    store=VoiceJobStore(tmp_path/'state')
    admission=store.accept(source,source_app='test',device_id='test',idempotency_key='one')
    worker=VoiceJobWorker(store,transcriber=Mock(side_effect=p.ProviderError('transcript_seam_unresolved')))
    assert worker.process_one()
    record=store.get(admission.job_id)
    assert record is not None
    assert record['state']=='failed_retryable'
    assert record['transcript'] is None
    assert record['error']=='transcript_seam_unresolved'
    retained=list(store.audio_dir.iterdir())
    assert len(retained)==1 and retained[0].read_bytes()==source.read_bytes()
    assert record['audio_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert retained[0].stat().st_mode & 0o777 == 0o600


