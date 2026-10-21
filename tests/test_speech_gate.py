import io
import wave
from unittest.mock import Mock
import numpy as np
import pytest
from fastapi.testclient import TestClient
import speech_gate as gate
import providers
import server
from config import ProviderConfig

def wav_bytes(seconds=1):
    out=io.BytesIO()
    with wave.open(out,'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000)
        w.writeframes(b'\0\0'*int(seconds*16000))
    return out.getvalue()

@pytest.mark.parametrize('seconds',[1,5,15])
def test_real_detector_digital_silence(seconds):
    assert gate.pcm_is_no_speech(b'\0\0'*seconds*16000)

def test_short_or_truncated_audio_is_not_discarded():
    # Short non-silent audio (a cut-off syllable) still goes to the ASR.
    assert gate.pcm_is_no_speech(np.full(1000, 300, dtype='<i2').tobytes()) is False
    assert gate.pcm_is_no_speech(b'') is False
    with pytest.raises(gate.SpeechGateError):gate.pcm_is_no_speech(b'x')

def test_short_digital_silence_tail_is_no_speech():
    # Live failure: a 30.66 s note left a 0.72 s all-zero second segment; the
    # ASR returned empty text and every retry failed "chunk 2/2".
    assert gate.pcm_is_no_speech(b'\0\0'*1000) is True
    assert gate.pcm_is_no_speech(b'\0\0'*7185) is True

@pytest.mark.parametrize('probabilities,expected',[
    ([0.01,0.7,0.01],False),([0.01,0.4,0.4],True),([0.34]*20,False),
])
def test_consecutive_evidence_preserves_short_words(monkeypatch,probabilities,expected):
    monkeypatch.setattr(gate,'speech_probabilities',lambda audio:iter(probabilities))
    assert gate.has_possible_speech(np.zeros(16000,dtype=np.float32)) is expected

def test_gain_normalized_second_pass_preserves_soft_speech(monkeypatch):
    peaks=[]
    def possible(audio):
        peaks.append(float(np.max(np.abs(audio))))
        return peaks[-1] > 0.1
    monkeypatch.setattr(gate,'has_possible_speech',possible)
    assert gate.pcm_is_no_speech(np.full(16000,10,dtype='<i2').tobytes()) is False
    assert len(peaks)==2 and peaks[1]>peaks[0]

def test_corrupt_audio_fails_without_becoming_no_speech(tmp_path):
    p=tmp_path/'invalid.wav';p.write_bytes(b'invalid')
    with pytest.raises(gate.SpeechGateError):gate.detect_no_speech(p)

def configure(monkeypatch):
    config=ProviderConfig(name='tinfoil',model='voxtral-small-24b',endpoint='http://127.0.0.1:3301/v1/audio/transcriptions',api_key_env='TEST_KEY',attested=True)
    monkeypatch.setattr(providers,'load_provider_config',lambda:config)
    monkeypatch.setattr(providers,'load_api_key',lambda _: 'test')
    return config

@pytest.mark.parametrize('encoding',['wav','m4a'])
def test_real_api_silence_has_no_text_and_never_calls_provider(monkeypatch,tmp_path,encoding):
    import subprocess
    configure(monkeypatch);call=Mock(side_effect=AssertionError('provider must not be called'))
    monkeypatch.setattr(providers,'_transcribe_chunk',call)
    audio=tmp_path/'zero.wav';audio.write_bytes(wav_bytes())
    if encoding=='m4a':
        encoded=tmp_path/'zero.m4a'
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(audio),'-c:a','aac',str(encoded)],check=True)
        audio=encoded
    response=TestClient(server.app).post('/v1/audio/transcriptions',files={'file':(audio.name,audio.read_bytes())})
    assert response.status_code==422
    assert response.json()=={'detail':'No speech detected'}
    call.assert_not_called()

def test_silent_part_of_long_audio_does_not_discard_speech(monkeypatch,tmp_path):
    config=configure(monkeypatch)
    monkeypatch.setattr(providers,'normalize_audio_chunks',lambda *a:[tmp_path/'one',tmp_path/'two'])
    monkeypatch.setattr(providers,'detect_no_speech',lambda p:p.name=='one')
    call=Mock(return_value='Actual spoken sentence.')
    monkeypatch.setattr(providers,'_transcribe_chunk',call)
    result=providers.transcribe(tmp_path/'source',config,api_key='test')
    assert result.text=='Actual spoken sentence.'
    assert result.model=='voxtral-small-24b' and call.call_count==1

def test_gate_failure_does_not_call_provider_or_report_no_speech(monkeypatch,tmp_path):
    config=configure(monkeypatch)
    monkeypatch.setattr(providers,'normalize_audio_chunks',lambda *a:[tmp_path/'one'])
    monkeypatch.setattr(providers,'detect_no_speech',Mock(side_effect=gate.SpeechGateError('unavailable')))
    call=Mock();monkeypatch.setattr(providers,'_transcribe_chunk',call)
    with pytest.raises(providers.ProviderError,match='unavailable') as error:
        providers.transcribe(tmp_path/'source',config,api_key='test')
    assert not isinstance(error.value,providers.NoSpeechError)
    call.assert_not_called()
