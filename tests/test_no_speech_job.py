from unittest.mock import Mock
from jobs import VoiceJobWorker
from providers import NoSpeechError

def test_no_speech_job_is_terminal_without_three_identical_retries():
    store=Mock()
    store.claim_for_processing.return_value={'job_id':'test','audio_path':'/unused/audio.wav','language':'en','attempts':1}
    transcriber=Mock(side_effect=NoSpeechError('No speech detected'))
    assert VoiceJobWorker(store,transcriber=transcriber).process_one()
    store.complete.assert_not_called()
    store.fail.assert_called_once_with('test','No speech detected',attempts=1,max_attempts=1)
