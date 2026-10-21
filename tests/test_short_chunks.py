import os
from pathlib import Path
import subprocess
import sys
import unittest


class ShortChunkConfigTests(unittest.TestCase):
    def test_chunk_bounds_and_unchanged_default(self):
        for value, expected in [(None,720),('30',30),('1',30),('60',60),('900',780)]:
            with self.subTest(value=value):
                env=dict(os.environ)
                env.pop('WIZARD_STT_PROVIDER_CHUNK_SECONDS',None)
                if value is not None:
                    env['WIZARD_STT_PROVIDER_CHUNK_SECONDS']=value
                output=subprocess.check_output([sys.executable,'-c',
                    'import providers; print(providers.PROVIDER_CHUNK_SECONDS); print(providers.normalize_audio_chunks.__kwdefaults__["chunk_seconds"])'],
                    cwd=Path(__file__).resolve().parents[1],env=env,text=True)
                self.assertEqual(output.strip().splitlines(),[str(expected),str(expected)])
