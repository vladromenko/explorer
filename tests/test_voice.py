from pathlib import Path
import tempfile
import unittest
import wave
import numpy as np
from voice import audio_quality

class VoiceTests(unittest.TestCase):
    def test_silence_and_clipping_are_measured(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'sample.wav'
            for values,quiet in [(np.zeros(16000,dtype='<i2'),True),(np.full(16000,32767,dtype='<i2'),False)]:
                with wave.open(str(p),'wb') as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(values.tobytes())
                result=audio_quality(p)
                self.assertEqual(result['duration_s'],1)
                if quiet:self.assertLess(result['rms_dbfs'],-55)
                else:self.assertGreater(result['clipped_fraction'],.99)

    def test_wrong_sample_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'stereo.wav'
            with wave.open(str(p),'wb') as w:
                w.setnchannels(2);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\0'*4000)
            with self.assertRaises(ValueError):audio_quality(p)

if __name__=='__main__':unittest.main()
