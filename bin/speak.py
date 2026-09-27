#!/usr/bin/env python3
"""Local Russian speech synthesis; receives text on stdin, writes one WAV."""
import json,sys,wave
from pathlib import Path
from piper import PiperVoice,SynthesisConfig
root=Path('/home/vlad/Explorer')
config=json.loads((root/'config/voice.json').read_text())
text=sys.stdin.read(1201).strip()
if not text or len(text)>1200:raise ValueError('Speech text must be 1..1200 characters')
voice=PiperVoice.load(root/config['voice_model'])
with wave.open(sys.argv[1],'wb') as wav:
    voice.synthesize_wav(text,wav,syn_config=SynthesisConfig(volume=config['playback_volume'],length_scale=1.05))
