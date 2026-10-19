#!/usr/bin/env python3
"""Denoise + dereverb a reference clip with DeepFilterNet (DeepFilterNet3).

    python scripts/clean_ref.py input.wav output.wav

Runs as a subprocess of the API (see engine.clean_reference) so the server's
torch stack and DeepFilterNet's never have to agree on versions, and the
DeepFilterNet weights are not kept resident between registrations.
Output is 48 kHz mono WAV; the caller resamples to 24 kHz.
"""
import sys
import warnings

warnings.filterwarnings("ignore")


def main() -> int:
    src, dst = sys.argv[1], sys.argv[2]
    from df.enhance import enhance, init_df, load_audio, save_audio

    model, state, _ = init_df(log_level="ERROR")
    audio, _ = load_audio(src, sr=state.sr())
    cleaned = enhance(model, state, audio)
    save_audio(dst, cleaned, state.sr())
    return 0


if __name__ == "__main__":
    sys.exit(main())
