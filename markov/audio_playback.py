import os
import subprocess
import shutil
# Not used in this file, but apparently needed for some reason
# If you remove this, the program won't find the systems fluid synth executable, even though it's in the PATH. No idea why.
import pretty_midi

class AudioPlayback:
    _player = None # Subprocess for audio playback, handle to running fluidsynth program

    @staticmethod
    def stop_playback():
        # .poll() returns:
        #   None → still running
        #   number → already finished
        # This ensures you don’t try killing a dead process
        if AudioPlayback._player and AudioPlayback._player.poll() is None:
            AudioPlayback._player.terminate()
            # Wait for child process to terminate to avoid zombies
            AudioPlayback._player.wait()
            AudioPlayback._player = None

    @staticmethod
    def play_midi_fluidsynth(midi_path, sf2_path, gain=0.8):
        midi_path = os.path.abspath(midi_path)
        sf2_path = os.path.abspath(sf2_path)
    
        if not os.path.exists(midi_path):
            raise FileNotFoundError(midi_path)
        if not os.path.exists(sf2_path):
            raise FileNotFoundError(sf2_path)

        # Stop any existing playback
        AudioPlayback.stop_playback()

        # Automatically locate fluidsynth in PATH
        fluidsynth_path = shutil.which("fluidsynth")
        if fluidsynth_path is None:
            raise FileNotFoundError(
                "Fluidsynth executable not found. Install it and make sure it is in your PATH."
            )
    
        # Build the FluidSynth command
        # This is basically running: fluidsynth -ni -g 0.8 soundfont.sf2 music.mid
        # -n : no shell interaction
        # -i : no interactive mode
        # -g : gain (volume)
        cmd = [fluidsynth_path, "-ni", "-g", str(gain), sf2_path, midi_path]

        # Start FluidSynth as a subprocess
        AudioPlayback._player = subprocess.Popen(
            cmd,
            # Without this, FluidSynth would spam your console with logs and warnings
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    @staticmethod
    def is_playing():
        return AudioPlayback._player is not None and AudioPlayback._player.poll() is None
