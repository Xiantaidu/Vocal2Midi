import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import sys
from pathlib import Path

# Add the project root to the path so we can import from inference
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from inference.API.slicer_api import slice_audio

def plot_slicing(audio_path, output_image_path):
    print(f"Loading audio: {audio_path}")
    # Load audio
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    
    # Calculate global pitch and voiced flag using pyin
    print("Calculating global pitch curve using pyin (this might take a while)...")
    f0_min, f0_max = 65.0, 1100.0
    hop_length = 512
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=f0_min, fmax=f0_max, sr=sr, frame_length=hop_length*4, hop_length=hop_length
    )
    f0[~voiced_flag] = np.nan # Set unvoiced to NaN for plotting
    times_f0 = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    print("Running smart slicing (智能切片)...")
    # We use the calculated voiced flag to simulate the RMVPE mask override
    chunks = slice_audio(
        y, 
        sr, 
        method="智能切片",
        rmvpe_voiced_mask=voiced_flag,
        rmvpe_time_step_seconds=hop_length/sr
    )
    
    print(f"Sliced into {len(chunks)} chunks. Generating plot...")

    # Calculate the total duration in seconds to make a really long plot
    total_duration_sec = len(y) / sr
    
    # Let's say we want 4 seconds per inch for highly detailed zooming
    width_inches = max(15, total_duration_sec / 2)
    height_inches = 12 # Give it a bit more vertical breathing room
    
    # Plotting
    plt.figure(figsize=(width_inches, height_inches))
    
    # Subplot 1: Waveform with slice boundaries
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5, label='Waveform')
    
    # To prevent x-axis squishing, we can set xlim based on total duration explicitly
    plt.xlim(0, total_duration_sec)
    
    # Plot chunks with alternating colors
    colors = ['#1f77b4', '#ff7f0e']
    for i, chunk in enumerate(chunks):
        start_sec = chunk['offset']
        dur_sec = len(chunk['waveform']) / sr
        end_sec = start_sec + dur_sec
        color = colors[i % 2]
        
        plt.axvspan(start_sec, end_sec, alpha=0.3, color=color, label='Segment' if i < 2 else "")
        plt.axvline(x=start_sec, color='r', linestyle='--', alpha=0.8)
        if i == len(chunks) - 1:
            plt.axvline(x=end_sec, color='r', linestyle='--', alpha=0.8)
            
        # Add text for chunk index
        plt.text(start_sec + dur_sec/2, np.max(y)*0.9, f"C{i}", horizontalalignment='center', color='black', fontweight='bold')

    plt.title('Waveform and Slicer Segments')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    # Get fine grid on the X-axis for every second to make details easy to read
    plt.xticks(np.arange(0, total_duration_sec + 1, step=1.0), rotation=90)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Subplot 2: Pitch track and Voiced Flag
    plt.subplot(2, 1, 2)
    
    # Set xlim strictly here too
    plt.xlim(0, total_duration_sec)
    
    # Plot voicing probability as background shaded regions
    plt.fill_between(times_f0, 0, voiced_probs * np.nanmax(f0), alpha=0.2, color='gray', label='Voicing Prob')
    
    # Plot pitch
    plt.plot(times_f0, f0, label='F0 (pyin)', color='green', linewidth=2)
    
    # Add slice boundaries
    for i, chunk in enumerate(chunks):
        start_sec = chunk['offset']
        plt.axvline(x=start_sec, color='r', linestyle='--', alpha=0.8)
        if i == len(chunks) - 1:
            dur_sec = len(chunk['waveform']) / sr
            plt.axvline(x=start_sec + dur_sec, color='r', linestyle='--', alpha=0.8)

    plt.title('Pitch Contour (F0) and Slice Points')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend(loc='upper right')
    
    # Align the X-ticks with the subplot above
    plt.xticks(np.arange(0, total_duration_sec + 1, step=1.0), rotation=90)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize smart slicing results.")
    parser.add_argument("audio_path", type=str, help="Path to the input audio file.")
    parser.add_argument("--output", type=str, default="slicer_visualization.png", help="Output image path.")
    
    args = parser.parse_args()
    plot_slicing(args.audio_path, args.output)
