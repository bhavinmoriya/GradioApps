import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import windows, fftconvolve
from scipy.fft import fft, fftshift

def apply_window(audio_file, window_type):
    # Read audio file
    sample_rate, signal = audio_file
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # Convert to mono
    signal = signal.astype(float) / np.max(np.abs(signal))  # Normalize

    # Apply selected window
    if window_type == "Rectangular":
        window = np.ones(len(signal))
    elif window_type == "Hann":
        window = windows.hann(len(signal))
    elif window_type == "Hamming":
        window = windows.hamming(len(signal))
    elif window_type == "Blackman":
        window = windows.blackman(len(signal))
    else:
        window = np.ones(len(signal))  # Default: Rectangular

    # Apply window
    windowed_signal = signal * window

    # Compute FFT for visualization
    fft_signal = fft(windowed_signal)
    fft_magnitude = np.abs(fftshift(fft_signal))
    freqs = np.linspace(-sample_rate/2, sample_rate/2, len(fft_magnitude))

    # Generate plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Time domain plot
    ax1.plot(signal, label="Original Signal", alpha=0.5, color="blue")
    ax1.plot(windowed_signal, label="Windowed Signal", color="red")
    ax1.set_title(f"Time Domain: {window_type} Window")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.legend()
    ax1.grid(True)

    # Frequency domain plot (log scale to see side lobes)
    ax2.plot(freqs, 20 * np.log10(fft_magnitude + 1e-10), label="FFT Magnitude")
    ax2.set_title(f"Frequency Domain: {window_type} Window (Side Lobes)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude (dB)")
    ax2.set_ylim(-100, 0)  # Focus on side lobes
    ax2.grid(True)

    plt.tight_layout()

    # Return results
    return (
        fig,
        (sample_rate, windowed_signal),  # Windowed audio
        f"Applied {window_type} window. Listen for artifacts (e.g., clicks with Rectangular)."
    )

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 Window Functions: Hear the Difference!")
    gr.Markdown("Upload an audio file and apply different window functions to hear and see the effects of spectral leakage and side lobes.")

    with gr.Row():
        audio_input = gr.Audio(sources=["upload"], type="numpy", label="Upload Audio File")
        window_type = gr.Dropdown(
            choices=["Rectangular", "Hann", "Hamming", "Blackman"],
            value="Hann",
            label="Select Window Function"
        )

    process_btn = gr.Button("Apply Window and Play")

    with gr.Row():
        with gr.Column():
            plot_output = gr.Plot(label="Time and Frequency Domain")
        with gr.Column():
            windowed_audio = gr.Audio(label="Windowed Audio Output")
            info_text = gr.Textbox(label="Info", interactive=False)

    process_btn.click(
        fn=apply_window,
        inputs=[audio_input, window_type],
        outputs=[plot_output, windowed_audio, info_text]
    )

# Launch the app
demo.launch()
