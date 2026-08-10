import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct, idct

def update_slider(audio_file):
    # Read the uploaded WAV file to get its length
    sample_rate, signal = audio_file
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # Convert to mono
    dct_coeffs = dct(signal.astype(float))
    max_coeffs = len(dct_coeffs)
    default_coeffs = max_coeffs // 2  # Default to half the coefficients
    return gr.Slider(
        minimum=1,
        maximum=max_coeffs,
        value=default_coeffs,
        step=1,
        label="Number of DCT Coefficients to Keep"
    )

def process_audio(audio_file, num_coeffs):
    # Read the uploaded WAV file
    sample_rate, signal = audio_file
    if len(signal.shape) > 1:
        signal = signal[:, 0]  # Convert to mono
    signal = signal.astype(float) / np.max(np.abs(signal))  # Normalize

    # Apply DCT
    dct_coeffs = dct(signal)

    # Reconstruct signal using top N coefficients
    reconstructed_coeffs = np.zeros_like(dct_coeffs)
    reconstructed_coeffs[:num_coeffs] = dct_coeffs[:num_coeffs]
    reconstructed_signal = idct(reconstructed_coeffs)

    # Trim the reconstructed signal to match the original length
    reconstructed_signal = reconstructed_signal[:len(signal)]
    reconstructed_signal = reconstructed_signal / np.max(np.abs(reconstructed_signal))  # Normalize

    # Generate plots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10))

    # Original signal
    ax1.plot(signal, label="Original Signal", color="blue")
    ax1.set_title("Original Audio Signal")
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True)

    # DCT coefficients
    ax2.plot(np.abs(dct_coeffs), label="DCT Coefficients", color="red")
    ax2.set_title("Magnitude of DCT Coefficients")
    ax2.set_xlabel("Coefficient Index")
    ax2.set_ylabel("Magnitude")
    ax2.grid(True)

    # Reconstructed signal
    ax3.plot(signal, label="Original Signal", color="blue", alpha=0.5)
    ax3.plot(reconstructed_signal, label="Reconstructed Signal", color="green")
    ax3.set_title(f"Reconstructed Signal (Top {num_coeffs} DCT Coefficients)")
    ax3.set_xlabel("Samples")
    ax3.set_ylabel("Amplitude")
    ax3.legend()
    ax3.grid(True)

    # Error
    error = signal - reconstructed_signal
    ax4.plot(error, label="Error", color="red")
    ax4.set_title("Error (Original - Reconstructed)")
    ax4.set_xlabel("Samples")
    ax4.set_ylabel("Amplitude")
    ax4.grid(True)

    plt.tight_layout()

    # Calculate compression ratio
    compression_ratio = (len(signal) - num_coeffs) / len(signal) * 100
    compression_info = f"Compression Ratio: {compression_ratio:.2f}% (Kept {num_coeffs}/{len(signal)} coefficients)"

    return (
        fig,
        compression_info,
        (sample_rate, signal),
        (sample_rate, reconstructed_signal)
    )

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🎵 DCT for Audio Compression")
    gr.Markdown("Upload a WAV file and adjust the number of DCT coefficients to see how compression affects the audio signal.")

    with gr.Row():
        audio_input = gr.Audio(sources=["upload"], type="numpy", label="Upload Audio File")
        num_coeffs = gr.Slider(
            minimum=1,
            maximum=1000,  # Placeholder, will be updated
            value=500,     # Placeholder, will be updated
            step=1,
            label="Number of DCT Coefficients to Keep",
            interactive=False  # Disable until audio is uploaded
        )

    process_btn = gr.Button("Process Audio")

    with gr.Row():
        with gr.Column():
            plot_output = gr.Plot(label="Signal and DCT Analysis")
            compression_info = gr.Textbox(label="Compression Info")
        with gr.Column():
            original_audio = gr.Audio(label="Original Audio")
            reconstructed_audio = gr.Audio(label="Reconstructed Audio")

    # Update slider when audio is uploaded
    audio_input.change(
        fn=update_slider,
        inputs=[audio_input],
        outputs=[num_coeffs]
    )

    # Enable the slider and button after audio is uploaded
    def enable_slider(audio_file):
        return gr.Slider(interactive=True), gr.Button(interactive=True)

    audio_input.change(
        fn=enable_slider,
        inputs=[audio_input],
        outputs=[num_coeffs, process_btn]
    )

    process_btn.click(
        fn=process_audio,
        inputs=[audio_input, num_coeffs],
        outputs=[plot_output, compression_info, original_audio, reconstructed_audio]
    )

# Launch the app
demo.launch(share=True)
