import yt_dlp

def download_youtube_audio_as_wav(url, output_path="output.wav", volume_boost_db=5):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        # Add ffmpeg args to boost volume by volume_boost_db dB
        'postprocessor_args': [
            '-af', f'volume={volume_boost_db}dB'
        ],
        'quiet': False,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    # Rename the output file to the desired output_path
    import os
    if os.path.exists("temp_audio.wav"):
        os.replace("temp_audio.wav", output_path)
        print(f"Download and conversion done. Saved as: {output_path}")
    else:
        print("Error: Expected output file not found.")

if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    output_file = input("Enter output WAV filename (e.g., myfile.wav): ")
    download_youtube_audio_as_wav(url, output_file)