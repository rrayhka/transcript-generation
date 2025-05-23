# 🇮🇩 Whisper Subtitle Generator (Bahasa Indonesia)

Skrip Python untuk menghasilkan subtitle Bahasa Indonesia dari file video menggunakan model [Whisper](https://huggingface.co/cahya/whisper-medium-id) dari HuggingFace. Mendukung ekstraksi audio, segmentasi, transkripsi, dan penyimpanan ke format `.srt`.

## 🔧 Fitur

* Ekstraksi audio dari file video (MP4, MKV, dll)
* Konversi ke audio mono 16kHz
* Segmentasi audio otomatis dengan overlap
* Transkripsi Bahasa Indonesia menggunakan model Whisper
* Output dalam format subtitle `.srt`

## 📦 Dependencies

Pastikan Python ≥ 3.8.

Instal semua dependensi dengan:

```bash
pip install -r requirements.txt
```

> Untuk performa optimal, gunakan GPU (CUDA).

## 🚀 Cara Pakai

```bash
python main.py path/to/video.mp4
```

### Contoh:

```bash
python main.py test_video.mp4
```

Output:

* `test_video.wav`: Audio hasil ekstraksi
* `test_video.srt`: File subtitle

## 🧠 Model

Model yang digunakan:

* [`cahya/whisper-medium-id`](https://huggingface.co/cahya/whisper-medium-id): Whisper fine-tuned Bahasa Indonesia dari Openai

## 🛠 Struktur Kode

* `extract_audio()`: Ekstrak dan simpan audio dari video
* `load_audio()`: Normalisasi waveform (mono, 16kHz)
* `segment_audio()`: Membagi audio menjadi segmen dengan overlap
* `transcribe_segments()`: Transkripsi tiap segmen menggunakan Whisper
* `generate_srt()`: Tulis file `.srt` dari hasil transkripsi

## ⚠️ Catatan

* File audio akan dihasilkan secara otomatis, atau menggunakan yang sudah ada.
* Segmentasi menggunakan 5 detik dengan overlap 0.5 detik (default).
* Subtitle mungkin tidak 100% akurat, tergantung kualitas audio dan model.
