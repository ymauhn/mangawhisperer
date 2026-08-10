# Project Context: MangaWhisperer (Immersive Manga Audio)

## Objective
Develop a modular Python machine learning pipeline that converts manga PDFs into immersive, multi-voice audio experiences (podcasts/audiobooks) to assist visually impaired readers.

## Dataset & Language
* **Target:** 42 volumes of *Berserk* in Portuguese (PT-BR), formatted as PDFs.
* **Localization Requirement:** The OCR, Text processing, and Text-to-Speech (TTS) layers must natively support Portuguese accents, slang, and phonetics.

## Core Generative Pipeline Stages
1. **Layout & Bubble Segmentation:** Slice PDFs into pages. Detect individual panels and speech bubbles using specialized comic vision models. Sort them using spatial geometry into a right-to-left, top-to-bottom reading matrix.
2. **Optical Character Recognition (OCR):** Isolate speech bubble regions and extract PT-BR text using a Latin-alphabet compatible engine.
3. **Multi-Modal Context Engine (VLM):** Feed the isolated panel images combined with the extracted OCR text into a Vision-Language Model. The VLM acts as a scriptwriter, performing two functions:
   - **Speaker Diarization:** Identify which character spoke the text (e.g., "Guts", "Griffith", "Casca").
   - **Scene Description:** Generate short, highly descriptive audio-description cues for purely visual actions (e.g., "[Ação]: Uma espada massiva rasga o ar com um som metálico").
4. **Voice Matching & TTS Synthesis:** Route the structured script to a multi-speaker TTS engine capable of voice cloning. Map consistent character IDs to specific dark-fantasy voice profiles.
5. **Audio Mastering & Compilation:** Concatenate the speech and action narration files, adding minor operational padding (pauses between panels) to produce a unified MP3/WAV file.

## Design Philosophy
* **Strict Test-Driven Development (TDD):** Every pipeline interface must be verified using mocked data streams before deep learning weights are integrated.
* **Decoupled Architecture:** Pydantic schemas will serve as the immutable data contracts between pipeline stages.