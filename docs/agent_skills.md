# Agent Roles & Skills Required

## 1. Vision & Layout Parsing Engineer
* **Skills:** PyTorch, OpenCV, spatial coordinate math, bounding box IOUs.
* **Responsibility:** Extract high-quality images from manga PDFs, isolate panel boundaries, and implement robust right-to-left sequencing metrics.

## 2. Multi-Modal Scripting Architect
* **Skills:** Vision-Language Model prompting, JSON schema enforcement, regex text sanitization.
* **Responsibility:** Engineer highly constrained VLM prompts that reliably output machine-readable JSON containing speaker mappings and descriptive action logs.

## 3. Audio Synthesis & Voice Engineer
* **Skills:** Coqui XTTSv2, Bark, Voice cloning, sound array mastering (`pydub`/`librosa`).
* **Responsibility:** Assign, maintain, and generate highly descriptive, expressive voices in Portuguese, ensuring clear separation between dialogues and scene descriptions.

## 4. Automation & TDD Specialist
* **Skills:** Pytest, clean architecture, abstract base classes, typing.
* **Responsibility:** Enforce code health by building a full test suite utilizing robust mocks to test data parsing without launching heavy local GPU models.