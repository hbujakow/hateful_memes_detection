# Memes Analysis

## Overview

The Memes Analysis project is designed with a modular architecture to enhance extensibility and adaptability for future enhancements. The project focuses on analyzing memes, with specific modules for handling data, detecting hateful content, and providing a web application interface.

## Folder Structure

The project follows a well-organized folder structure for clarity and maintainability:

- **data:** Contains subdirectories for storing images and captions.
- **hateful_memes:**
  - **captions:** API for handling captions. Refer to [README.md](hateful_memes/captions/README.md) for setup and requirements.
  - **inpainting:** API for inpainting. Refer to [README.md](hateful_memes/inpainting/README.md) for setup and requirements.
  - **demo:** Contains the web application source code.
  - **llava:** Module for fine-tuning and evaluating the Visual Large Language Model (LLaVA). Refer to [README.md](hateful_memes/llava/README.md) for setup and requirements.
  - **procap:** API and module for training and evaluating the ProCap model. Refer to [README.md](hateful_memes/procap/README.md) for setup and requirements.
- **tests:** Holds unit and integration tests for the project.

## Essential Configuration Files

- **.gitignore:** Specifies files and directories ignored by the version control system.
- **.isort.cfg:** Configuration file for the [isort](https://pycqa.github.io/isort/) tool, ensuring consistent and organized library sorting.
- **.pre-commit-config.yaml:** Configuration file for pre-commit hooks that enforce checks for code quality and formatting, following [PEP8 standards](https://peps.python.org/pep-0008/).
- **README.md:** This file – provides essential documentation and information about the project.

## Authors

- [Hubert Bujakowski](https://github.com/hbujakow)
- [Mikołaj Gałkowski](https://github.com/galkowskim)
- [Wiktor Jakubowski](https://github.com/WJakubowsk)
- ...

## Getting Started

Follow the steps below to set up and run the Memes Analysis project:

1. Clone the repository: `git clone <repository-url>`
2. Navigate to the project directory: `cd memes_analysis`
3. Each module (captions, inpainting, llava, procap) has specific setup instructions detailed in its own `README` located within its respective module folder. Refer to the individual `README` files for module-specific instructions.

## Contribution

We welcome contributions to enhance and improve the project. Feel free to open issues, submit pull requests, or reach out to the maintainers for discussion.
