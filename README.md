# Hateful Memes Detection

## Overview

The Hateful Memes Detection project is designed with a modular architecture to enhance extensibility and adaptability for future enhancements. The project focuses on analyzing memes, with specific modules for handling data, detecting hateful content, and providing a web application interface.

## Folder Structure

The project follows a well-organized folder structure for clarity and maintainability:

- **data:** Contains subdirectories for storing images and captions.
- **src:**
  - **captions:** API for handling captions. Refer to [README.md](src/captions/README.md) for setup and requirements.
  - **inpainting:** API for inpainting. Refer to [README.md](src/inpainting/api/README.md) for setup and requirements.
  - **demo:** Contains the web application source code.
  - **llava:** Module for fine-tuning and evaluating the Visual Large Language Model (LLaVA). Refer to [README.md](src/llava/README.md) for setup and requirements.
  - **procap:** API ([README.md](src/procap/api/README.md) with setup) and module for training and evaluating the ProCap model. Refer to [README.md](src/procap/architecture/README.md) for setup and requirements for training and evaluation module.
  - **hypotheses:** Scripts to perform statistical hypothesis testing of the results produced by the trained models.
- **tests:** Holds unit and integration tests for the project.

## Essential Configuration Files

- **.gitignore:** Specifies files and directories ignored by the version control system.
- **.isort.cfg:** Configuration file for the [isort](https://pycqa.github.io/isort/) tool, ensuring consistent and organized library sorting.
- **.pre-commit-config.yaml:** Configuration file for pre-commit hooks that enforce checks for code quality and formatting, following [PEP8 standards](https://peps.python.org/pep-0008/).

## Authors

- [Hubert Bujakowski](https://github.com/hbujakow)
- [Mikołaj Gałkowski](https://github.com/galkowskim)
- [Wiktor Jakubowski](https://github.com/WJakubowsk)

## Getting Started

Follow the steps below to set up and run the Memes Analysis project:

1. Clone the repository: `git clone https://github.com/hbujakow/hateful_memes_detection`
2. Navigate to the project directory: `cd hateful_memes_detection`
3. Each module (captions, inpainting, llava, procap) has specific setup instructions detailed in its own `README.md` located within its respective module folder. Refer to the individual `README.md` files for module-specific instructions.

## Contribution

We welcome contributions to enhance and improve the project. Feel free to open issues, submit pull requests, or reach out to the maintainers for discussion.

## License

This project is licensed under the MIT License.
