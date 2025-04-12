# Fatigue Detection with Python

<div align="center">
  <img src="https://github.com/user-attachments/assets/2bd8143d-725b-484b-99bc-8e034c43e17a" width="800" height="350" />
</div>


## Overview 

Welcome to the Fatigue Detection System repository! This project, developed during my Data Science postgraduate course at PUC Minas, focuses on detecting fatigue using Python. The system is designed to monitor signs of fatigue, such as eye blinking and head movements, leveraging advanced image processing techniques and machine learning models. By leveraging MediaPipe, the system monitors key indicators of fatigue, such as eye blinking and head movement, in real-time. A graphical user interface displays these results, highlighting critical moments when fatigue is detected and alerts are triggered. The system aims to improve safety by being applicable in both road safety and workplace environments, with future plans to integrate machine learning for enhanced accuracy and adaptability.

> Check out the video demonstration to see the code in action and understand how the system works! 🎥👇🏽

<div align="center">
<img src="https://github.com/user-attachments/assets/1c31cc98-eec3-4e68-9ddb-a25ca0d37ff3" width="720" height="400" />
</div>


## Table of Contents 📚

- Introduction
- Features
- Installation
- Usage
- Contributing
- License
- Contact

## Introduction 🎯⚙️

Fatigue detection is critical for ensuring safety and well-being in various environments, including workplaces and on the road. This project utilizes video capture techniques to monitor facial and ocular movements, helping to prevent accidents caused by drowsiness.

## Features 🤌🏽✨

- **Real-time Fatigue Detection**: Monitors eye blinking and head movements using video capture.
- **Advanced Image Processing**: Utilizes the MediaPipe library for detecting and analyzing facial landmarks.
- **User-friendly Interface**: Includes a graphical interface for real-time visualization of results and save as an image.

## Installation 🧩

1. Clone the repository:
    ```bash
    git clone https://github.com/IgorMoreira/fatigue-detection-system.git
    cd fatigue-detection-system
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage 🚨⚠️

To run the fatigue detection system:

1. Ensure your webcam is connected and functioning.
2. Execute the main script:
    ```bash
    python detector_fadiga.py
    ```
3. The system will start capturing video and monitoring for signs of fatigue. Results will be displayed in real-time.
4. When the simulation is finished, it is possible to save the graph as an image.
5. It is possible to collect the data generated during the simulation in the file "dados_abertura_olhos.csv".

## Contributing 🔃💭

We welcome contributions to enhance the project! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature-name`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/your-feature-name`).
5. Open a pull request.

## License ©️

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact 📞📩

For more information, feel free to reach out:

- GitHub: [IgorMoreira](https://github.com/IgorMoriera)
- LinkedIn: [igors-moreira](https://www.linkedin.com/in/igors-moreira/)
  

Thank you for your interest in the Fatigue Detection System! Your feedback and contributions are highly appreciated. 👋🏽

#Python #MachineLearning #Safety #FatigueDetection #Technology #Innovation
