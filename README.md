🚗 AI-Powered Autonomous Driving System
Welcome to my B.Tech Project! This repository showcases an AI-based autonomous driving system where a Convolutional Neural Network (CNN) imitates human driving behavior using Behavioral Cloning. The project includes data collection, preprocessing, model implementation, and real-time testing using the Udacity Simulator.

📌 Problem Statement
Most believe autonomous vehicles are a luxury, but they are a necessity now:

🚦 Rising traffic fatalities
❌ Ineffective traffic management systems
⚠️ 94% of road accidents caused by human errors
This project aims to address these issues by mimicking human intuition and steering decisions using machine learning.

💡 Proposed Solution
The AI system uses Behavioral Cloning to predict steering angles based on real-time camera images:

Data Collection: Capturing images from center, left, and right cameras with corresponding parameters (steering, throttle, speed).
Preprocessing & Augmentation: Cleaning noisy data and simulating real-world driving conditions.
CNN Model: A convolutional neural network processes the images and predicts steering angles.
Testing: Evaluated on unseen tracks in the Udacity Simulator.
🚀 Implementation and Methodology
1. Data Collection
Utilized the Udacity Simulator to log data (images + driving parameters).
Three perspectives (center, left, right) were captured to account for slight deviations.
2. Data Preprocessing & Augmentation
To ensure generalization and avoid overfitting:

Adjusted brightness to simulate day/night driving.
Added panning for realistic lane deviation.
Flipped images for left/right symmetry.
Zoomed for varying road views.
<p align="center"> <img src="https://via.placeholder.com/600x300?text=Augmented+Data" alt="Augmented Data Example" /> </p>
3. CNN Model
The model architecture is inspired by NVIDIA's CNN design for autonomous vehicles.

Input: Preprocessed images
Activation: Used ELU (Exponential Linear Unit) for non-linearity.
Output: Steering angles
4. Real-Time Testing
Successfully tested on unseen tracks in the simulator.
The car self-corrected deviations and performed well under different conditions.
📊 Results and Analysis
Training & Validation Loss: Continuously decreasing, showing effective learning.
Testing Performance: Car navigated smoothly on multiple tracks.
Key Metrics:
Loss Graph: Continuous improvement across epochs.
Model Output: Steady steering and recovery from deviations.
Videos and Screenshots:

🎥 Watch the Demo Video to see the model in action.

🚧 Challenges Faced
Insufficient Data: Solved through data augmentation (brightness, flipping, zooming, etc.).
Model Drift: Car deviated due to overfitting, addressed by fine-tuning the CNN.
🌟 Key Learnings
Effective data preprocessing significantly improves model performance.
Behavioral cloning provides an intuitive solution for AI-based steering prediction.
Data diversity and augmentation are crucial for generalizing to new tracks.
📈 Future Improvements
Incorporate object detection for traffic signs and pedestrians.
Use GPS data for route planning.
Add real-world datasets for enhanced robustness.
🧰 Tech Stack
Programming Language: Python
Libraries: Keras, OpenCV, TensorFlow
Tools: Udacity Simulator, Flask, Eventlet

🛠️ How to Run the Project
Clone the repository:

bash
Copy code
git clone https://github.com/AlekhyaGudibandla/Autonomous-Car.git
cd Autonomous-Car
Set up the environment:
Install required dependencies using:

bash
Copy code
pip install -r requirements.txt
Run the model:

Open Final_Behavioural_cloning.ipynb to train and evaluate the model.
Start the Udacity Simulator in autonomous mode.
Run the server:
bash
Copy code
python drive.py
Observe Results:
The car will navigate autonomously in the simulator window.

📊 Results
Training Loss: Shows consistent reduction across epochs.
Generalization: Model performs well on both training and unseen tracks.
Videos: Demonstrations of the car steering autonomously will be included.
Screenshots and videos demonstrating the model performance are added in the repository.

🔗 References
Udacity Self-Driving Car Simulator
NVIDIA's End-to-End Learning for Self-Driving Cars
Thank you for visiting this repository! 🚀 Feel free to explore the notebooks, and contributions are always welcome!

⭐ If you find this useful, don't forget to star this project.
