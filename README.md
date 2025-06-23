#  AI Virtual Fitness Coach - Bicep Curl

This is an AI-powered virtual fitness coach that uses computer vision (MediaPipe + OpenCV) to track bicep curl workouts via webcam or uploaded video. It gives real-time feedback, counts reps, tracks rep duration, and visualizes set-wise performance.

##  Features

-  Webcam or uploaded video-based tracking
-  Real-time rep counting
-  Feedback for rep quality (e.g., “Too fast”, “Nice rep”)
-  Customizable workout duration, sets, and minimum rep time
-  Visual charts for rep quality after each set
-  Downloadable workout logs

---

##  Tech Stack

- **Streamlit** for interactive UI
- **MediaPipe** for human pose estimation
- **OpenCV** for image processing and video capture
- **NumPy** & **Matplotlib** for analytics & visualization
- **Pandas** for logging and CSV export

---

##  Getting Started

### 1. Clone the Repository

git clone https://github.com/chaitanyasasi-ux/Ai_virtual_fitness_coach.git
cd Ai_virtual_fitness_coach
# Step 3: Create a virtual environment (recommended)
python -m venv venv

# Step 4: Activate the environment
.\venv\Scripts\activate  # Windows PowerShell
# OR use venv\Scripts\activate.bat in CMD

# Step 5: Install dependencies
pip install -r requirements.txt

# Step 6: Run the app
streamlit run app.py
