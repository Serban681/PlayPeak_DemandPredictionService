# PlayPeak_DemandPredictionService


**PlayPeak_DemandPredictionService** is a Flask-based API for predicting demand using a transformer-based deep learning model.

<details>

<summary>Model Architecture and Design</summary>

The script I am describing can be found here: [Model Creation Script](./model_creation.ipynb)

## Model Architecture

The goal is to process a data sequence and identify repetitive patterns within that sequence in order to make future predictions. The data sequence refers to the values I aim to detect as recurring, while the supplementary data indicates when these values repeat. Specifically, the data in the sequence represents the number of orders placed on a given date and the number of users registered on that date, while the supplementary data includes the day and the month. As shown in the figure below, I used a transformer-based architecture to take advantage of its attention mechanism, and applied the ReLU activation function to the supplementary data through fully connected layers. In the end, the two outputs are concatenated, followed by a final fully connected layer with ReLU functions.

To make future predictions, the output data is reused as input at the data sequence point.

![image](https://github.com/user-attachments/assets/91dbda33-7f82-42a3-a7a4-801edda85c6c)

## Training the Model

The figure below shows the model’s outputs compared to the test data. As can be seen, the prediction system tends to generalize slightly more, with a test error of 0.0067. However, the model provides sufficiently accurate predictions for the user to observe the general trends.

![image](https://github.com/user-attachments/assets/d35ca2c1-ab5d-4b14-9c94-4c8558f592fa)


</details>



<details>

<summary>How to set up</summary>

---

## Requirements

- Python 3.8 or higher
- pip (Python package installer)

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Serban681/PlayPeak_DemandPrediction.git
cd PlayPeak_DemandPrediction
```

### 2. Create a Virtual Environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configuration

Create a .env file in the root directory and add the following:

```dotenv
DB_API_URL=http://localhost:8080/api/v1
```

## Run the Flask App

```bash
python app.py
```

By default, the app will be available at:
http://127.0.0.1:5000/

</details>
