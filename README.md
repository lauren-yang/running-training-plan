# running-training-plan

# Training Plan Generator

## Instructions

1. **Input Your Data**:
   - Place your activities data in a CSV file named `activities.csv` inside the `data/` directory.
   - The CSV file should contain the following columns: `Activity Date`, `Activity Type`, `Distance`, `Average Speed`, `Average Heart Rate`, `Average Cadence`, and `Predicted 5K Time`.

2. **Install Dependencies**:
   - Make sure you have Python installed.
   - Install the required packages by running:
     ```bash
     pip install -r requirements.txt
     ```

3. **Run the Application**:
   - Navigate to the `scripts/` directory and run the main application script:
     ```bash
     python main.py
     ```

4. **Output**:
   - The generated training plan will be saved in the `output/` directory as `training_plan.csv`.
