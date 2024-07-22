# running-training-plan

# Training Plan Generator

## Instructions

1. **Input Your Data**:
   - Bulk download activities from Strava ([[link](https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export)])
   - Replace the existing file with your activity data in a CSV file named `activities.csv` inside the `data/` directory.
   - The CSV file should contain the following columns: `Activity Date`, `Activity Type`, `Distance`, `Average Speed`, `Average Heart Rate`, `Average Cadence`, and `Predicted 5K Time`.

3. **Install Dependencies**:
   - Make sure you have Python installed.
   - Install the required packages by running:
     ```bash
     pip install -r requirements.txt
     ```

4. **Run the Application**:
   - Set up Python virtual environment
   - Navigate to the `scripts/` directory and run the following scripts:
    ```bash
     python3 train_model.py
     ```
     ```bash
     python3 main.py
     ```

6. **Output**:
   - The generated training plan will be saved in the `output/` directory as `aggregated_recommended_training_plan.csv`.
   - The training plan will also be saved onto your computer as 'aggregated_recommended_training_plan.csv' if running in terminal.
