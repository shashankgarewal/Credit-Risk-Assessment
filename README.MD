### Introduction:
In this  project, I have build a model to classify customer into different credit risk category.

This project is designed to classify customers into different credit risk categories based on their financial data, which can help financial institutions and analysts in credit risk assessment. By leveraging machine learning techniques, this project aims to provide a 2 stage model for assessing credit risk without relying only on the credit scores. 

### Report:
An executive summary is mainly focuses on providing business insights and outcome from the project to the non-technical and business team. 

### Project Structure:
* `dataset/`: Contains the dataset files.
* `notebook/`: Jupyter notebooks for data exploration and inital model development.
* `artifacts/`: Preprocessed data created after various feature engineering steps.
* `src/`: Python scripts for model training and local deployment.
* `models/`: Saved multiple attempts of models and related files.
* `deliverables/`: Final model and reports.

Note: The preprocessing and deployment script would also be the part deliverables, however to prevent the repreatative storage they are not stored in deliverables.

### Dataset:

This project utilizes customer credit record from bank or any financial lending institute and CIBIL agency received seperately. The raw data received from the two entities are combined using python pandas in the notebook. The combined data can processed in the explore.ipynb which generates and stores processed data from multiple steps in artifacts. 

The processed data in artificats can be further be used to train models, and evaluate performance using the provided scripts in src, making it a valuable tool for credit risk analysis and decision-making.

Dataset can also be found at kaggle https://www.kaggle.com/datasets/saurabhbadole/leading-indian-bank-and-cibil-real-world-dataset

### Usage:
Follow these steps to run the pre-trained model and system on a local machine:
1. Clone the repository:
    ```bash
    git clone https://github.com/shashankgarewal/credit-risk-checker.git
    ```
2. Navigate to the project directory and create a virtual environment:
    ```bash
    cd credit-risk-assessment
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Flask app:
    ```bash
    python src/deploy.py
    ```
5. Open your web browser and go to `http://127.0.0.1:5000` to access the application.
    ```
### Exploration:

In addition to building the credit risk assessment system, I have observed that the accuracy of credit risk classification significantly degrades when credit score metrics formulated by CIBIL agencies are not used in the modeling process.

Interestingly, in another exploration, I attempted to formulate the credit score using the available credit records. The regression model's performance was surprisingly good, achieving an R² score of over 0.9, despite the notable degradation in credit risk classification when using this score metric.


### Next Steps:
- Implement dynamic model versioning for each training session.
- Enhance the front-end design for better user experience.
- Add interactive model widgets to the front-end.
- Display all feature importance values and names on the front-end.
- Fine-tune the model or experiment with different models for P1-P3 classification.
- Integrate additional data sources to improve model accuracy.
- Gather user feedback and iterate on the design, functionality, and performance of the model.

### Contributing:
Feel free to fork this repository, make improvements, and submit a pull request. Contributions are welcome!

### License:
This project is licensed under the MIT License. See the `LICENSE` file for more details.

### Author
- Linkedin
- Kaggle
- E-Mail