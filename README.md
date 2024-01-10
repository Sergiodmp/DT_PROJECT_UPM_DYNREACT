
# Appliance of Decision Transformer Architecture for Production Planning
## Overview
This project explores the innovative use of Decision Transformer architecture in the realm of production planning. Specifically, it focuses on scheduling the production of steel coils for a company, aiming to minimize costs. The core of the project involves developing a code that tests the efficacy of this architecture in a real-world manufacturing scenario.

## Features
OpenAI Gym Environment: Custom environment tailored for production planning scenarios.
Decision Transformer Integration: Leveraging the power of Decision Transformers for efficient production scheduling.
Dataset Compatibility: Ability to work with custom datasets structured for Decision Transformers.
Model Training and Evaluation: Facilities to train the model and evaluate its performance.
Installation
To get started with this project, clone the repository and install the necessary dependencies.

git clone https://github.com/Sergiodmp/DT_PROJECT_UPM_DYNREACT/tree/main

## Usage
After setting up the project, follow these steps to train and evaluate the model:

Training the Model: Run train2.py using the included dataset or your own dataset.

bash
Copy code
python train2.py
This step will generate a .pt file representing the trained model.

Evaluating the Model: To view the results and analyze the learning process, run plot.py.

```bash```
python plot.py
The model operates on data extracted from an Excel file, detailing the number of coils and their specific characteristics that need to be scheduled.

## Contributing
Contributions to enhance the functionality or efficiency of this project are welcome. Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License.

## Contact
For any queries or collaborations, please reach out through the Issues section on GitHub.

