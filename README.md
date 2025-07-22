# fer_project
Facial Emotion Recognition project

# Facial Emotion Recognition using CNN

PyTorch-based framework for Facial Emotion Recognition (FER), built around custom CNN architectures.  


---

## 📦 Project Structure

```text
fer_project/
│
├── main.py
├── models/
├── trainers/
├── utils/
│   ├── dataset.py
│   └── losses.py
├── cfgs/
├── dataset/
│   └── icml_face_data.csv
├── notebooks/
└── README.md
```


## 🔧 Setup

1. **Clone the repository**

```bash
git clone https://github.com/dinar-orazgaliyev/fer_project.git
cd fer_project

Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

Install the dependencies
pip install -r requirements.txt

📁 Dataset
This project uses FER2013 dataset which can be downloaded from:
https://www.kaggle.com/datasets/msambare/fer2013

Please Note that folder "dataset" to be created in the project root.
fer_project/dataset/icml_face_data.csv

🚀 Running the Model
To train a CNN-based model on FER2013:
python main.py --model cnn_fer
additional parameters can be changed in the yaml config files:
fer_projec/cfgs




The FER2013 dataset is distributed under the [MIT License](https://opensource.org/licenses/MIT).  




