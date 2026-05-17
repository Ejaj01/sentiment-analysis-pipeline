# NLP Sentiment Analysis Pipeline 🚀

A lightweight, end-to-end Machine Learning pipeline featuring a custom **PyTorch Deep Learning Binary Classifier** built from scratch, deployed via an interactive **Streamlit** web application dashboard. 

---

## 🛠️ Architecture & Workflow

The system takes raw text string inputs and processes them through an entire localized MLOps pipeline:

```text
[Raw Text Input] 
       │
       ▼
[CountVectorizer (16-Word Vocabulary Map)] ➡️ Tokenizes & Vectorizes into Sparse Matrix
       │
       ▼
[PyTorch Linear Model (Neural Network Layer)] ➡️ Multiplies features by learned parameters/weights
       │
       ▼
[Sigmoid Activation Function] ➡️ Squashes raw numerical outputs to a scale between 0.0 and 1.0
       │
       ▼
[Streamlit Frontend Dashboard] ➡️ Interprets final values as a positive/negative probability percentage# sentiment-analysis-pipeline



Core Technical Highlights
From-Scratch Deep Learning Architecture: Built a custom neural network module using PyTorch's nn.Module containing a linear mapping layer and a Sigmoid activation sequence.

Vectorized Data Processing: Handled text transformation via matrix mapping, converting custom unstructured text tokens into structured frequency arrays.

Optimization & Loss Tracking: Configured Binary Cross-Entropy Loss (nn.BCELoss) combined with the Adam optimizer, tracking model evaluation over a localized training epoch loop.

Serialized Matrix Pipelines: Saved and managed model checkpoint arrays (.pth) and mapping data structures (.pkl) to separate training execution from production runtime.



Tech Stack Used
Language: Python 3.13

Deep Learning Framework: PyTorch

Data Processing & NLP: Scikit-Learn (CountVectorizer)

Frontend Web Framework: Streamlit

Deployment Automation: Localized sub-process mapping


How To Run the Application Locally
1. Clone the Repository
Bash
git clone [https://github.com/Ejaj01/sentiment-analysis-pipeline.git](https://github.com/Ejaj01/sentiment-analysis-pipeline.git)
cd sentiment-analysis-pipeline

2. Configure Virtual Environment & Dependencies
Ensure your environment paths are set up and that you have installed torch, streamlit, and scikit-learn.

3. Launch the App via the Automated Entrypoint
Simply execute the root script, which will handle paths programmatically and boot up the interactive Streamlit server automatically:

Bash
python main.py
