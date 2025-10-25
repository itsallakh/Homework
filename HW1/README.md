# ThreadUp_BassModel_AllaKhojayan

This project was completed as part of the Innovation Diffusion Analysis (Bass Model Homework) for the Business Analytics course at AUA.
The goal of this assignment is to model and predict the diffusion of an innovation using the Bass Diffusion Model.
Chosen innovation (2024): ThredUp — a leading online thrift marketplace and sustainable fashion platform, recognized in TIME’s Best Innovations of 2024.
Look-alike innovation: eBay — chosen for its comparable role as a pioneering peer-to-peer online marketplace that transformed how consumers buy and sell second-hand goods.
Scope: Global analysis based on eBay’s historical adoption data.
The Bass Model parameters — coefficient of innovation (p), coefficient of imitation (q), and market potential (M) — were estimated using time-series data of active eBay users. Predictions were then used to simulate ThredUp’s potential diffusion path.

⚙️ How to Run the Notebook
Clone the repository:
git clone https://github.com/itsallakh/ThreadUp_BassModel_AllaKhojayan.git
cd ThreadUp_BassModel_AllaKhojayan

Open the analysis notebook:
Launch Jupyter Notebook or JupyterLab.

Open the file:
BASSMODEL.ipynb

Run all cells in order.
The notebook will:
Load the dataset from the /data directory:
df = pd.read_excel("data/eBay (active buyers).xlsx")
⚠️ Important: Make sure the Excel file eBay (active buyers).xlsx is available inside the data folder before running the notebook.