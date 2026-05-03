# Movie Recommendation System

A modern, full-stack movie recommendation engine built using **Item-Item Collaborative Filtering**. The system analyzes user rating patterns from the MovieLens 100K dataset to find movies with similar audience overlap using **Cosine Similarity**.

## Team
- **Syed Asjad Ali Zaidi** (22K-4234)
- **Affan Jan** (22K-4475)
- **Muhammad Saad** (22K-4407)

## Core Technique
The engine uses **Item-Item Collaborative Filtering**. Unlike traditional user-based filtering, this approach calculates similarity between items (movies) based on their rating vectors across all users. 

$$\cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$

## Project Structure
- `app.py`: Flask REST API serving recommendations and movie lists.
* `Movie_Recommender_User_Input.py`: Core logic and CLI version of the recommender.
* `RS_Front/`: React frontend (Vite + React 19) included in this repository — start from that folder.
* `u.data` / `u.item`: MovieLens 100K dataset files.

## How to Run

### 1. Web Application (Recommended)
You need to run both the backend and the frontend servers:

**Start the Backend API:**
```powershell
python app.py
```
*The API will run on http://127.0.0.1:5000*

**Start the Frontend UI:**
```powershell
cd RS_Front
npm install
npm run dev
```
*Open http://localhost:5173 in your browser to interact with the system.*

### 2. Command Line Interface
For a simple terminal-based version:
```powershell
python Movie_Recommender_User_Input.py
```

### 3. Analysis Notebook
To view the data processing pipeline and evaluation (Precision@K):
```powershell
jupyter notebook Movie_Recommender_Notebook.ipynb
```

## Features
* **Autocomplete Search**: Instantly find movies from the 1,600+ available titles.
* **Partial Matching**: Smart search that finds the most popular movie even if you don't type the full title.
* **Visual Similarity**: Ranked results with a visual bar showing the match strength (0-100%).
* **Quality Filter**: Only includes movies with at least 50 ratings to ensure high-quality recommendations.