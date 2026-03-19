# CAFSS

# STEP 1: Setup VS Code for Python

1. Open VS Code
2. Go to **Extensions (left sidebar)**
3. Search → **Python**
4. Install extension by Microsoft

---

# STEP 2: Create Your Project Folder

1. Create a folder (example):

```
CAFSS_Project
```

2. Open it in VS Code:

* File → Open Folder → Select your folder

---

# STEP 3: Create Python File

1. Inside folder → create file:

```
main.py
```

2. Paste your full code into it
3. Save (Ctrl + S)

# STEP 4: Install Required Libraries

Open **Terminal in VS Code**:

```
Terminal → New Terminal
```

Then run:

```
pip install easyocr
pip install nltk
pip install rapidfuzz
pip install numpy
pip install opencv-python
pip install pillow
```

# IMPORTANT (FIRST TIME ONLY)

Run Python once to download NLTK data:

```
python
```

Then type:

```python
import nltk
nltk.download('stopwords')
nltk.download('opinion_lexicon')
exit()
```

# STEP 5: Add Your Image

* Put your image in the same folder
  Example:

```
amazon_dataset2.jpg
```
OR update path in code:
```python
image_path = "your_image.jpg"
```

# STEP 6: Run the Code

In terminal:
```
python main.py
```

# EXPECTED OUTPUT

You’ll see:

```
Loading OCR model...
OCR extracted...
PER-WORD DEBUG...
Final Score...
FINAL SENTIMENT LABEL: Positive/Negative/Neutral
```

