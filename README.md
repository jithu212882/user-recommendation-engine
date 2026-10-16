# Automated Recommendation System

A clean, modular recommendation system architecture built with **FastAPI**, **scikit-learn**, **NumPy**, **SQLAlchemy**, and **Pydantic**.

---

## Project Structure

```
d:/USer REccomendation/
├── .venv/                      # Python virtual environment
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Documentation and run instructions
└── app/
    ├── __init__.py             # App package definition
    ├── main.py                 # FastAPI application entry point
    ├── core/
    │   ├── __init__.py
    │   └── config.py           # Application settings & environment config
    ├── models/
    │   ├── __init__.py
    │   └── schemas.py          # Pydantic data schemas
    ├── services/
    │   ├── __init__.py
    │   └── recommender.py      # TF-IDF & Cosine Similarity recommendation engine
    ├── database/
    │   ├── __init__.py
    │   └── connection.py       # SQLAlchemy database engine and session
    └── api/
        ├── __init__.py
        └── routes.py           # REST API endpoints (/health, /catalog, /recommend)
```

---

## Virtual Environment Activation

### Windows (PowerShell)
```powershell
.\.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
.\.venv\Scripts\activate.bat
```

### macOS / Linux (bash/zsh)
```bash
source .venv/bin/activate
```

---

## Dependencies

- **FastAPI**: High-performance asynchronous REST API framework
- **Uvicorn**: Lightning-fast ASGI web server
- **Requests**: HTTP client for external service integration
- **Scikit-learn**: Machine learning algorithms (TF-IDF vectorization, Cosine Similarity)
- **NumPy**: Numerical and matrix operations
- **SQLAlchemy**: ORM and SQL database layer
- **Pydantic**: Robust data validation and serialization

To reinstall or update dependencies:
```powershell
pip install -r requirements.txt
```

---

## Running the API

Start the development server with live reload:
```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once running, interactive documentation is available at:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Health check**: [http://127.0.0.1:8000/api/v1/health](http://127.0.0.1:8000/api/v1/health)

---

## Example Recommendation Request

### Request (`POST /api/v1/recommend`):
```json
{
  "user_id": 101,
  "top_k": 3,
  "user_interests": ["mechanical keyboard", "coding", "productivity"]
}
```

### Response:
```json
{
  "user_id": 101,
  "recommendations": [
    {
      "item_id": 3,
      "name": "Ergonomic Mechanical Keyboard",
      "category": "Electronics",
      "score": 0.6281
    },
    {
      "item_id": 6,
      "name": "4K Ultra-HD Curved Monitor",
      "category": "Electronics",
      "score": 0.2145
    },
    {
      "item_id": 8,
      "name": "Machine Learning Fundamentals Book",
      "category": "Books",
      "score": 0.1873
    }
  ],
  "total_returned": 3
}
```
