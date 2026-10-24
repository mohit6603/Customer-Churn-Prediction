# Examples

Synthetic customers (not from the real dataset) for trying the API and the
Streamlit app without downloading the data.

| File | Use with |
|---|---|
| `customer.json` | `POST /predict` (one customer, API field names) |
| `sample_customers.csv` | `POST /predict/batch`, `python -m src.monitoring`, or the Streamlit "Batch scoring" tab (raw IBM column names) |

```bash
uvicorn api.main:app --reload

curl -X POST http://127.0.0.1:8000/predict \
     -H "Content-Type: application/json" -d @examples/customer.json

curl -X POST "http://127.0.0.1:8000/predict/batch?top_n=5" \
     -F "file=@examples/sample_customers.csv"
```
