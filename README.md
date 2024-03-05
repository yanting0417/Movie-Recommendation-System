# Movie-Recommendation-System
This project aims to provide personalized movie recommendations based on user preferences by analyzing user behavior and historical movie ratings to suggest movies that users are likely to enjoy.

## To Test
Install dependencies
`pip install -r requirements.txt`

Run this on terminal
`uvicorn main:app --reload`

Navigate to `http://localhost:8000/docs` on your browser

Click on the `GET /recommendations/` and enter any user id; a JSON response will be shown.

