# Iotav2

SAMA/NORA Compliance Assistant: RAG backend + chat frontend.

## Running the app

1. **Backend** (FastAPI, port 8000):

   ```bash
   cd backend
   pip install -r requirements.txt
   # Set .env for OpenAI/Supabase etc. See backend/README.md
   uvicorn server:app --reload
   ``` 

   Or from project root: `uvicorn backend.server:app --reload`

2. **Frontend** (Next.js, port 3000):

   ```bash
   cd frontend
   npm install
   # Set NEXT_PUBLIC_API_URL=http://localhost:8000 in .env.local (see frontend/.env.example)
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) and submit a SAMA/NORA question. The frontend calls the backend at `http://localhost:8000/api/query`.

## Layout

- `backend/` — RAG pipeline (domain gate, retrieval, Qwen generation), FastAPI server
- `frontend/` — Next.js chat UI (same as IOTARegTechAI frontend), talks to backend `/api/query`
