# Iotav2 Frontend

Same UI as IOTARegTechAI's frontend, wired to the Iotav2 backend (SAMA/NORA RAG API).

## Environment

Set the backend API base URL (no trailing slash):

```bash
# .env.local (create from .env.example)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Setup

```bash
cd frontend
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Ensure the Iotav2 backend is running on port 8000 (see project root or backend README).

## Features

- Chat interface with markdown support
- Chat history (localStorage)
- Answers and source citations from `POST /api/query` (Iotav2 backend)
- Reference citations and source panel
