"""
Clintelligence - Minimal Version for Testing
"""
import os
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Clintelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Clintelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen flex items-center justify-center">
    <div class="text-center p-8">
        <h1 class="text-5xl font-bold mb-4 bg-gradient-to-r from-orange-500 to-blue-500 bg-clip-text text-transparent">
            Clintelligence
        </h1>
        <p class="text-xl text-gray-300 mb-8">AI-Powered Clinical Trial Protocol Intelligence</p>
        <div class="bg-gray-800 rounded-lg p-8 max-w-2xl">
            <h2 class="text-2xl font-semibold mb-4">Platform Features</h2>
            <div class="grid grid-cols-2 gap-4 text-left">
                <div class="bg-gray-700 p-4 rounded">
                    <h3 class="font-bold text-orange-400">Protocol Risk Scoring</h3>
                    <p class="text-sm text-gray-400">Predict amendments & delays</p>
                </div>
                <div class="bg-gray-700 p-4 rounded">
                    <h3 class="font-bold text-blue-400">Site Intelligence</h3>
                    <p class="text-sm text-gray-400">Find optimal trial sites</p>
                </div>
                <div class="bg-gray-700 p-4 rounded">
                    <h3 class="font-bold text-green-400">Endpoint Analysis</h3>
                    <p class="text-sm text-gray-400">Benchmark against similar trials</p>
                </div>
                <div class="bg-gray-700 p-4 rounded">
                    <h3 class="font-bold text-purple-400">Enrollment Forecast</h3>
                    <p class="text-sm text-gray-400">Predict timelines accurately</p>
                </div>
            </div>
            <p class="mt-6 text-gray-400">Analyzing 500,000+ clinical trials</p>
        </div>
    </div>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
