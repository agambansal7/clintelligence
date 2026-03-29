"""
Vercel Serverless Entry Point
"""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

app = FastAPI(title="Clintelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Clintelligence | AI Protocol Intelligence</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white min-h-screen">
    <div class="container mx-auto px-4 py-16">
        <div class="text-center">
            <h1 class="text-5xl font-bold mb-4 bg-gradient-to-r from-orange-500 to-blue-500 bg-clip-text text-transparent">
                Clintelligence
            </h1>
            <p class="text-xl text-gray-300 mb-8">AI-Powered Clinical Trial Protocol Intelligence</p>
            <div class="bg-gray-800 rounded-lg p-8 max-w-2xl mx-auto">
                <h2 class="text-2xl font-semibold mb-4">Coming Soon</h2>
                <p class="text-gray-400 mb-6">
                    Our platform analyzes 500,000+ historical clinical trials to help you design better protocols.
                </p>
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
            </div>
            <p class="mt-8 text-gray-500">Contact: info@trialclintelligence.com</p>
        </div>
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML_TEMPLATE

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "clintelligence"}

handler = Mangum(app, lifespan="off")
