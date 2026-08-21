@echo off
echo Starting up the AI stack...

:: 1. Check if Ollama is running, start if not
curl -s http://localhost:11434 > nul
if errorlevel 1 (
    echo Starting Ollama background service...
    :: Starts the Ollama engine invisibly
    start /min "" ollama serve
    
    :: Wait for Ollama to fully boot up
    :waitollama
    curl -s http://localhost:11434 > nul
    if errorlevel 1 (
        timeout /t 1 /nobreak > nul
        goto waitollama
    )
)

:: 2. Start FastAPI backend silently using pythonw
cd local-rag-api
start "" pythonw orchestrator.py
cd ..

echo Waiting for FastAPI backend to start...

:: Poll FastAPI health check or root endpoint until it answers
:waitfastapi
curl -s http://localhost:8000 > nul
if errorlevel 1 (
    timeout /t 1 /nobreak > nul
    goto waitfastapi
)

:: 3. Start Blazor frontend (minimized)
cd LocalRagUI
start /min "" dotnet run
cd ..

echo Waiting for Blazor server to start...

:: 4. Poll the Blazor URL every 1 second until it answers
:waitblazor
curl -s http://localhost:5109 > nul
if errorlevel 1 (
    timeout /t 1 /nobreak > nul
    goto waitblazor
)

:: 5. Open the default web browser to the app
start http://localhost:5109

:: 6. Close this launcher window
exit