@echo off
echo Stopping the local RAG stack...

:: 1. Stop the Blazor UI
:: Kills the dotnet process and the underlying compiled app executable
echo Stopping Blazor frontend...
taskkill /F /IM dotnet.exe /T > nul 2>&1

:: 2. Stop the FastAPI Backend
:: Kills the invisible pythonw process running orchestrator.py
echo Stopping FastAPI backend...
taskkill /F /IM pythonw.exe /T > nul 2>&1

:: 3. Stop Ollama
:: Kills the local Ollama server process
echo Stopping Ollama...
taskkill /F /IM ollama.exe /T > nul 2>&1
taskkill /F /IM ollama_llama_server.exe /T > nul 2>&1

echo.
echo All services stopped successfully.
pause