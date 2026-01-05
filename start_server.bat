@echo off
REM Start Pokemon Showdown Server for BDSP Battle AI Training

echo ================================================
echo Starting BDSP Battle AI Training Server
echo ================================================
echo.
echo Server will be available at:
echo   - HTTP: http://127.0.0.1:8000
echo   - WebSocket: ws://127.0.0.1:8000/showdown/websocket
echo.
echo Press Ctrl+C to stop the server
echo ================================================
echo.

cd showdown-server
node pokemon-showdown start
