@echo off
for %%e in (*.tar) do (
"C:\Program Files\7-Zip\"7z.exe x -o%%~ne %%e
)
