@echo off
git add .
git commit -m "Fix Vercel deployment: remove environment variable references and use SQLite"
git push
pause
