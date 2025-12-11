Write-Host "=== Step 1: Downloading BFG Repo-Cleaner ===" -ForegroundColor Cyan

# Create a local variable for the download path
$bfgPath = "$PWD\bfg.jar"

# Download BFG if it doesn't exist
if (-Not (Test-Path $bfgPath)) {
    Write-Host "Downloading BFG Repo-Cleaner..."
    Invoke-WebRequest -Uri "https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar" -OutFile $bfgPath
    Write-Host "BFG downloaded successfully."
} else {
    Write-Host "BFG already exists, skipping download."
}

# Check if Java is installed
if (-Not (Get-Command java -ErrorAction SilentlyContinue)) {
    Write-Host "Java not found! Installing OpenJDK (Temurin 17)..." -ForegroundColor Yellow
    winget install EclipseAdoptium.Temurin.17.JDK
}

Write-Host "`n=== Step 2: Cleaning large files from history ===" -ForegroundColor Cyan

# Run BFG to remove large files/folders
java -jar $bfgPath --delete-folders ".venv"
java -jar $bfgPath --delete-files "*.dll"
java -jar $bfgPath --delete-files "*.h5"
java -jar $bfgPath --delete-files "*.npy"
java -jar $bfgPath --delete-files "*.keras"

Write-Host "`n=== Step 3: Running Git garbage collection ===" -ForegroundColor Cyan
git reflog expire --expire=now --all
git gc --prune=now --aggressive

Write-Host "`n=== Step 4: Ensuring .gitignore exists and is correct ===" -ForegroundColor Cyan

# Create or update .gitignore
@"
# Ignore virtual environments
.venv/
venv/

# Ignore large data and model files
data/
data_out/
models_out/

# Ignore Python cache, logs, and binaries
__pycache__/
*.pyc
*.pyo
*.pyd
*.dll
*.h5
*.npy
*.keras
*.log
"@ | Out-File -Encoding UTF8 .gitignore

git add .gitignore
git commit -m "Ensure .gitignore excludes large and environment files" --allow-empty

Write-Host "`n=== Step 5: Final force push to GitHub ===" -ForegroundColor Cyan
git push origin main --force

Write-Host "`n All done! Your repository has been cleaned and pushed successfully!" -ForegroundColor Green
