$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Test-Path -LiteralPath '.runtime/python.exe') {
    & ./.runtime/python.exe app.py
} elseif (Test-Path -LiteralPath '.venv/Scripts/python.exe') {
    & ./.venv/Scripts/python.exe app.py
} else {
    Write-Error 'Install Python and dependencies using README.md first.'
}
