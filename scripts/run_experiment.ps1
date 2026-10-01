param([ValidateSet('smoke','full')][string]$Stage='smoke')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$taskPython=Join-Path $taskRoot '.venv\Scripts\python.exe'
$env:PYTHONUTF8='0'
$env:PYTHONIOENCODING='utf-8'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
Set-Location -LiteralPath $taskRoot
if(-not (Test-Path -LiteralPath $taskPython)) { throw 'Create the independent environment first; see README.md.' }
function Invoke-TaskPython {
    param([string[]]$Arguments)
    & $taskPython @Arguments
    if($LASTEXITCODE -ne 0) { throw "Python command failed: $Arguments" }
}
Invoke-TaskPython -Arguments @('scripts/check_data.py','--tokenizer')
Invoke-TaskPython -Arguments @('-m','unittest','discover','-s','tests','-v')
if($Stage -eq 'smoke') {
    Invoke-TaskPython -Arguments @('-m','src.train','--profile','smoke')
    Invoke-TaskPython -Arguments @('-m','src.inference','--verify-smoke')
} else {
    Invoke-TaskPython -Arguments @('-m','src.evaluate','--split','dev','--mode','zero')
    Invoke-TaskPython -Arguments @('-m','src.evaluate','--split','dev','--mode','few')
    Invoke-TaskPython -Arguments @('-m','src.train','--profile','formal')
    Invoke-TaskPython -Arguments @('-m','src.evaluate','--split','dev','--mode','finetuned')
    Invoke-TaskPython -Arguments @('-m','src.inference','--verify-formal')
    foreach($taskMode in @('zero','few','finetuned')) {
        Invoke-TaskPython -Arguments @('-m','src.evaluate','--split','test','--mode',$taskMode)
    }
    Invoke-TaskPython -Arguments @('scripts/build_report.py')
}
