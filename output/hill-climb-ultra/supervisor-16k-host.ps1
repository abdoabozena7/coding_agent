$ErrorActionPreference = "Stop"
$repoRoot = "D:\projects\Ai\coding_agent"
$labRoot = "D:\projects\Ai\coding_agent\projects\agent-harness-lab-01"
$inputPath = "D:\projects\Ai\coding_agent\output\hill-climb-ultra\supervisor-16k-input.txt"
$stdoutPath = "D:\projects\Ai\coding_agent\output\hill-climb-ultra\supervisor-16k.stdout.log"
$stderrPath = "D:\projects\Ai\coding_agent\output\hill-climb-ultra\supervisor-16k.stderr.log"
$pythonPath = "D:\projects\Ai\coding_agent\.venv\Scripts\python.exe"

$info = [System.Diagnostics.ProcessStartInfo]::new()
$info.FileName = $pythonPath
$info.WorkingDirectory = $labRoot
$info.UseShellExecute = $false
$info.CreateNoWindow = $true
$info.RedirectStandardInput = $true
$info.RedirectStandardOutput = $true
$info.RedirectStandardError = $true
$info.StandardOutputEncoding = [System.Text.Encoding]::UTF8
$info.StandardErrorEncoding = [System.Text.Encoding]::UTF8
$info.Arguments = (
    '-m agent --workspace "' + $labRoot +
    '" --session workspace-session-16k --provider ollama --model gemma4:e4b' +
    ' --mode ultra --permissions normal --plain --interactive'
)
$info.Environment["LLM_PROVIDER"] = "ollama"
$info.Environment["OLLAMA_MODEL"] = "gemma4:e4b"
$info.Environment["OLLAMA_NUM_GPU"] = "999"
$info.Environment["OLLAMA_CONTEXT_SIZE"] = "16384"
$info.Environment["AGENT_REQUIRE_LOCAL_GPU"] = "1"
$info.Environment["AGENT_REPOSITORY_INDEX_WARMUP_FILES"] = "0"
$info.Environment["AGENT_GLOBAL_MEMORY_PATH"] = "$repoRoot\output\hill-climb-ultra\isolated-global-lessons-16k.json"

[System.IO.File]::WriteAllText($stdoutPath, "")
[System.IO.File]::WriteAllText($stderrPath, "")
$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $info

$stdoutHandler = [System.Diagnostics.DataReceivedEventHandler]{
    param($sender, $eventArgs)
    if ($null -ne $eventArgs.Data) {
        [System.IO.File]::AppendAllText($stdoutPath, $eventArgs.Data + [Environment]::NewLine)
    }
}.GetNewClosure()
$stderrHandler = [System.Diagnostics.DataReceivedEventHandler]{
    param($sender, $eventArgs)
    if ($null -ne $eventArgs.Data) {
        [System.IO.File]::AppendAllText($stderrPath, $eventArgs.Data + [Environment]::NewLine)
    }
}.GetNewClosure()
$process.add_OutputDataReceived($stdoutHandler)
$process.add_ErrorDataReceived($stderrHandler)
[void]$process.Start()
[System.IO.File]::AppendAllText($stderrPath, "HOST_CHILD_PID=" + $process.Id + [Environment]::NewLine)
$process.BeginOutputReadLine()
$process.BeginErrorReadLine()

$cursor = 0
while (-not $process.HasExited) {
    $lines = @(Get-Content -LiteralPath $inputPath -Encoding UTF8)
    if ($lines.Count -gt $cursor) {
        foreach ($line in $lines[$cursor..($lines.Count - 1)]) {
            $process.StandardInput.WriteLine($line)
            $process.StandardInput.Flush()
        }
        $cursor = $lines.Count
    }
    Start-Sleep -Milliseconds 250
}
$process.WaitForExit()
[System.IO.File]::AppendAllText($stderrPath, "HOST_EXIT_CODE=" + $process.ExitCode + [Environment]::NewLine)
