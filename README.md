# AutoClassNotes


python -m venv venv
venv\Scripts\activate

python.exe -m pip install --upgrade pip
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt


Ejecutar en PowerShell como admin en Windows:
- Ejecutar para poder ejecutar los comandos: Set-ExecutionPolicy RemoteSigned
- Ir al path del proyecto
- Activar entorno: venv\Scripts\activate
- Ejecutar:
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
- Ejecutar:
choco install ffmpeg



Modelos:
WHISPER: https://huggingface.co/openai/whisper-large-v3
LLAMA-2 todos: https://huggingface.co/meta-llama/
Llama-2-7b-chat-hf: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 
