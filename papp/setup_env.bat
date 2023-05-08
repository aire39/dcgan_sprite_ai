python -m venv env
call .\env\Scripts\activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install matplotlib
python -m pip install numpy
pip install ipython


echo "Test import torch..."
python --version
python -c "import torch;x = torch.rand(5, 3);print(x)"
echo "complete!"

echo "setup complete! run env\Scripts\activate.bat to load environment"
echo "env\Scripts\activate.bat" > activate_torch_env.bat
echo "$PSScriptRoot\env\Scripts\activate.ps1" > activate_torch_env.ps1

deactivate
