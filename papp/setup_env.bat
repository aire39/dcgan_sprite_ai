python -m venv env
call .\env\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

echo "Test import torch..."
python --version
python -c "import torch;x = torch.rand(5, 3);print(x)"
echo "complete!"

echo "setup complete! run env\Scripts\activate.bat to load environment"
echo "env\Scripts\activate.bat" > activate_torch_env.bat

deactivate
