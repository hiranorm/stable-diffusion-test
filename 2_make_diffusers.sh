source venv/bin/activate
git clone https://github.com/ShivamShrirao/diffusers
cd diffusers/examples/dreambooth/
pip install -r requirements.txt
pip install diffusers
pip install bitsandbytes
# 学習させる画像を入れる
mkdir inputs