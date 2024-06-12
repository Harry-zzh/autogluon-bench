# Download zip dataset from Google Drive
filename='Meta-Transformer_base_patch16_encoder.pth'
fileid='19ahcN2QKknkir_bayhTW5rucuAiX0OXq'
# 链接格式：https://drive.google.com/file/d/{fileid}/view?usp=drive_link
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt