# Startup Steps

## sudo user
```sh
adduser sam
usermod -aG sudo sam
visudo # interactive, change to %sudo ALL=(ALL:ALL) NOPASSWD: ALL
su - sam
whoami
```

## user remote ssh

```sh
mkdir -p /home/sam/.ssh
chmod 700 /home/sam/.ssh

# Public key
vim /home/sam/.ssh/authorized_keys
chmod 600 /home/sam/authorized_keys
chown -R sam:sam /home/sam/.ssh
```

## code
```sh
ssh-keygen
cat ~/.ssh/id_edjfdk.pub
# paste into github settings

mkdir Code/
git clone git@github.com:afiaka87/raw-pytorch-sdxl.git
```

## uvx b2 authorize and download data
```sh
uvx b2 account authorize
mkdir Data/ && uvx b2 sync b2://dalle-blog-data/ Data/
```


## weights download
```sh
cd ~/Code/raw-pytorch-sdxl/
uv add "huggingface_hub[cli]"
hf --help
hf auth login
# Download LoRA checkpoint
mkdir Checkpoints/ && uvx b2 sync b2://dalle-blog-sdxl/ Checkpoints/
# Download SDXL full weights
bash download_weights.sh
```
