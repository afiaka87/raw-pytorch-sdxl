# Startup Steps


## Sudo User
```sh
adduser sam
usermod -aG sudo sam
visudo # interactive, change to %sudo ALL=(ALL:ALL) NOPASSWD: ALL
su - sam
whoami
```

## SSH

```sh
mkdir -p /home/sam/.ssh
chmod 700 /home/sam/.ssh

# Public key
vim /home/sam/.ssh/authorized_keys
chmod 600 /home/sam/authorized_keys
chown -R sam:sam /home/sam/.ssh
```
mkdir Checkpoints/ && uvx b2 sync b2://dalle-blog-sdxl/ Checkpoints/
mkdir Code/
git clone git@github.com:afiaka87/raw-pytorch-sdxl.git
mkdir Data/ && uvx b2 sync b2://dalle-blog-data/ Data/



