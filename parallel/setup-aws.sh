sudo yum install -y python3 git screen htop
sudo pip3 install numpy pandas scipy

echo '
alias ls="ls --color"
alias l="ls -lh"
alias ll="ls -l"
alias lr="ls -lrth"
alias ..="cd .."
' >> ~/.bashrc
