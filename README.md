# Closer-Challenge
Closer Project Challenge on Insurance Data

Table of content
- [Closer-Challenge](#closer-challenge)
- [Goal](#goal)
  - [Version Control](#version-control)
  - [Setup git](#setup-git)
  - [Setup github](#setup-github)
  - [Pull repository](#pull-repository)
- [Conda Environment](#conda-environment)
# Goal

Taken from the slides:
```
A Sense Assurance, uma empresa no sector dos seguros, tem vindo a crescer e apresenta, hoje, um volume significativo de clientes. 

O departamento de Marketing procura compreender melhor os diferentes perfis de clientes que possuem e contratou a Closer para ajudar. 

O vosso objetivo é desenvolver uma segmentação dos clientes de forma a que o departamento de Marketing possa ter um melhor entendimento dos diferentes perfis de clientes que a empresa possui. 

No final deverão apresentar a vossa proposta ao cliente.
```

## Version Control
## Setup git

Download: https://git-scm.com/download/win

Config:
```bash
git config --global user.name "FIRST_NAME LAST_NAME" 

git config --global user.email "MY_NAME@example.com"
```

## Setup github

First:
```bash
conda install gh --channel conda-forge
```

Second:
```bash
gh auth login
```

Steps:   
? What account do you want to log into? GitHub.com   
? What is your preferred protocol for Git operations? HTTPS   
? Authenticate Git with your GitHub credentials? Yes     
? How would you like to authenticate GitHub CLI? Login witha web browser   

It opens a page in the browser, we put the one-time code, and login with our account.  
All should be set now.

## Pull repository

Write in the terminal:
```bash
gh repo clone Portugapt/Closer-Challenge
```

# Conda Environment

https://stackoverflow.com/questions/59657306/invalidarchiveerror-when-executing-conda-install-notebook
> P.S.: Make sure you installed Anaconda or Miniconda into a directory that contains only 7-bit ASCII characters and no spaces, such as C:\anaconda!

To do that: 
https://stackoverflow.com/questions/58131555/how-to-change-the-path-of-conda-base
```bash
conda config --prepend pkgs_dirs C:/.conda/pkgs
conda config --prepend envs_dirs C:/.conda/envs
```

Then create the conda environment
```bash
conda env create -f environment.yaml
```

