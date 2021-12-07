# IMG App

Select a img and find segments whit the model.

## Requirements

- Docker
- Python 3.7
- pipreqs

## Generate requirements.txt
```
pipreqs --force
```

## Generate docker image
```
sudo docker build -t img-app .
```

## Run a container 
```
sudo docker run -d -p 8000:8000 img-app
```
