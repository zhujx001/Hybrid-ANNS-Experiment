# Milvus environment installation and script testing

## Environment installation

### Use Docker Compose intall Milvus

```
wget https://github.com/milvus-io/milvus/releases/download/v2.5.6/milvus-standalone-docker-compose.yml -O docker-compose.yml

sudo docker compose up -d
```



### Check if the Docker container is running normally.

```bash
sudo docker ps
# the containers named milvus-standalone, milvus-minio, and milvus-etcd start.
```



## Script execution



### Install Python Environment

The Python version is 3.8.10

Download some necessary Python libraries

```
pip install pymilvus
```

Each dataset corresponds to a directory, and the test scripts are divided into a single-threaded test script (including table creation and index creation) and a multi-threaded test script.

### Run script

Change the host in the script to the IP of your own Milvs-standalone container. 
The paths for the dataset and labels also need to be changed to the paths where you have installed them.