1. In the root level directory run 'poetry install' (Make sure you have poetry installed).
2. When working with notebooks make sure to use the poetry environment as a kernel for jupyter.

## Secrets env file

Create a secrets.env file with the following variables:

- QDRANT_URL
- QDRANT_API_KEY
- LLM_IP_ADDRESS

## Run docker

sudo docker compose --env-file secrets.env build

sudo docker compose --env-file secrets.env up

## Tutorial for the DB

http://localhost:6333/dashboard#/tutorial/
