# Compose
# sudo docker compose up --force-recreate

# Compose, updating all files
# sudo docker compose up --force-recreate --build

# Compose, remove lod container versions
sudo docker compose up --build -d --remove-orphans --force-recreate
