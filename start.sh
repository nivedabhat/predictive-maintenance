docker ps               # to view running containers

docker-compose down -v --remove-orphans
docker container prune -f
docker volume prune -f
docker network prune -f
docker rmi $(docker images | predictive-maintenance-dev | awk '{print $3}')
docker-compose build --no-cache
docker-compose up --force-recreate
