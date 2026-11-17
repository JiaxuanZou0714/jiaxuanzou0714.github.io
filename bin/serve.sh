#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME=jiaxuan-blog
BUNDLE_VOLUME=jiaxuan-blog-bundle

if ! command -v container >/dev/null 2>&1; then
    echo "Install Apple container first: https://github.com/apple/container/releases" >&2
    exit 1
fi

container system start --enable-kernel-install

if ! container volume inspect "$BUNDLE_VOLUME" >/dev/null 2>&1; then
    container volume create --opt size=4G "$BUNDLE_VOLUME"
fi

if container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
    container stop "$CONTAINER_NAME"
    container delete "$CONTAINER_NAME"
fi

container run --detach \
    --name "$CONTAINER_NAME" \
    --cpus 2 --memory 2G \
    --publish 127.0.0.1:8080:8080 \
    --publish 127.0.0.1:35729:35729 \
    --volume "$PROJECT_ROOT:/srv/jekyll" \
    --volume "$BUNDLE_VOLUME:/usr/local/bundle" \
    --workdir /srv/jekyll \
    --env JEKYLL_ENV=development \
    --env BUNDLE_FROZEN=true \
    docker.io/amirpourmand/al-folio:v0.16.3 \
    /srv/jekyll/bin/entry_point.sh

echo "Jekyll is starting at http://localhost:8080"
echo "Follow startup progress: container logs --follow $CONTAINER_NAME"
