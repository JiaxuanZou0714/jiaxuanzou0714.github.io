#!/bin/bash
set -euo pipefail

CONFIG_FILE=_config.yml
BUNDLER_VERSION="$(tail -n 1 Gemfile.lock | tr -d '[:space:]')"

git config --global --add safe.directory "$PWD"
gem list --installed bundler --version "$BUNDLER_VERSION" >/dev/null || \
    gem install bundler --version "$BUNDLER_VERSION" --no-document
bundle "_${BUNDLER_VERSION}_" check || bundle "_${BUNDLER_VERSION}_" install

start_jekyll() {
    bundle "_${BUNDLER_VERSION}_" exec jekyll serve \
        --watch --port=8080 --host=0.0.0.0 --livereload --trace --force_polling &
    jekyll_pid=$!
}

stop_jekyll() {
    kill -TERM "$jekyll_pid" 2>/dev/null || true
    wait "$jekyll_pid" 2>/dev/null || true
}

trap 'exit 0' TERM INT
trap stop_jekyll EXIT

config_hash="$(sha256sum "$CONFIG_FILE")"
start_jekyll

while kill -0 "$jekyll_pid" 2>/dev/null; do
    sleep 2
    # Host edits on the shared filesystem do not reliably produce inotify events.
    if [[ -f "$CONFIG_FILE" ]]; then
        current_hash="$(sha256sum "$CONFIG_FILE")"
        if [[ "$current_hash" == "$config_hash" ]]; then
            continue
        fi
        config_hash="$current_hash"
        echo "$CONFIG_FILE changed; restarting Jekyll"
        stop_jekyll
        start_jekyll
    fi
done

wait "$jekyll_pid"
