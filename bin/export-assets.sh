#!/bin/bash
set -euo pipefail
destination="${1:?Укажите пустую папку назначения}"
mkdir -p "$destination"
cd /home/vlad/Explorer
rsync -a --prune-empty-dirs \
  --include='/models/***' --include='/vendor/***' --exclude='*' ./ "$destination"/
cp config/components.json "$destination/components.json"
echo "Экспорт готов: $destination. В нём нет data/, журналов, кадров, VPN и токенов."
