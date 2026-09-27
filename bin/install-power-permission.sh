#!/bin/bash
# Run locally with sudo once. Only a root-owned, argument-free shutdown helper
# is delegated; no user-writable scripts may run as root.
set -euo pipefail
if [ "$(id -u)" != 0 ]; then
  echo 'Administrator privileges are required to install controlled shutdown.' >&2
  exit 1
fi
install -d -m 0755 /usr/local/sbin
cat > /usr/local/sbin/explorer-poweroff <<'HELPER'
#!/bin/sh
set -eu
[ "$#" = 0 ] || exit 2
/bin/sync
exec /usr/bin/systemctl poweroff
HELPER
chown root:root /usr/local/sbin/explorer-poweroff
chmod 0755 /usr/local/sbin/explorer-poweroff
temporary=$(mktemp)
trap 'rm -f "$temporary"' EXIT
echo 'vlad ALL=(root) NOPASSWD: /usr/local/sbin/explorer-poweroff ""' > "$temporary"
visudo -cf "$temporary"
install -o root -g root -m 0440 "$temporary" /etc/sudoers.d/explorer-poweroff
