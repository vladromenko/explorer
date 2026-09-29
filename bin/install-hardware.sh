#!/bin/bash
set -euo pipefail
serial="${EXPLORER_MCU_SERIAL:-}"
if [ -z "$serial" ]; then
  mapfile -t found < <(udevadm info --export-db | awk -F= '/^E: ID_VENDOR_ID=10c4$/{v=1} v&&/^E: ID_MODEL_ID=ea60$/{m=1} v&&m&&/^E: ID_SERIAL_SHORT=/{print $2;v=m=0}' | sort -u)
  if [ "${#found[@]}" != 1 ]; then
    echo 'Укажите серийный номер: EXPLORER_MCU_SERIAL=... bin/install-hardware.sh' >&2; exit 2
  fi
  serial="${found[0]}"
fi
rule=$(sed "s/@SERIAL@/$serial/g" /home/vlad/Explorer/config/99-explorer-mcu.rules.in)
printf '%s\n' "$rule" | sudo tee /etc/udev/rules.d/99-explorer-mcu.rules >/dev/null
sudo usermod -aG dialout vlad
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "Настроен /dev/explorer_mcu для CP210x $serial. После первого добавления в dialout войдите в систему заново."
