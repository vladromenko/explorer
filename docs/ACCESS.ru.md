# Доступ к Explorer

Фактические секреты не хранятся в Git. Локальная приватная памятка создаётся в `data/ACCESS-PRIVATE.ru.md` и имеет права `0600`.

- Web и телефон используют один ключ из `/home/vlad/Explorer/config/access_token`. Посмотреть ссылку: `bin/explorer url` или `bin/explorer mobile-url`.
- SSH: пользователь `vlad`, адрес `explorer.local` или Tailscale IP. Пароль Linux нельзя прочитать из системы в открытом виде; используется пароль, заданный владельцем, либо SSH-ключ.
- Telegram не имеет отдельного пользовательского пароля. Доступ задаётся allowlist в `config/telegram.json`; bot token находится в `~/.config/explorer/secrets/telegram-token`.
- Tailscale использует вход в tailnet-аккаунт, отдельного пароля робота нет.
- AmneziaWG использует приватные ключи в `config/vpn/amnezia.conf`, а не пароль web-интерфейса.
- GitHub использует SSH-ключ аккаунта на машине.

После публикации токена в сообщении или журнале его следует заменить у BotFather и обновить локальный файл секрета.
