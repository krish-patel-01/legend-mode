#!/usr/bin/env bash
# Start the SearXNG instance backing Legend Mode's `web` tool family.
#
#   ./deploy/searxng/up.sh          start (or restart) it
#   ./deploy/searxng/up.sh check    just verify JSON search works
#
# Bound to 127.0.0.1 deliberately: nothing outside this machine has any business
# querying it, and it runs with the bot limiter off, which is only safe because of that.
set -euo pipefail

NAME=legend-searxng
PORT=8080
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

check() {
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 20 \
    "http://127.0.0.1:${PORT}/search?q=test&format=json" || echo 000)
  case "$code" in
    200) echo "ok: JSON search is working on ${PORT}" ;;
    403) echo "FAIL: 403 — either settings.yml is missing 'json' under search.formats,"
         echo "      or it never reached the container. Check both:"
         echo "        docker exec ${NAME} cat /etc/searxng/settings.yml | head"
         echo "        docker inspect ${NAME} --format '{{range .Mounts}}{{.Source}} -> {{.Destination}}{{end}}'"
         return 1 ;;
    000) echo "FAIL: nothing answering on ${PORT}" ; return 1 ;;
    *)   echo "FAIL: HTTP ${code}" ; return 1 ;;
  esac
}

if [ "${1:-}" = "check" ]; then check; exit $?; fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

# settings.yml is gitignored and generated from the template with a fresh key, so no
# literal secret is ever committed.
if [ ! -f "${HERE}/settings.yml" ]; then
  if [ ! -f "${HERE}/settings.yml.example" ]; then
    echo "missing both settings.yml and settings.yml.example in ${HERE}" >&2
    exit 1
  fi
  KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
  sed "s/GENERATED_BY_UP_SH/${KEY}/" "${HERE}/settings.yml.example" > "${HERE}/settings.yml"
  echo "wrote ${HERE}/settings.yml with a generated secret key"
fi

# The settings file is bind-mounted, and **if the source path does not exist Docker
# silently creates a directory there**. SearXNG then dies with exit 127 and no log
# output at all, which is exactly how the unrelated instance on this machine broke.
# Checking for a regular file specifically is what catches the directory case.
if [ ! -f "${HERE}/settings.yml" ]; then
  echo "${HERE}/settings.yml is not a regular file — refusing to start" >&2
  exit 1
fi

# **Both halves of the -v argument need protecting under Git Bash, and getting it wrong
# fails silently.** MSYS rewrites anything that looks like an absolute POSIX path, and it
# applies that to the *container* side too. With HERE=/k/Projects/... the mount came out as
#
#   K:\...\settings.yml;C  ->  \Program Files\Git\etc\searxng\settings.yml;ro
#
# Docker accepted it, mounted to that nonsense destination, and /etc/searxng quietly fell
# back to an anonymous volume holding SearXNG's 205-byte defaults. The container came up
# healthy and returned 403 to every JSON request — indistinguishable from a real config
# mistake, which is exactly what the error message here first blamed it on.
#
# MSYS_NO_PATHCONV stops the rewriting; cygpath gives Docker Desktop the Windows path it
# wants. Both are no-ops on Linux and macOS.
HOST_DIR="$HERE"
if command -v cygpath >/dev/null 2>&1; then
  HOST_DIR="$(cygpath -w "$HERE")"
fi

MSYS_NO_PATHCONV=1 docker run -d --name "$NAME" --restart unless-stopped \
  -p "127.0.0.1:${PORT}:8080" \
  -v "${HOST_DIR}/settings.yml:/etc/searxng/settings.yml:ro" \
  -e "INSTANCE_NAME=legend-mode" \
  -e "BASE_URL=http://127.0.0.1:${PORT}/" \
  searxng/searxng:latest

echo "waiting for it to come up..."
for _ in $(seq 1 45); do
  if check >/dev/null 2>&1; then
    check
    # Confirm the file actually arrived rather than trusting that the mount worked.
    # Checked here rather than right after `docker run`, because the container cannot
    # serve an exec for the first few seconds and a premature check reports a failure
    # that is not real.
    #
    # MSYS_NO_PATHCONV again, and for the same reason — the first version of this check
    # reported a broken mount on a perfectly good container, because Git Bash rewrote the
    # path in the *exec* into C:/Program Files/Git/etc/searxng/settings.yml. Any absolute
    # path handed to docker from this shell needs it, arguments and commands alike.
    if ! MSYS_NO_PATHCONV=1 docker exec "$NAME" \
        grep -q "json" /etc/searxng/settings.yml 2>/dev/null; then
      echo "WARNING: the mounted settings.yml is not the one in ${HERE}" >&2
    fi
    exit 0
  fi
  sleep 2
done
echo "did not come up in 90s; docker logs ${NAME}" >&2
exit 1
