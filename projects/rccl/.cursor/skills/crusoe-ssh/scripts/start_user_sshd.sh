#!/usr/bin/env bash
# Start a user-space sshd listening on port 22222 on the current host.
# Run this ON the allocated compute node (not the login node).
#
# Env:
#   SSHD_PORT   listen port (default: 22222)
#   SSHD_DIR    state dir (default: /tmp/${USER}-sshd-${SLURM_JOB_ID:-manual})
#   ALLOW_USER  AllowUsers value (default: $USER)

set -euo pipefail

: "${SSHD_PORT:=22222}"
: "${SSHD_DIR:=/tmp/${USER}-sshd-${SLURM_JOB_ID:-manual}}"
: "${ALLOW_USER:=${USER}}"

if [[ ! -x /usr/sbin/sshd ]]; then
  echo "error: /usr/sbin/sshd not found" >&2
  exit 1
fi

mkdir -p "${SSHD_DIR}"
chmod 700 "${SSHD_DIR}"

if [[ ! -f "${SSHD_DIR}/ssh_host_ed25519_key" ]]; then
  ssh-keygen -t ed25519 -f "${SSHD_DIR}/ssh_host_ed25519_key" -N '' -q
fi

mkdir -p "${HOME}/.ssh"
chmod 700 "${HOME}/.ssh"
if [[ -f "${HOME}/.ssh/id_ed25519.pub" ]]; then
  touch "${HOME}/.ssh/authorized_keys"
  chmod 600 "${HOME}/.ssh/authorized_keys"
  if ! grep -qxF "$(cat "${HOME}/.ssh/id_ed25519.pub")" "${HOME}/.ssh/authorized_keys" 2>/dev/null; then
    cat "${HOME}/.ssh/id_ed25519.pub" >> "${HOME}/.ssh/authorized_keys"
  fi
fi

cat > "${SSHD_DIR}/sshd_config" << EOF
Port ${SSHD_PORT}
ListenAddress 0.0.0.0
HostKey ${SSHD_DIR}/ssh_host_ed25519_key
PidFile ${SSHD_DIR}/sshd.pid
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no
PubkeyAuthentication yes
ChallengeResponseAuthentication no
KbdInteractiveAuthentication no
UsePAM no
PermitRootLogin no
AllowUsers ${ALLOW_USER}
PrintMotd no
EOF

if [[ -f "${SSHD_DIR}/sshd.pid" ]] && kill -0 "$(cat "${SSHD_DIR}/sshd.pid")" 2>/dev/null; then
  echo "sshd already running pid=$(cat "${SSHD_DIR}/sshd.pid") port=${SSHD_PORT} host=$(hostname)"
  ss -ltn 2>/dev/null | grep -E ":${SSHD_PORT}\\b" || true
  exit 0
fi

/usr/sbin/sshd -f "${SSHD_DIR}/sshd_config" -E "${SSHD_DIR}/sshd.log"
sleep 0.3

if [[ ! -f "${SSHD_DIR}/sshd.pid" ]] || ! kill -0 "$(cat "${SSHD_DIR}/sshd.pid")" 2>/dev/null; then
  echo "error: sshd failed to start; log:" >&2
  tail -n 40 "${SSHD_DIR}/sshd.log" >&2 || true
  exit 1
fi

echo "started pid=$(cat "${SSHD_DIR}/sshd.pid") port=${SSHD_PORT} host=$(hostname)"
ss -ltn 2>/dev/null | grep -E ":${SSHD_PORT}\\b" || true
echo "connect: ssh -p ${SSHD_PORT} -i ~/.ssh/id_ed25519 $(hostname)"
