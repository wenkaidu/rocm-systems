---
name: crusoe-ssh
description: >-
  Start a user-space sshd on port 22222 inside a Crusoe/SPUR Slurm allocation so
  Cursor Remote-SSH (or plain ssh) can land on the allocated compute node.
  Use when host :22 rejects pubkey, when the agent must not run on the login
  node, or when the user asks to ssh / open Cursor on an allocated node
  (crsuse2-m2m-*, crs-m2m-cpu-spur-*).
disable-model-invocation: true
---

# Crusoe SSH (user sshd on allocated node)

On Crusoe Slurm login nodes (`crs-m2m-cpu-spur-*`), Cursor Remote-SSH and the
agent run on the **login** host. Compute-node host `sshd` on `:22` often
**rejects** user pubkeys (`Permission denied (publickey)`). Start a **user
`sshd` on port 22222** inside the allocation, then connect to that port.

Script: [scripts/start_user_sshd.sh](scripts/start_user_sshd.sh).

## Prerequisites

- Active Slurm/SPUR allocation (`squeue -u "$USER"` shows `R` and a nodelist).
- Shell **on the allocated node** (`srun --jobid=<JOBID> --overlap --pty bash`,
  or an existing node shell). Do **not** start `sshd` on the login node.
- `/usr/sbin/sshd` present on the compute image.
- `~/.ssh/id_ed25519` (or another key) and matching line in
  `~/.ssh/authorized_keys` (shared `/home` is enough once set on login).

## Steps

### 1. Ensure your pubkey is authorized

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh
cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### 2. Start user sshd on 22222

On the **allocated node**:

```bash
DIR=/tmp/${USER}-sshd-${SLURM_JOB_ID:-manual}
mkdir -p "$DIR" && chmod 700 "$DIR"
ssh-keygen -t ed25519 -f "$DIR/ssh_host_ed25519_key" -N '' -q

cat > "$DIR/sshd_config" << EOF
Port 22222
ListenAddress 0.0.0.0
HostKey $DIR/ssh_host_ed25519_key
PidFile $DIR/sshd.pid
AuthorizedKeysFile .ssh/authorized_keys
PasswordAuthentication no
PubkeyAuthentication yes
UsePAM no
PermitRootLogin no
AllowUsers $USER
EOF

/usr/sbin/sshd -f "$DIR/sshd_config" -E "$DIR/sshd.log"
ss -ltn | grep 22222
```

Or run the bundled script (same behavior; job-scoped dir under `/tmp`):

```bash
# already on the node:
bash .cursor/skills/crusoe-ssh/scripts/start_user_sshd.sh

# from login, attach to an existing job (example):
srun --jobid="$JOBID" --overlap bash \
  "$HOME/rocm-systems/projects/rccl/.cursor/skills/crusoe-ssh/scripts/start_user_sshd.sh"
```

### 3. Connect

From the login node (or via ProxyJump from your laptop):

```bash
NODE=$(squeue -h -j "$JOBID" -o '%N' | head -1)   # or scontrol show hostnames …
ssh -p 22222 -i ~/.ssh/id_ed25519 "$NODE"
```

Example:

```bash
ssh -p 22222 -i ~/.ssh/id_ed25519 crsuse2-m2m-185
```

### 4. Point Cursor at the compute node (optional)

Add a host to the SSH config used by Cursor Remote-SSH, then
**Remote-SSH: Connect to Host…** so the agent runs on the node, not login:

```sshconfig
Host crsuse2-m2m-185
  HostName crsuse2-m2m-185
  User wenkaidu
  Port 22222
  IdentityFile ~/.ssh/id_ed25519
```

If connecting from outside the cluster, add `ProxyJump <login-or-bastion>`.

## Notes

- User `sshd` dies when the allocation ends; restart after each new job.
- Prefer `srun --jobid=… --overlap` over host `:22` when pubkey auth fails.
- Login default shell may be `csh`/`tcsh` — run the blocks under `/bin/bash`.
- Do not build RCCL or heavy work on the login node; use the allocation.
