#!/usr/bin/env python3
"""Shared queue: ENV=a FLEET_TARGET=host | launcher arguments.
STOP_<FLEET_HOST>/--deadline drains new claims; claims never auto-retry.
Inherited NGLAB_* are removed. root/--python (this interpreter by default)
supply NGLAB_ROOT/PY defaults; explicit job variables can override them."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time

POLL, GRACE, FREE_MIB = 60, 240, 1500
MIN_FREE_BYTES = 20 * 1024**3
LOCK_DIR = f"/tmp/nglab_fleet_gpu_locks_{os.getuid()}"


def gpus_free():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                          "--format=csv,noheader,nounits"], capture_output=True,
                         text=True, check=True, timeout=15).stdout
    return {int(a): int(b) for a, b in (line.split(",") for line in out.strip().splitlines())}


def jobs(path, log):
    with open(path) as stream:
        for number, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                envs, args = [x.strip() for x in line.split("|", 1)]
                dict(kv.split("=", 1) for kv in shlex.split(envs))
                shlex.split(args)
            except ValueError as error:
                log.write(f"{time.strftime('%F %T')} invalid job line {number}: {error}\n")
                continue
            yield hashlib.sha1(line.encode()).hexdigest()[:12], envs, args, line


def lock_file(name):
    os.makedirs(LOCK_DIR, mode=0o700, exist_ok=True)
    if os.stat(LOCK_DIR).st_dev != os.stat("/tmp").st_dev:
        raise OSError("GPU locks must stay on the /tmp filesystem")
    fd = os.open(os.path.join(LOCK_DIR, name), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


def stopped(qdir, host, deadline):
    return (deadline is not None and time.time() >= deadline
            or os.path.exists(os.path.join(qdir, f"STOP_{host}")))


def job_environment(envs, root, python):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NGLAB_")}
    env.update(NGLAB_ROOT=root, NGLAB_PY=python)
    env.update(kv.split("=", 1) for kv in shlex.split(envs))
    return env


def dispatch(qdir, root, launcher, host, only, deadline, python, active, last, log):
    if stopped(qdir, host, deadline):
        return
    free = gpus_free()
    for gpu, used in sorted(free.items()):
        if only is not None and gpu not in only:
            continue
        child = active.get(gpu)
        if child is not None and child.poll() is None:
            continue
        if used >= FREE_MIB or time.time() - last.get(gpu, 0) < GRACE:
            continue
        fd = lock_file(f"gpu{gpu}.lock")
        if fd is None:
            continue
        try:
            if gpus_free().get(gpu, FREE_MIB) >= FREE_MIB:
                continue
            for key, envs, args, line in jobs(os.path.join(qdir, "jobs.txt"), log):
                cdir = os.path.join(qdir, "claims", key)
                if os.path.exists(cdir):
                    continue
                job_vars = dict(kv.split("=", 1) for kv in shlex.split(envs))
                if job_vars.get("FLEET_TARGET", host) != host:
                    continue
                env = job_environment(envs, root, python)
                command = shlex.join(["bash", launcher, str(gpu), *shlex.split(args)])
                work = env["NGLAB_ROOT"]
                output = os.path.join(work, "data", "runs_fixed")
                paths = [root, work, qdir, output if os.path.isdir(output) else work]
                if any(shutil.disk_usage(p).free < MIN_FREE_BYTES for p in paths):
                    log.write(f"{time.strftime('%F %T')} disk below 20 GiB: {key}\n")
                    continue
                if stopped(qdir, host, deadline):
                    return
                try:
                    os.mkdir(cdir)
                except FileExistsError:
                    continue
                try:
                    with open("/proc/sys/kernel/random/boot_id") as stream:
                        boot_id = stream.read().strip()
                except FileNotFoundError:
                    boot_id = None
                meta = dict(line=line, host=host, gpu=gpu, start=time.strftime("%F %T"),
                            agent_pid=os.getpid(), boot_id=boot_id)
                quote = shlex.quote
                cmd = (f"cd {quote(root)} && {command} > {quote(cdir + '/launcher.log')} 2>&1; "
                       f"echo $? > {quote(cdir + '/rc')}; date '+%F %T' > {quote(cdir + '/end')}")
                with open(os.path.join(cdir, "claim.json"), "w") as stream:
                    json.dump(meta, stream)
                try:
                    child = subprocess.Popen(["bash", "-c", cmd], env=env, pass_fds=(fd,),
                                             start_new_session=True, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
                    active[gpu] = child
                    meta["wrapper_pid"] = child.pid
                except OSError as error:
                    meta["spawn_error"] = str(error)
                    with open(os.path.join(cdir, "rc"), "w") as stream:
                        stream.write("127\n")
                    with open(os.path.join(cdir, "end"), "w") as stream:
                        stream.write(time.strftime("%F %T") + "\n")
                with open(os.path.join(cdir, "claim.json"), "w") as stream:
                    json.dump(meta, stream)
                log.write(f"{time.strftime('%F %T')} gpu{gpu} <- {line} [{key}]\n")
                last[gpu] = time.time()
                break
        finally:
            # The wrapper inherits this open file description until its training exits.
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queue_dir")
    parser.add_argument("root")
    parser.add_argument("launcher")
    parser.add_argument("--gpus")
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.deadline is not None and not math.isfinite(args.deadline):
        parser.error("--deadline must be a finite Unix timestamp")
    qdir, root = os.path.abspath(args.queue_dir), os.path.abspath(args.root)
    only = {int(x) for x in args.gpus.split(",")} if args.gpus else None
    host = os.environ.get("FLEET_HOST", socket.gethostname())
    agent_key = hashlib.sha256((os.path.realpath(qdir) + host).encode()).hexdigest()[:16]
    agent_lock = lock_file(f"agent_{agent_key}.lock")
    if agent_lock is None:
        parser.exit(1, "Another agent already owns this queue and host\n")
    os.makedirs(os.path.join(qdir, "claims"), exist_ok=True)
    active, last = {}, {}
    try:
        with open(os.path.join(qdir, f"agent_{host}.log"), "a", buffering=1) as log:
            while not stopped(qdir, host, args.deadline):
                try:
                    dispatch(qdir, root, args.launcher, host, only, args.deadline,
                             args.python, active, last, log)
                except (OSError, ValueError, subprocess.SubprocessError) as error:
                    log.write(f"{time.strftime('%F %T')} dispatch failed: {error}\n")
                time.sleep(POLL)
            log.write(f"{time.strftime('%F %T')} drained, leaving active wrappers running\n")
    finally:
        os.close(agent_lock)


if __name__ == "__main__":
    main()
