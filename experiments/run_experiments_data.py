"""Shared synthetic-corpus generator (extracted verbatim from run_experiments.py)."""
import random
import numpy as np
import pandas as pd

TOPICS = {
    "python_install": [
        "How do I install Python 3.11 on Ubuntu?",
        "You need to add the deadsnakes PPA first.",
        "Added PPA – do I need to update the cache?",
        "Yes, run sudo apt update before installing.",
        "Got it working, thanks!",
        "What version are you using exactly?",
        "Python 3.11.4 is the latest stable.",
    ],
    "segfault_debug": [
        "My terminal crashes when I run a shell script – any help?",
        "It shows segmentation fault after a few seconds.",
        "That usually means a memory issue in your script.",
        "Try running it under valgrind to find the leak.",
        "Disabling GPU acceleration in the script fixed it!",
        "Good catch – GPU memory can cause that on some drivers.",
    ],
    "vim_config": [
        "How do I set vim to use 4-space indentation?",
        "Add 'set tabstop=4 shiftwidth=4 expandtab' to your .vimrc",
        "Where is the .vimrc file?",
        "It's in your home directory – ~/.vimrc",
        "Perfect, that worked.",
        "You can also use a plugin like vim-sleuth to auto-detect.",
    ],
    "git_rebase": [
        "I messed up a rebase – how do I abort it?",
        "Run git rebase --abort to get back to where you started.",
        "It says 'no rebase in progress'.",
        "Then you already completed it. Use git reflog to find the old HEAD.",
        "Found it – git reset --hard HEAD@{3} worked.",
        "Great. Always make a backup branch before rebasing.",
    ],
    "docker_network": [
        "How do containers communicate in Docker?",
        "By default they're on the bridge network.",
        "How do I connect two containers?",
        "Use docker-compose with a shared network, or docker network create.",
        "Thanks – got them talking via a custom bridge.",
        "You can also use container names as hostnames within the same network.",
    ],
    "ssl_cert": [
        "Getting SSL certificate error in curl.",
        "Try curl -k to skip verification temporarily.",
        "What's the right permanent fix?",
        "Install the CA cert: sudo update-ca-certificates",
        "That fixed it – thanks.",
    ],
    "screen_resolution": [
        "Screen resolution not changing in Ubuntu 22.04.",
        "Try xrandr to see available modes.",
        "It doesn't list my monitor's native resolution.",
        "You may need to create a custom mode with cvt and xrandr --newmode.",
        "That worked after a restart – appreciate it.",
    ],
    "pip_conflict": [
        "pip install fails with dependency conflict.",
        "Try creating a virtual environment first: python -m venv myenv",
        "What about conda?",
        "Conda works too – conda create -n myenv python=3.10",
        "Went with venv – cleaner for my use case.",
        "Good choice for lightweight projects.",
    ],
}

USERS = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"]


def generate_irc_session(n_threads=5, interleave_factor=2):
    """Generate a realistic interleaved IRC session with known thread labels."""
    topics = random.sample(list(TOPICS.keys()), min(n_threads, len(TOPICS)))
    threads = {t: TOPICS[t].copy() for t in topics}

    # Assign users to threads (multiple users per thread)
    thread_users = {}
    for t in topics:
        users = random.sample(USERS, min(3, len(USERS)))
        thread_users[t] = users

    # Build interleaved message sequence
    messages = []
    queues = {t: list(enumerate(msgs)) for t, msgs in threads.items()}
    active = list(topics)
    msg_id = 0

    while any(queues[t] for t in active):
        # Pick thread to advance (weighted by remaining messages)
        weights = [len(queues[t]) for t in active]
        if sum(weights) == 0:
            break
        chosen = random.choices(active, weights=weights, k=1)[0]
        if not queues[chosen]:
            continue

        idx, text = queues[chosen].pop(0)
        user = thread_users[chosen][idx % len(thread_users[chosen])]

        messages.append({
            "msg_id": msg_id,
            "user": user,
            "text": text,
            "thread_id": chosen,
            "thread_idx": topics.index(chosen),
        })
        msg_id += 1

        # Occasionally inject a burst of replies from the same thread
        if random.random() < 0.3 and queues[chosen]:
            idx2, text2 = queues[chosen].pop(0)
            user2 = thread_users[chosen][idx2 % len(thread_users[chosen])]
            messages.append({
                "msg_id": msg_id,
                "user": user2,
                "text": text2,
                "thread_id": chosen,
                "thread_idx": topics.index(chosen),
            })
            msg_id += 1

    return messages, topics


def build_corpus(n_sessions=40, threads_per_session=5):
    """Build a corpus of IRC sessions."""
    all_messages = []
    session_id = 0
    for _ in range(n_sessions):
        msgs, topics = generate_irc_session(n_threads=threads_per_session)
        for m in msgs:
            m["session_id"] = session_id
            m["global_thread_id"] = f"s{session_id}_{m['thread_id']}"
        all_messages.extend(msgs)
        session_id += 1
    return pd.DataFrame(all_messages)
