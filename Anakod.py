#!/usr/bin/env python3
"""
CPU Scheduling Simulator

Usage:
  python3 scheduler.py --input case1.csv [--quantum 4] [--output out] [--threads]

Assumptions:
- Input CSV columns (header required): pid,arrival,burst,priority
- arrival, burst can be integers or floats. priority: lower number = higher priority.
- Context switch time is 0.001 time units (used for efficiency calculation and counted per switch).

This script implements: FCFS, Preemptive SJF (SRTF), Non-Preemptive SJF, Round Robin,
Preemptive Priority, Non-Preemptive Priority. Each algorithm writes its own result files.
"""

import argparse
import csv
import os
import threading
from collections import deque
import math


CONTEXT_SWITCH = 0.001


class Process:
    def __init__(self, pid, arrival, burst, priority=0):
        self.pid = str(pid)
        self.arrival = float(arrival)
        self.burst = float(burst)
        self.remaining = float(burst)
        self.priority = int(priority)
        self.start_time = None
        self.completion_time = None

    def copy(self):
        p = Process(self.pid, self.arrival, self.burst, self.priority)
        return p


def read_csv(path):
    procs = []
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = r.get('pid') or r.get('id') or r.get('PID')
            arrival = r.get('arrival') or r.get('arrive') or r.get('Arrival')
            burst = r.get('burst') or r.get('cpu') or r.get('Burst')
            priority = r.get('priority') or r.get('Priority') or 0
            if pid is None or arrival is None or burst is None:
                raise ValueError('CSV must contain headers pid, arrival, burst (priority optional)')
            procs.append(Process(pid, float(arrival), float(burst), int(priority)))
    return procs


def write_timeline(path, timeline):
    # timeline: list of (start, end, pid or 'IDLE')
    with open(path, 'w') as f:
        for seg in timeline:
            s, e, pid = seg
            f.write(f"[ {format_time(s)} ] - - {pid} - - [ {format_time(e)} ]\n")


def format_time(t):
    # show integer if whole, else 3 decimals
    if abs(t - round(t)) < 1e-9:
        return str(int(round(t)))
    return f"{t:.3f}"


def calc_stats(processes, timeline, context_switches):
    # processes: list of Process with completion_time set
    n = len(processes)
    waiting = []
    turnaround = []
    total_busy = 0.0
    for p in processes:
        tat = p.completion_time - p.arrival
        wt = tat - p.burst
        waiting.append(wt)
        turnaround.append(tat)
        total_busy += p.burst

    max_wt = max(waiting) if waiting else 0
    avg_wt = sum(waiting) / n if n else 0
    max_tat = max(turnaround) if turnaround else 0
    avg_tat = sum(turnaround) / n if n else 0
    makespan = max((p.completion_time for p in processes), default=0)

    # Throughput for T = [50, 100, 150, 200]: count processes with completion_time <= T
    throughputs = {}
    for T in [50, 100, 150, 200]:
        throughputs[T] = sum(1 for p in processes if p.completion_time <= T)

    # CPU efficiency: total_busy / (makespan + context_switches*CONTEXT_SWITCH)
    denom = makespan + context_switches * CONTEXT_SWITCH
    cpu_eff = (total_busy / denom) if denom > 0 else 0

    return {
        'max_waiting': max_wt,
        'avg_waiting': avg_wt,
        'max_turnaround': max_tat,
        'avg_turnaround': avg_tat,
        'throughput': throughputs,
        'cpu_efficiency': cpu_eff,
        'makespan': makespan,
        'total_busy': total_busy,
    }


def write_stats(path, stats, context_switches):
    with open(path, 'w') as f:
        f.write(f"Max Waiting Time: {stats['max_waiting']:.3f}\n")
        f.write(f"Avg Waiting Time: {stats['avg_waiting']:.3f}\n")
        f.write(f"Max Turnaround Time: {stats['max_turnaround']:.3f}\n")
        f.write(f"Avg Turnaround Time: {stats['avg_turnaround']:.3f}\n")
        f.write("Throughput (T -> completed processes):\n")
        for T, v in stats['throughput'].items():
            f.write(f"  {T}: {v}\n")
        f.write(f"CPU Efficiency: {stats['cpu_efficiency']*100:.3f} %\n")
        f.write(f"Total Context Switches: {context_switches}\n")
        f.write(f"Makespan: {stats['makespan']:.3f}\n")


def run_algorithm(name, procs_in, output_dir, quantum=4, preemptive=True):
    procs = [p.copy() for p in procs_in]
    procs.sort(key=lambda p: (p.arrival, p.pid))
    timeline = []
    time = 0.0
    context_switches = 0
    last_pid = None

    if name == 'FCFS':
        queue = deque(sorted(procs, key=lambda p: (p.arrival, p.pid)))
        ready = deque()
        while queue or ready:
            # move arrivals
            while queue and queue[0].arrival <= time:
                ready.append(queue.popleft())
            if not ready:
                # idle until next arrival
                if queue:
                    next_t = queue[0].arrival
                    timeline.append((time, next_t, 'IDLE'))
                    time = next_t
                    continue
                else:
                    break
            p = ready.popleft()
            if last_pid is not None and last_pid != p.pid:
                context_switches += 1
                time += CONTEXT_SWITCH
            start = time if time >= p.arrival else p.arrival
            if p.start_time is None:
                p.start_time = start
            end = start + p.burst
            timeline.append((start, end, p.pid))
            time = end
            p.completion_time = end
            last_pid = p.pid

    elif name == 'NonPreemptiveSJF':
        remaining = procs[:]
        ready = []
        while remaining or ready:
            # fill ready
            for p in remaining[:]:
                if p.arrival <= time:
                    ready.append(p)
                    remaining.remove(p)
            if not ready:
                if remaining:
                    next_t = min(p.arrival for p in remaining)
                    timeline.append((time, next_t, 'IDLE'))
                    time = next_t
                    continue
                else:
                    break
            # choose shortest burst
            ready.sort(key=lambda p: (p.burst, p.arrival, p.pid))
            p = ready.pop(0)
            if last_pid is not None and last_pid != p.pid:
                context_switches += 1
                time += CONTEXT_SWITCH
            start = max(time, p.arrival)
            p.start_time = p.start_time or start
            end = start + p.burst
            timeline.append((start, end, p.pid))
            time = end
            p.completion_time = end
            last_pid = p.pid

    elif name == 'PreemptiveSJF':
        # SRTF
        remaining = procs[:]
        ready = []
        current = None
        while remaining or ready or current:
            # move arrivals
            arrivals = [p for p in remaining if p.arrival <= time]
            for p in arrivals:
                ready.append(p)
                remaining.remove(p)
            if current is None and not ready:
                if remaining:
                    next_t = min(p.arrival for p in remaining)
                    timeline.append((time, next_t, 'IDLE'))
                    time = next_t
                    continue
                else:
                    break
            # pick shortest remaining among ready and current
            if current:
                candidates = ready + [current]
            else:
                candidates = ready[:]
            candidates.sort(key=lambda p: (p.remaining, p.arrival, p.pid))
            chosen = candidates[0]
            if chosen is current:
                # run until either completion or next arrival that may preempt
                if remaining:
                    next_arrival = min(p.arrival for p in remaining)
                else:
                    next_arrival = math.inf
                run_until = min(time + current.remaining, next_arrival)
                if last_pid is not None and last_pid != current.pid:
                    context_switches += 1
                    time += CONTEXT_SWITCH
                start = time
                executed = run_until - time
                end = run_until
                timeline.append((start, end, current.pid))
                current.remaining -= executed
                time = end
                if abs(current.remaining) < 1e-9:
                    current.completion_time = time
                    current = None
                    last_pid = chosen.pid
                else:
                    last_pid = current.pid
            else:
                # switch to chosen
                if last_pid is not None and last_pid != chosen.pid:
                    context_switches += 1
                    time += CONTEXT_SWITCH
                # run chosen until either completion or next arrival
                if remaining:
                    next_arrival = min(p.arrival for p in remaining)
                else:
                    next_arrival = math.inf
                start = max(time, chosen.arrival)
                run_until = min(start + chosen.remaining, next_arrival)
                timeline.append((start, run_until, chosen.pid))
                executed = run_until - start
                chosen.remaining -= executed
                time = run_until
                if abs(chosen.remaining) < 1e-9:
                    chosen.completion_time = time
                    # if chosen was in ready remove
                    if chosen in ready:
                        ready.remove(chosen)
                    current = None
                    last_pid = chosen.pid
                else:
                    # preempted by arrival
                    if chosen in ready:
                        # already in ready
                        pass
                    else:
                        # make it current
                        current = chosen

    elif name == 'RoundRobin':
        q = float(quantum)
        remaining = procs[:]
        queue = deque()
        while remaining or queue:
            # enqueue arrivals
            for p in remaining[:]:
                if p.arrival <= time:
                    queue.append(p)
                    remaining.remove(p)
            if not queue:
                if remaining:
                    next_t = min(p.arrival for p in remaining)
                    timeline.append((time, next_t, 'IDLE'))
                    time = next_t
                    continue
                else:
                    break
            p = queue.popleft()
            if last_pid is not None and last_pid != p.pid:
                context_switches += 1
                time += CONTEXT_SWITCH
            start = max(time, p.arrival)
            run = min(q, p.remaining)
            timeline.append((start, start + run, p.pid))
            p.start_time = p.start_time or start
            time = start + run
            p.remaining -= run
            # enqueue arrivals that happened during run
            for r in remaining[:]:
                if r.arrival <= time:
                    queue.append(r)
                    remaining.remove(r)
            if p.remaining > 0:
                queue.append(p)
            else:
                p.completion_time = time
            last_pid = p.pid

    elif name == 'PreemptivePriority':
        # lower numerical priority value = higher priority
        remaining = procs[:]
        ready = []
        current = None
        while remaining or ready or current:
            arrivals = [p for p in remaining if p.arrival <= time]
            for p in arrivals:
                ready.append(p)
                remaining.remove(p)
            if current is None and not ready:
                if remaining:
                    next_t = min(p.arrival for p in remaining)
                    timeline.append((time, next_t, 'IDLE'))
                    time = next_t
                    continue
                else:
                    break
            if current:
                candidates = ready + [current]
            else:
                candidates = ready[:]
            candidates.sort(key=lambda p: (p.priority, p.arrival, p.pid))
            chosen = candidates[0]
            if chosen is current:
                # run until completion or next arrival
                if remaining:
                    next_arrival = min(p.arrival for p in remaining)
                else:
                    next_arrival = math.inf
                run_until = min(time + current.remaining, next_arrival)
                if last_pid is not None and last_pid != current.pid:
                    context_switches += 1
                    time += CONTEXT_SWITCH
                start = time
                executed = run_until - time
                timeline.append((start, run_until, current.pid))
                current.remaining -= executed
                time = run_until
                if abs(current.remaining) < 1e-9:
                    current.completion_time = time
                    current = None
                    last_pid = chosen.pid
                else:
                    last_pid = current.pid
            else:
                # switch to chosen
                if last_pid is not None and last_pid != chosen.pid:
                    context_switches += 1
                    time += CONTEXT_SWITCH
                if remaining:
                    next_arrival = min(p.arrival for p in remaining)
                else:
                    next_arrival = math.inf
                start = max(time, chosen.arrival)
                run_until = min(start + chosen.remaining, next_arrival)
                timeline.append((start, run_until, chosen.pid))
                executed = run_until - start
                chosen.remaining -= executed
                time = run_until
                if abs(chosen.remaining) < 1e-9:
                    chosen.completion_time = time
                    if chosen in ready:
                        ready.remove(chosen)
                    current = None
                    last_pid = chosen.pid
                else:
                    current = chosen

    elif name == 'NonPreemptivePriority':
        remaining = procs[:]
        ready = []
        while remaining or ready:
            for p in remaining[:]:
                if p.arrival <= time:
                    ready.append(p)
                    remaining.remove(p)
            if not ready:
                if remaining:
                    next_t = min(p.arrival for p in remaining)
                    timeline.append((time, next_t, 'IDLE'))
                    time = next_t
                    continue
                else:
                    break
            ready.sort(key=lambda p: (p.priority, p.arrival, p.pid))
            p = ready.pop(0)
            if last_pid is not None and last_pid != p.pid:
                context_switches += 1
                time += CONTEXT_SWITCH
            start = max(time, p.arrival)
            p.start_time = p.start_time or start
            end = start + p.burst
            timeline.append((start, end, p.pid))
            time = end
            p.completion_time = end
            last_pid = p.pid

    else:
        raise ValueError('Unknown algorithm')

    # make sure all processes have completion_time set (for algorithms that modified objects in place)
    for p in procs:
        if p.completion_time is None:
            # if never run, set completion to arrival
            p.completion_time = p.arrival

    stats = calc_stats(procs, timeline, context_switches)
    alg_dir = os.path.join(output_dir, name)
    os.makedirs(alg_dir, exist_ok=True)
    write_timeline(os.path.join(alg_dir, 'timeline.txt'), timeline)
    write_stats(os.path.join(alg_dir, 'stats.txt'), stats, context_switches)
    # also write per-process completion times
    with open(os.path.join(alg_dir, 'processes.csv'), 'w') as f:
        f.write('pid,arrival,burst,priority,completion\n')
        for p in procs:
            f.write(f"{p.pid},{p.arrival},{p.burst},{p.priority},{p.completion_time}\n")
    print(f"{name}: Done. Results in {alg_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', default='out')
    parser.add_argument('--quantum', '-q', default=4, type=float)
    parser.add_argument('--threads', action='store_true', help='run each algorithm concurrently (bonus)')
    args = parser.parse_args()

    procs = read_csv(args.input)
    algs = [
        'FCFS', 'PreemptiveSJF', 'NonPreemptiveSJF', 'RoundRobin', 'PreemptivePriority', 'NonPreemptivePriority'
    ]

    if args.threads:
        threads = []
        for a in algs:
            t = threading.Thread(target=run_algorithm, args=(a, procs, args.output, args.quantum))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    else:
        for a in algs:
            run_algorithm(a, procs, args.output, args.quantum)


if __name__ == '__main__':
    main()
