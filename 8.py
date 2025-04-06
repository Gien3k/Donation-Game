import time
import os
import sys
import argparse
import random
import numpy as np
import csv
from numba import njit
import multiprocessing as mp
from tqdm import tqdm

# Constants for donations
B = 1.0  # Benefit for recipient
C = 0.1  # Cost for donor



@njit
def tick_base(n, m, q, strategies, imageScores, rewards):
    """
    Model base – brak szumu, donor nie aktualizuje swojej własnej reputacji.
    Aktualizacje reputacji dokonują jedynie obserwatorzy (dla i != donor).
    """
    for _ in range(m):
        donor = np.random.randint(0, n)
        recipient = np.random.randint(0, n-1)
        if recipient >= donor:
            recipient += 1

        if imageScores[donor, recipient] >= strategies[donor]:
            # Donacja: nadanie payoffów
            rewards[donor]     -= C
            rewards[recipient] += B
            # Aktualizacja reputacji u obserwatorów (donor pomijany)
            for i in range(n):
                if i == donor:
                    continue
                if q < 1.0:
                    if np.random.random() < q:
                        imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                else:
                    imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
        else:
            # Defection: brak zmian payoffów
            for i in range(n):
                if i == donor:
                    continue
                if q < 1.0:
                    if np.random.random() < q:
                        imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                else:
                    imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)

    return strategies, imageScores, rewards


@njit
def tick_noisy(n, m, q, strategies, imageScores, rewards, ea, ep):
    """
    Model noisy – donor nie aktualizuje swojej własnej reputacji.
    Wprowadzony jest action noise (ea) i perception noise (ep).
    """
    for _ in range(m):
        donor = np.random.randint(0, n)
        recipient = np.random.randint(0, n-1)
        if recipient >= donor:
            recipient += 1

        intend = (imageScores[donor, recipient] >= strategies[donor])
        # Action noise: jeśli donor chciał donować, ale szum powoduje zmianę intencji
        if intend and ea > 0 and np.random.random() < ea:
            intend = False

        if intend:
            rewards[donor]     -= C
            rewards[recipient] += B
            # Aktualizacja reputacji u obserwatorów:
            for i in range(n):
                if i == donor:
                    continue
                if i == recipient:
                    imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                else:
                    if q < 1.0:
                        if np.random.random() < q:
                            if ep > 0 and np.random.random() < ep:
                                imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                            else:
                                imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                    else:
                        if ep > 0 and np.random.random() < ep:
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                        else:
                            imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
        else:
            # Defection – brak payoffów dla defection
            for i in range(n):
                if i == donor:
                    continue
                if i == recipient:
                    imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                else:
                    if q < 1.0:
                        if np.random.random() < q:
                            if ep > 0 and np.random.random() < ep:
                                imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                            else:
                                imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                    else:
                        if ep > 0 and np.random.random() < ep:
                            imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                        else:
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)

    return strategies, imageScores, rewards

@njit
def cooperateImageUpdate_g(n, donor, recipient, q, ep, g1, imageScores):
    if q < 1.0:
        for i in range(n):
            if i == recipient:
                imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
            else:
                if i != donor and np.random.random() < q:
                    if ep > 0.0 and np.random.random() < ep:
                        if np.random.random() < g1:
                            imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                        else:
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                    else:
                        imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
    else:
        for i in range(n):
            if i == recipient:
                imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
            elif i != donor:
                if ep > 0.0 and np.random.random() < ep:
                    if np.random.random() < g1:
                        imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                    else:
                        imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                else:
                    imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)


@njit
def defectImageUpdate_g(n, donor, recipient, q, ep, g1, imageScores):
    if q < 1.0:
        for i in range(n):
            if i == recipient:
                imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
            else:
                if i != donor and np.random.random() < q:
                    if ep > 0.0 and np.random.random() < ep:
                        imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                    else:
                        if np.random.random() < g1:
                            imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                        else:
                            if ep > 0.0:
                                imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                                imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                            else:
                                imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
    else:
        for i in range(n):
            if i == recipient:
                imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
            elif i != donor:
                if ep > 0.0 and np.random.random() < ep:
                    imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                else:
                    if np.random.random() < g1:
                        imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                    else:
                        if ep > 0.0:
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                        else:
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)

@njit
def tick_generosity(n, m, q, strategies, imageScores, rewards, ea, ep, g1, g2):
    if g1 == 0.0 and g2 == 0.0:
        return tick_noisy(n, m, q, strategies, imageScores, rewards, ea, ep)

    for _ in range(m):
        donor = np.random.randint(0, n)
        recipient = donor
        while recipient == donor:
            recipient = np.random.randint(0, n)
        imageScore = imageScores[donor, recipient]
        intend = (imageScore >= strategies[donor])
        if intend and ea > 0.0 and np.random.random() < ea:
            intend = False
        if (not intend) and (ep > 0.0) and (np.random.random() < g2):
            intend = True
        if intend:
            rewards[donor] -= C
            rewards[recipient] += B
            cooperateImageUpdate_g(n, donor, recipient, q, ep, g1, imageScores)
        else:
            defectImageUpdate_g(n, donor, recipient, q, ep, g1, imageScores)
    return strategies, imageScores, rewards

@njit
def tick_forgiveness(n, m, q, strategies, imageScores, rewards, ea, ep,
                     forgiveness_strategies, fa, fr):
    """
    Model forgiveness – donor aktualizuje również swoją własną reputację,
    zgodnie z logiką z oryginalnego kodu Java (ForgivenessDonationGame).
    """
    for _ in range(m):
        donor = np.random.randint(0, n)
        recipient = np.random.randint(0, n-1)
        if recipient >= donor:
            recipient += 1

        intend = (imageScores[donor, recipient] >= strategies[donor])
        if intend and ea > 0 and np.random.random() < ea:
            intend = False
        if (not intend) and (fa == 1):
            probf = np.exp(-((-imageScores[donor, recipient] + 5) / forgiveness_strategies[donor]))
            if np.random.random() < probf:
                intend = True

        if intend:
            rewards[donor]     -= C
            rewards[recipient] += B
            # Aktualizacja reputacji – donor i wszyscy obserwatorzy:
            for i in range(n):
                # Donor i recipient zawsze aktualizują (bez szumu):
                if i == donor or i == recipient:
                    imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                else:
                    if q < 1.0 and np.random.random() >= q:
                        continue
                    perceived = True
                    if ep > 0 and np.random.random() < ep:
                        perceived = not perceived
                    if perceived:
                        imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                    else:
                        if fr == 1:
                            sc = imageScores[i, donor]
                            probf = np.exp(-((-sc + 5) / forgiveness_strategies[i]))
                            if np.random.random() >= probf:
                                imageScores[i, donor] = max(sc - 1.0, -5.0)
                        else:
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
        else:
            # W przypadku defection – donor i recipient aktualizują swój obraz:
            for i in range(n):
                if i == donor or i == recipient:
                    imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)
                else:
                    if q < 1.0 and np.random.random() >= q:
                        continue
                    perceived = False
                    if ep > 0 and np.random.random() < ep:
                        perceived = not perceived
                    if perceived:
                        imageScores[i, donor] = min(imageScores[i, donor] + 1.0, 5.0)
                    else:
                        if fr == 1:
                            sc = imageScores[i, donor]
                            probf = np.exp(-((-sc + 5) / forgiveness_strategies[i]))
                            if np.random.random() >= probf:
                                imageScores[i, donor] = max(sc - 1.0, -5.0)
                        else:
                            imageScores[i, donor] = max(imageScores[i, donor] - 1.0, -5.0)

    return strategies, imageScores, rewards

@njit
def run_simulation(model, n, m, q, mr, ea, ep, generations, g1, g2,
                   forgiveness_strategies, fa, fr):
    """
    Główna pętla symulacji:
      - Inicjalizacja strategii, imageScores i rewards (wszystkie zerowane).
      - W każdej generacji wykonanie tick(...) dla wybranego modelu.
      - Selekcja ruletkowa + mutacja.
      - Reset rewards i imageScores.
      - Zwracana jest średnia wartość rewardów z generacji.
    """
    strategies = np.empty(n, dtype=np.int32)
    for i in range(n):
        strategies[i] = np.random.randint(-5, 7)
    imageScores = np.zeros((n, n), dtype=np.float64)
    rewards = np.zeros(n, dtype=np.float64)
    reward_averages = np.zeros(generations, dtype=np.float64)

    for g in range(generations):
        if model == 0:
            strategies, imageScores, rewards = tick_base(n, m, q, strategies, imageScores, rewards)
        elif model == 1:
            strategies, imageScores, rewards = tick_noisy(n, m, q, strategies, imageScores, rewards, ea, ep)
        elif model == 2:
            strategies, imageScores, rewards = tick_generosity(n, m, q, strategies, imageScores, rewards, ea, ep, g1, g2)
        elif model == 3:
            strategies, imageScores, rewards = tick_forgiveness(n, m, q, strategies, imageScores, rewards, ea, ep, forgiveness_strategies, fa, fr)

        reward_averages[g] = rewards.mean()

        # Selekcja ruletkowa
        scaled = rewards.copy()
        mn = scaled.min()
        if mn < 0:
            scaled = scaled - mn
        if scaled.min() == 0:
            scaled += 0.1
        tot = scaled.sum()
        new_strategies = strategies.copy()
        for i in range(n):
            r_val = np.random.random() * tot
            s = 0.0
            for j in range(n):
                s += scaled[j]
                if s >= r_val:
                    new_strategies[i] = strategies[j]
                    break
        strategies = new_strategies

        # Mutacja
        for i in range(n):
            if np.random.random() < mr:
                newv = np.random.randint(-5, 7)
                while newv == strategies[i]:
                    newv = np.random.randint(-5, 7)
                strategies[i] = newv

        # Reset
        rewards[:] = 0.0
        imageScores[:] = 0.0

    return reward_averages.mean()




def simulation_worker(params):
    (model, q, ea, ep, generations, n, m, mr, g1, g2, fflag, seed) = params
    np.random.seed(seed)
    if model == 3:
        arr = np.array([0.001, 0.5, 1.0, 1.355, 1.67])
        forg = np.empty(n, dtype=np.float64)
        for i in range(n):
            forg[i] = arr[np.random.randint(0, 5)]
    else:
        forg = np.zeros(n, dtype=np.float64)
    fa = fflag[0]
    fr = fflag[1]
    avg = run_simulation(model, n, m, q, mr, ea, ep, generations, g1, g2, forg, fa, fr)
    return (q, ea, ep, g1, g2, avg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=300)
    parser.add_argument("--mutation", type=float, default=0.001)
    parser.add_argument("--q_values", nargs="+", type=float, default=[1.0])
    parser.add_argument("--ea_values", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    parser.add_argument("--ep_values", nargs="+", type=float, default=[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    parser.add_argument("--generations", type=int, default=100000)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--generosity", action="store_true")
    parser.add_argument("-g1", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    parser.add_argument("-g2", nargs="+", type=float, default=[0.0])
    parser.add_argument("--forgiveness_action", action="store_true")
    parser.add_argument("--forgiveness_reputation", action="store_true")
    parser.add_argument("--preventNegativePayoffs", action="store_true",
                        help="Enables offset +0.1 for each interaction")
    args = parser.parse_args()

    n = args.size
    m = args.pairs
    mr = args.mutation
    q_vals = args.q_values
    ea_vals = args.ea_values
    ep_vals = args.ep_values
    runs = args.runs
    generations = args.generations
    fflag = (1 if args.forgiveness_action else 0,
             1 if args.forgiveness_reputation else 0)

    if len(ea_vals) != len(ep_vals):
        sys.exit("Error: --ea_values and --ep_values must have the same number of elements.")

    # Create task list
    tasks = []
    seed_base = 12345
    for q in q_vals:
        for i in range(len(ea_vals)):
            ea = ea_vals[i]
            ep = ep_vals[i]
            for g1 in args.g1:
                for g2 in args.g2:
                    for run in range(runs):
                        seed = seed_base + hash((q, ea, ep, g1, g2, run)) % 999999
                        tasks.append((args.model, q, ea, ep, generations, n, m, mr, g1, g2, fflag, seed))

    print(f"Number of simulations: {len(tasks)}")
    num_workers = max(1, mp.cpu_count() - 2)
    pool = mp.Pool(processes=num_workers)
    results = []

    start_time = time.time()
    with tqdm(total=len(tasks), desc="Simulations") as pbar:
        for i, out in enumerate(pool.imap_unordered(simulation_worker, tasks)):
            results.append(out)
            pbar.update(1)

            # Percentage of completed runs
            tasks_done = i + 1
            total_tasks = len(tasks)
            progress = int(100 * tasks_done / total_tasks)

            # Time
            elapsed = time.time() - start_time
            time_per_task = elapsed / tasks_done
            tasks_left = total_tasks - tasks_done
            time_remaining = time_per_task * tasks_left

            # Output to stdout
            print(f"PROGRESS: {progress}", flush=True)
            print(f"INFO: Completed {tasks_done}/{total_tasks} runs", flush=True)
            print(f"INFO: Elapsed {elapsed:.1f}s, estimated remain {time_remaining:.1f}s", flush=True)

    pool.close()
    pool.join()

    # Aggregate results and write to file
    data = {}
    for (q, ea, ep, g1, g2, avg) in results:
        key = (q, ea, ep, g1, g2)
        data.setdefault(key, []).append(avg)

    rows = [["q", "ea", "ep", "g1", "g2", "avg", "std", "runs"]]
    for k in sorted(data.keys()):
        arr = data[k]
        mean_ = np.mean(arr)
        std_ = np.std(arr)
        rows.append([k[0], k[1], k[2], k[3], k[4], mean_, std_, len(arr)])
        print(f"q={k[0]}, ea={k[1]:.3f}, ep={k[2]:.3f}, g1={k[3]:.3f}, g2={k[4]:.3f} -> avg={mean_:.3f} std={std_:.3f} runs={len(arr)}")

    with open(args.output, "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerows(rows)

if __name__ == "__main__":
    main()

