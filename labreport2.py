import random
from typing import List, Dict, Tuple

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# BSD2513 / BSD3513 - Lab Report 2
# Genetic Algorithm: 80-bit chromosome, target = 50 ones
# ============================================================

POPULATION_SIZE = 300
CHROMOSOME_LENGTH = 80
TARGET_ONES = 50
MAX_FITNESS = 80
GENERATIONS = 50

# GA operator settings (not specified in the lab sheet)
CROSSOVER_RATE = 0.80
MUTATION_RATE = 1.0 / CHROMOSOME_LENGTH
TOURNAMENT_SIZE = 3
ELITE_COUNT = 2


def create_individual() -> List[int]:
    """Create one random binary chromosome of length 80."""
    return [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]


def count_ones(chromosome: List[int]) -> int:
    return sum(chromosome)


def fitness(chromosome: List[int]) -> int:
    """
    Assumed fitness formula because the lab sheet does not print
    the actual preceding formula:

        fitness = 80 - abs(number_of_ones - 50)

    Therefore, when the chromosome contains exactly 50 ones,
    fitness = 80 (the maximum).
    """
    ones = count_ones(chromosome)
    return MAX_FITNESS - abs(ones - TARGET_ONES)


def create_population() -> List[List[int]]:
    return [create_individual() for _ in range(POPULATION_SIZE)]


def tournament_selection(
    population: List[List[int]], scores: List[int]
) -> List[int]:
    """Select the fittest chromosome from a random tournament."""
    candidates = random.sample(range(len(population)), TOURNAMENT_SIZE)
    winner = max(candidates, key=lambda index: scores[index])
    return population[winner][:]


def crossover(parent1: List[int], parent2: List[int]) -> Tuple[List[int], List[int]]:
    """Apply one-point crossover."""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    point = random.randint(1, CHROMOSOME_LENGTH - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome: List[int]) -> List[int]:
    """Flip each bit with the configured mutation probability."""
    mutated = chromosome[:]
    for index in range(CHROMOSOME_LENGTH):
        if random.random() < MUTATION_RATE:
            mutated[index] = 1 - mutated[index]
    return mutated


def run_genetic_algorithm(seed: int = 42):
    random.seed(seed)
    population = create_population()
    history = []
    best_overall = None
    best_overall_fitness = -1

    for generation in range(GENERATIONS + 1):
        scores = [fitness(individual) for individual in population]
        best_index = max(range(len(population)), key=lambda i: scores[i])
        generation_best = population[best_index][:]
        generation_best_fitness = scores[best_index]

        if generation_best_fitness > best_overall_fitness:
            best_overall = generation_best[:]
            best_overall_fitness = generation_best_fitness

        history.append({
            "Generation": generation,
            "Best Fitness": generation_best_fitness,
            "Average Fitness": sum(scores) / len(scores),
            "Worst Fitness": min(scores),
            "Best Number of Ones": count_ones(generation_best),
        })

        # The initial population is Generation 0. Create the next
        # population after recording each generation, up to Generation 50.
        if generation == GENERATIONS:
            break

        # Elitism: carry the best chromosomes into the next generation.
        ranked_indices = sorted(
            range(len(population)), key=lambda i: scores[i], reverse=True
        )
        new_population = [
            population[index][:] for index in ranked_indices[:ELITE_COUNT]
        ]

        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population, scores)
            parent2 = tournament_selection(population, scores)
            child1, child2 = crossover(parent1, parent2)

            new_population.append(mutate(child1))
            if len(new_population) < POPULATION_SIZE:
                new_population.append(mutate(child2))

        population = new_population

    history_df = pd.DataFrame(history)
    best_string = "".join(str(bit) for bit in best_overall)

    return best_overall, best_overall_fitness, history_df, best_string


# -------------------------- Streamlit UI --------------------------

st.set_page_config(
    page_title="BSD2513 | Genetic Algorithm",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Genetic Algorithm Simulator")
st.caption(
    "BSD2513 / BSD3513 Artificial Intelligence · Lab Report 2 · "
    "Binary chromosome optimization"
)

st.markdown(
    """
This application uses a Genetic Algorithm to find an 80-bit chromosome
containing **exactly 50 ones**. The required population size is **300**
and the algorithm runs for **50 generations**.
"""
)

with st.expander("Lab parameters and algorithm settings", expanded=True):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Population size", POPULATION_SIZE)
    c2.metric("Chromosome length", CHROMOSOME_LENGTH)
    c3.metric("Target number of ones", TARGET_ONES)
    c4.metric("Generations", GENERATIONS)

    st.write("**GA operator settings** (selected implementation values):")
    st.write(
        f"- Selection: Tournament selection (size {TOURNAMENT_SIZE})  \n"
        f"- Crossover: One-point crossover (rate {CROSSOVER_RATE:.0%})  \n"
        f"- Mutation: Bit-flip mutation (rate {MUTATION_RATE:.4f} per bit)  \n"
        f"- Elitism: {ELITE_COUNT} individuals carried to the next generation"
    )

st.warning(
    "Fitness-function assumption: the supplied lab sheet states that "
    "50 ones must produce a maximum fitness of 80, but does not show the "
    "actual formula. This app uses fitness = 80 - abs(number_of_ones - 50). "
    "Confirm this formula with your lecturer if a different preceding "
    "formula was provided in class."
)

seed = st.number_input(
    "Random seed (use the same seed to reproduce the same result)",
    min_value=0,
    max_value=999999,
    value=42,
    step=1
)

if st.button("▶ Run Genetic Algorithm", type="primary", use_container_width=True):
    with st.spinner("Running the Genetic Algorithm for 50 generations..."):
        best_chromosome, best_score, history_df, best_string = (
            run_genetic_algorithm(int(seed))
        )

    st.session_state["ga_best_chromosome"] = best_chromosome
    st.session_state["ga_best_score"] = best_score
    st.session_state["ga_history"] = history_df
    st.session_state["ga_best_string"] = best_string

if "ga_history" in st.session_state:
    history_df = st.session_state["ga_history"]
    best_chromosome = st.session_state["ga_best_chromosome"]
    best_score = st.session_state["ga_best_score"]
    best_string = st.session_state["ga_best_string"]

    st.divider()
    st.header("Results")

    final_ones = count_ones(best_chromosome)
    r1, r2, r3 = st.columns(3)
    r1.metric("Best fitness found", f"{best_score} / {MAX_FITNESS}")
    r2.metric("Number of ones", f"{final_ones} / {TARGET_ONES}")
    r3.metric("Generations completed", GENERATIONS)

    if final_ones == TARGET_ONES:
        st.success("Target achieved: the chromosome contains exactly 50 ones.")
    else:
        st.info(
            "The target was not reached in this run. The best chromosome "
            "is displayed below. Try another random seed and run again."
        )

    st.subheader("Best chromosome")
    st.code(best_string, language=None)
    st.write(
        f"**Chromosome length:** {len(best_chromosome)} bits  ·  "
        f"**Number of zeros:** {CHROMOSOME_LENGTH - final_ones}  ·  "
        f"**Number of ones:** {final_ones}"
    )

    st.subheader("Fitness progress across generations")
    chart_data = history_df.set_index("Generation")[
        ["Best Fitness", "Average Fitness"]
    ]
    st.line_chart(chart_data, y=["Best Fitness", "Average Fitness"])

    st.subheader("Generation summary")
    st.dataframe(
        history_df,
        hide_index=True,
        use_container_width=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Download generation results (CSV)",
        data=csv_data,
        file_name="genetic_algorithm_generation_results.csv",
        mime="text/csv"
    )

    chromosome_data = (
        "Best chromosome: " + best_string + "\n"
        + f"Number of ones: {final_ones}\n"
        + f"Fitness: {best_score}/{MAX_FITNESS}\n"
        + f"Random seed: {seed}\n"
    )
    st.download_button(
        "⬇ Download best chromosome (TXT)",
        data=chromosome_data,
        file_name="best_chromosome.txt",
        mime="text/plain"
    )

    st.subheader("How to interpret the result")
    st.markdown(
        """
- **Best Fitness** is the highest fitness value in each generation.
- **Average Fitness** is the mean fitness of all 300 chromosomes.
- A chromosome with exactly **50 ones** receives fitness **80** under
  the assumed formula.
- Elitism preserves the two best chromosomes when forming the next
  generation, while selection, crossover and mutation create new
  offspring.
"""
    )
else:
    st.info("Click **Run Genetic Algorithm** to generate the population and results.")

st.divider()
st.caption(
    "Reproducibility: the random seed and GA operator settings are shown "
    "above. The same seed and settings produce the same run."
)
