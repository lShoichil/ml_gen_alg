import numpy as np

rng = np.random.default_rng()


def fitness(population, size_population, n, m, complexity, coeffs, time):
    sum_time_for_developers = np.zeros((size_population, m))
    for indIdx, individ in enumerate(population):
        individ_result = coeffs[individ - 1, complexity - 1] * time
        sum_time_for_developers[indIdx] = np.bincount(individ - 1, individ_result, minlength=m)
    max_time = np.max(sum_time_for_developers, axis=1)
    return max_time


def calc_score(time):
    return 1e6 / np.min(time)


def selection(population, size_selection, n, m, complexity, coeffs, time):
    fitness_values = fitness(population, len(population), n, m, complexity, coeffs, time)
    selected_indices = np.argsort(fitness_values)[:size_selection]
    selected = population[selected_indices]
    return selected


def crossover(selected, size_population, size_selection, n):
    parent_ids = rng.integers(0, size_selection, size=(size_population - size_selection, 2))
    crossover_points = rng.integers(-1, n - 3, size=(size_population - size_selection,))
    gen_mask_first_parent = np.arange(n) >= crossover_points[..., np.newaxis]
    gen_mask_second_parent = np.arange(n) <= crossover_points[..., np.newaxis] + n + 2
    childs = np.where(gen_mask_first_parent == gen_mask_second_parent,
                      selected[parent_ids[:, 0]], selected[parent_ids[:, 1]])
    return childs


def mutation(childs, p_mutation_ind, p_mutation_gen, m, n):
    mut_childs_mask = rng.choice(2, p=(1 - p_mutation_ind, p_mutation_ind), size=len(childs)) > 0
    generate_new_gen_childs = rng.integers(1, m + 1, size=(mut_childs_mask.sum(), n))
    gen_childs_mask = rng.random(size=generate_new_gen_childs.shape) <= p_mutation_gen
    childs[mut_childs_mask] = np.where(gen_childs_mask, generate_new_gen_childs, childs[mut_childs_mask])
    return childs


def step_by_step(population, size_population, size_selection, n, m, complexity, coeffs, time, p_mutation_ind,
                 p_mutation_gen):
    selected = selection(population, size_selection, n, m, complexity, coeffs, time)
    childs = crossover(selected, size_population, size_selection, n)
    childs = mutation(childs, p_mutation_ind, p_mutation_gen, m, n)
    population = np.concatenate([selected, childs], axis=0)
    return population


def genetic_algorithm(n, complexity, time, m, coeffs,
                      size_population, size_selection, p_mutation_ind, p_mutation_gen):
    population = np.random.default_rng().integers(1, m + 1, size=(size_population, n))
    best_fitness = np.inf
    best_population = None

    for i in range(100):
        fit = fitness(population, size_population, n, m, complexity, coeffs, time)
        score = calc_score(fit)

        if np.min(fit) < best_fitness:
            best_fitness = np.min(fit)
            best_population = population[np.argmin(fit)]

        print(f"fitness: {np.round(np.min(fit), 3)};\nscore: {np.round(score, 5)};\niteration: {i}\n")

        population = step_by_step(population, size_population, size_selection, n, m, complexity, coeffs, time,
                                  p_mutation_ind,
                                  p_mutation_gen)

    return best_population


def solve_genetic_algorithm(input_file, output_file):
    with open(input_file, 'r') as input_file:
        n = int(input_file.readline())
        complexity = np.array(list(map(int, input_file.readline().split())))
        time = np.array(list(map(float, input_file.readline().split())))

        m = int(input_file.readline())
        coeffs = np.array([list(map(float, line.split())) for line in input_file])

    population = genetic_algorithm(n, complexity, time, m, coeffs, 5000, 75, 0.7, 0.03)

    with open(output_file, 'w') as output_file:
        for task in population:
            print(task, file=output_file, end=' ')


solve_genetic_algorithm('input.txt', 'output.txt')
