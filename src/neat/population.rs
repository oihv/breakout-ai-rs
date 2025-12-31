use rand::seq::IndexedRandom;
use super::crossover::crossover;
use super::mutation::mutate;

pub fn sort_individuals_by_fitness(individuals: &mut Vec<crate::neat::Individual>) {
    individuals.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
}

use crate::neat::Individual;
struct Population {
    individuals: Vec<Individual>,
    best: Individual
}

impl Population {
    pub fn populate(&mut self) {
        let config = crate::config::Config::global();

        for _ in 0..config.population_size {
            self.individuals.push(Individual{ genome: super::Genome::new(config.num_inputs, config.num_outputs), fitness: 0.0})
        }
    }

    pub fn reproduce(&mut self) -> Vec<Individual> {
        let config = crate::config::Config::global();

        let reproduction_cutoff = (config.survival_threshold * self.individuals.len() as f32).ceil() as usize;

        let mut survived_members:Vec<Individual> = Vec::new();
        for _ in 0..reproduction_cutoff {
            survived_members.push(self.individuals.pop().unwrap());
        }

        let mut new_gen:Vec<Individual> = Vec::new();
        let spawn_size = config.population_size;

        let mut rng = rand::rng();

        for _ in 0..spawn_size {
            let p1 = survived_members.choose(&mut rng).unwrap();
            let p2 = survived_members.choose(&mut rng).unwrap();
            let mut offspring = crossover(p1, p2);
            mutate(&mut offspring);
            new_gen.push(Individual{genome: offspring, fitness: 0.0});
        }

        new_gen
    }

    pub fn run(&mut self, compute_fitness: fn(&mut Vec<Individual>), num_generations: i32) {
        for _ in 0..num_generations {
            compute_fitness(&mut self.individuals);
            sort_individuals_by_fitness(&mut self.individuals);
            self.best = self.individuals.first().unwrap().clone();
            self.individuals = self.reproduce();
        }
    }
}
