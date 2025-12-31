mod breakout;
mod config;
mod neat;
mod serialization;
mod training;

use neat::population::Population;

fn main() {
    println!("Starting NEAT training on Breakout...");
    println!(
        "Using parallel processing with {} CPU cores",
        rayon::current_num_threads()
    );

    let mut population = Population::new();

    if std::env::args().any(|x| &x == "--prev") {
        let genome = match serialization::load_genome("best_genome.pb") {
            Ok(g) => {
                println!("Successfully loaded best genome!");
                g
            }
            Err(e) => {
                eprintln!("Failed to load genome: {}", e);
                eprintln!("Please run training first: cargo run --release --bin breakout-train");
                return;
            }
        };
        population.populate_from_genome(genome);
    } else {
        population.populate();
    }

    let config = config::Config::global();
    let mut num_generations = config.num_generations;
    let mut num_steps = config.num_steps;

    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == "--num-gens" {
            num_generations = args
                .next()
                .expect("Parameter for num-gen flag not given!")
                .parse()
                .expect("Invalid num-gen number!");
            println!("num-gens flag loaded successfully, num_gens: {num_generations}");
        }
        if arg == "--num-steps" {
             num_steps = args
                .next()
                .expect("Parameter for num-steps flag not given!")
                .parse()
                .expect("Invalid num-steps number!");
            println!("num-steps flag loaded successfully, num_steps: {num_steps}");
        }
    }
    let overall_start = std::time::Instant::now();

    for generation in 0..num_generations {
        println!(
            "\n=== Generation {}/{} ===",
            generation + 1,
            num_generations
        );

        // Train the population with statistics
        let stats = training::train_population_with_stats(&mut population.individuals, num_steps);

        println!("  Evaluation time: {:.2}s", stats.duration.as_secs_f32());
        println!(
            "  Evaluations/sec: {:.2}",
            stats.population_size as f32 / stats.duration.as_secs_f32()
        );
        println!("  Avg fitness: {:.2}", stats.avg_fitness);
        println!("  Min fitness: {:.2}", stats.min_fitness);
        println!("  Max fitness: {:.2}", stats.max_fitness);

        // Sort by fitness
        neat::population::sort_individuals_by_fitness(&mut population.individuals);

        // Get best individual
        if let Some(best) = population.individuals.first() {
            population.best = best.clone();
            println!("  Best genome ID: {}", best.genome.id);
            println!("  Best fitness: {:.2}", best.fitness);
        }

        // Reproduce for next generation
        if generation < num_generations - 1 {
            population.individuals = population.reproduce();
        }
    }

    let total_duration = overall_start.elapsed();

    println!("\n=== Training Complete! ===");
    println!("Total training time: {:.2}s", total_duration.as_secs_f32());
    println!(
        "Average time per generation: {:.2}s",
        total_duration.as_secs_f32() / num_generations as f32
    );
    println!("Best genome ID: {}", population.best.genome.id);
    println!("Best fitness: {:.2}", population.best.fitness);

    // Save the best individual
    if let Err(e) = serialization::save_individual(&population.best, "best_individual.pb") {
        eprintln!("Failed to save best individual: {}", e);
    }

    // Also save just the genome
    if let Err(e) = serialization::save_genome(&population.best.genome, "best_genome.pb") {
        eprintln!("Failed to save best genome: {}", e);
    }

    println!("\nResults saved to:");
    println!("  - best_individual.pb");
    println!("  - best_genome.pb");
}
