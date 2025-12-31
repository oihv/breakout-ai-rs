use crate::breakout::{BreakoutEngine, engine::Action};
use crate::config;
use crate::neat::{Individual, nn::FeedForwardNeuralNetwork};
use rayon::prelude::*;

/// Evaluate a single individual on the breakout game
/// This function is thread-safe and can be called in parallel
pub fn evaluate_individual(individual: &Individual, num_steps: usize) -> f32 {
    let mut engine = BreakoutEngine::new();
    let mut network = FeedForwardNeuralNetwork::create_from_genome(&individual.genome);
    
    // Add randomness to make evaluation non-deterministic
    // Randomize initial ball direction
    use rand::Rng;
    let mut rng = rand::rng();
    let angle: f32 = rng.random_range(-0.5..0.5); // Vary angle by ±0.5 radians
    let speed = 6.5;
    engine.dx = speed * angle.cos();
    engine.dy = -speed * angle.sin().abs(); // Always start going up
    
    // Randomize initial platform position
    engine.platform_x = rng.random_range(5.0..(engine.scr_w - 5.0));
    
    // Auto-start the game
    engine.stick = false;
    
    // Run the game for a maximum number of frames or until game over
    let max_frames = num_steps;
    let delta = 1.0 / 60.0; // Assume 60 FPS
    
    for _ in 0..max_frames {
        if engine.game_over {
            break;
        }
        
        // Get current game state
        let state = engine.get_state();
        
        // Run neural network to get action
        let outputs = network.activate(state);
        
        // Determine action from network outputs
        // outputs[0] = left, outputs[1] = stay, outputs[2] = right
        let action = if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
            Action::Left
        } else if outputs[2] > outputs[1] && outputs[2] > outputs[0] {
            Action::Right
        } else {
            Action::Stay
        };
        
        // Step the game
        engine.step(action, delta);
    }
    
    // Calculate and return fitness
    engine.calculate_fitness()
}

/// Train the population on the breakout game using parallel processing
/// This evaluates all individuals in parallel across all available CPU cores
pub fn train_population(individuals: &mut Vec<Individual>, num_steps: usize) {
    // Use rayon's parallel iterator to evaluate all individuals concurrently
    individuals.par_iter_mut().for_each(|individual| {
        individual.fitness = evaluate_individual(individual, num_steps);
    });
}

/// Train the population and return statistics about the run
pub fn train_population_with_stats(individuals: &mut Vec<Individual>, num_steps: usize) -> TrainingStats {
    let start_time = std::time::Instant::now();
    
    // Parallel evaluation
    individuals.par_iter_mut().for_each(|individual| {
        individual.fitness = evaluate_individual(individual, num_steps);
    });
    
    let duration = start_time.elapsed();
    
    // Calculate statistics
    let total_fitness: f32 = individuals.iter().map(|i| i.fitness).sum();
    let avg_fitness = total_fitness / individuals.len() as f32;
    let max_fitness = individuals.iter().map(|i| i.fitness).fold(0.0f32, f32::max);
    let min_fitness = individuals.iter().map(|i| i.fitness).fold(f32::INFINITY, f32::min);
    
    TrainingStats {
        duration,
        avg_fitness,
        max_fitness,
        min_fitness,
        population_size: individuals.len(),
    }
}

pub struct TrainingStats {
    pub duration: std::time::Duration,
    pub avg_fitness: f32,
    pub max_fitness: f32,
    pub min_fitness: f32,
    pub population_size: usize,
}
