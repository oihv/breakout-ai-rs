mod config;
mod neat;
mod breakout;
mod training;

fn main() {
    println!("=== Testing Initial Genome ===\n");
    
    // Create a new genome
    let genome = neat::Genome::new(3, 3);
    
    println!("Genome structure:");
    println!("  Inputs: {}", genome.num_inputs);
    println!("  Outputs: {}", genome.num_outputs);
    println!("\nNeurons:");
    for n in &genome.neurons {
        println!("  ID: {:3}, Bias: {:7.4}", n.id, n.bias);
    }
    println!("\nLinks:");
    for l in &genome.links {
        println!("  {:3} -> {:3}: weight={:7.4}, enabled={}", l.id.in_id, l.id.out_id, l.weight, l.is_enabled);
    }
    
    // Test what network outputs look like
    use neat::nn::FeedForwardNeuralNetwork;
    let mut network = FeedForwardNeuralNetwork::create_from_genome(&genome);
    
    println!("\n=== Testing Network Outputs ===");
    
    // Test case 1: Ball on left, paddle on right
    println!("\nTest 1: Ball left (0.2), Paddle right (0.8)");
    let outputs = network.activate(vec![0.2, 0.5, 0.8]);
    println!("  Outputs: L={:.4}, S={:.4}, R={:.4}", outputs[0], outputs[1], outputs[2]);
    let action = if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
        "LEFT"
    } else if outputs[2] > outputs[1] && outputs[2] > outputs[0] {
        "RIGHT"
    } else {
        "STAY"
    };
    println!("  Action: {}", action);
    
    // Test case 2: Ball on right, paddle on left
    println!("\nTest 2: Ball right (0.8), Paddle left (0.2)");
    let outputs = network.activate(vec![0.8, 0.5, 0.2]);
    println!("  Outputs: L={:.4}, S={:.4}, R={:.4}", outputs[0], outputs[1], outputs[2]);
    let action = if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
        "LEFT"
    } else if outputs[2] > outputs[1] && outputs[2] > outputs[0] {
        "RIGHT"
    } else {
        "STAY"
    };
    println!("  Action: {}", action);
    
    // Test case 3: Ball and paddle centered
    println!("\nTest 3: Ball center (0.5), Paddle center (0.5)");
    let outputs = network.activate(vec![0.5, 0.5, 0.5]);
    println!("  Outputs: L={:.4}, S={:.4}, R={:.4}", outputs[0], outputs[1], outputs[2]);
    let action = if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
        "LEFT"
    } else if outputs[2] > outputs[1] && outputs[2] > outputs[0] {
        "RIGHT"
    } else {
        "STAY"
    };
    println!("  Action: {}", action);
    
    // Simulate one full game
    println!("\n=== Simulating One Game ===");
    let individual = neat::Individual {
        genome: genome.clone(),
        fitness: 0.0,
    };
    let fitness = training::evaluate_individual(&individual);
    println!("Fitness: {:.2}", fitness);
    println!("Expected frames alive: ~{:.0}", fitness);
}
