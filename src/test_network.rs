mod neat;
mod config;
mod breakout;
mod training;
mod serialization;

use breakout::BreakoutEngine;
use neat::nn::FeedForwardNeuralNetwork;

fn main() {
    println!("=== Testing Genome Loading and Neural Network ===\n");
    
    // Load the best genome
    let genome = match serialization::load_genome("best_genome.pb") {
        Ok(g) => {
            println!("✓ Successfully loaded genome!");
            println!("  Genome ID: {}", g.id);
            println!("  Num inputs: {}", g.num_inputs);
            println!("  Num outputs: {}", g.num_outputs);
            println!("  Total neurons: {}", g.neurons.len());
            println!("  Total links: {}", g.links.len());
            println!("  Enabled links: {}", g.links.iter().filter(|l| l.is_enabled).count());
            println!();
            
            println!("All links:");
            for (i, link) in g.links.iter().enumerate() {
                println!("  Link {}: {} → {} (weight={:.3}, enabled={})",
                    i, link.id.in_id, link.id.out_id, link.weight, link.is_enabled);
            }
            println!();
            
            println!("All neurons:");
            for neuron in &g.neurons {
                println!("  Neuron ID {}: bias={:.3}", neuron.id, neuron.bias);
            }
            println!();
            g
        }
        Err(e) => {
            eprintln!("✗ Failed to load genome: {}", e);
            eprintln!("Please run training first: cargo run --release --bin breakout-train");
            return;
        }
    };

    // Create neural network
    let mut network = FeedForwardNeuralNetwork::create_from_genome(&genome);
    
    println!("✓ Neural network created:");
    println!("  Input IDs: {:?}", network.input_ids);
    println!("  Output IDs: {:?}", network.output_ids);
    println!("  Neurons in network: {}", network.neurons.len());
    println!();
    
    // Create a test game state
    let engine = BreakoutEngine::new();
    let state = engine.get_state();
    
    println!("Test game state (8 inputs):");
    for (i, val) in state.iter().enumerate() {
        println!("  Input {}: {:.4}", i, val);
    }
    println!();
    
    // Run the neural network
    println!("Running neural network...");
    let outputs = network.activate(state);
    
    println!("Neural network outputs (3 actions):");
    println!("  Output 0 (Left):  {:.6}", outputs[0]);
    println!("  Output 1 (Stay):  {:.6}", outputs[1]);
    println!("  Output 2 (Right): {:.6}", outputs[2]);
    println!();
    
    // Determine action
    let action = if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
        "Left"
    } else if outputs[2] > outputs[1] && outputs[2] > outputs[0] {
        "Right"
    } else {
        "Stay"
    };
    
    println!("✓ Selected action: {}", action);
    
    // Test with multiple frames
    println!("\n=== Testing 10 frames ===");
    let mut test_engine = BreakoutEngine::new();
    test_engine.stick = false;
    
    for frame in 0..10 {
        let state = test_engine.get_state();
        let outputs = network.activate(state);
        let action_str = if outputs[0] > outputs[1] && outputs[0] > outputs[2] {
            "Left"
        } else if outputs[2] > outputs[1] && outputs[2] > outputs[0] {
            "Right"
        } else {
            "Stay"
        };
        
        println!("Frame {}: Platform X={:.2}, Action={}, Outputs=[{:.3}, {:.3}, {:.3}]",
            frame, test_engine.platform_x, action_str, outputs[0], outputs[1], outputs[2]);
    }
}
