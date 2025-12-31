use std::sync::OnceLock;

#[derive(Debug)]
pub struct Config {
    pub init_mean: f32,
    pub init_stdev: f32,
    pub min: f32,
    pub max: f32,
    pub mutation_rate: f32,
    pub mutate_power: f32,
    pub replace_rate: f32,
    pub survival_threshold: f32,

    pub num_inputs: i32,
    pub num_outputs: i32,


    // Mutation probabilities
    pub add_node_prob: f32,
    pub add_link_prob: f32,
    pub enable_link_prob: f32,
    pub disable_link_prob: f32,
    pub shift_weight_prob: f32,
    pub random_weight_prob: f32,

    // Crossover coefficients
    pub c1_excess: f32,
    pub c2_disjoint: f32,
    pub c3_weight: f32,

    // Compatiblity threshold for speciation
    pub compatibility_threshold: f32,

    // General
    pub population_size: usize,
    pub num_generations: usize,
    pub num_steps: usize,
}

impl Config {
    pub fn global() -> &'static Config {
        static CONFIG: OnceLock<Config> = OnceLock::new();
        CONFIG.get_or_init(|| Config {
            init_mean: 0.0,
            init_stdev: 0.5,  // Smaller initial weights for better starting point
            min: -5.0,  // Smaller range
            max: 5.0,   // Smaller range
            mutation_rate: 0.2,
            mutate_power: 0.3,  // Smaller mutations
            replace_rate: 0.05,
            survival_threshold: 0.2,

            num_inputs: 3,
            num_outputs: 3,

            add_node_prob: 0.03,
            add_link_prob: 0.05,
            enable_link_prob: 0.01,
            disable_link_prob: 0.01,
            shift_weight_prob: 0.8,  // standard weight mutation
            random_weight_prob: 0.1, // replace weight entirely

            c1_excess: 1.0,
            c2_disjoint: 1.0,
            c3_weight: 0.4,

            compatibility_threshold: 3.0,

            population_size: 150,
            num_generations: 100,
            num_steps: 5000,
        })
    }
}
