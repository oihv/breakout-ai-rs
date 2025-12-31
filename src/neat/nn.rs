use super::{Genome, LinkGene};
use std::collections::{HashMap, HashSet};

pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[derive(Clone, Debug, PartialEq)]
pub struct NeuronInput {
    pub input_id: i32,
    pub weight: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Neuron {
    pub id: i32,
    pub bias: f32,
    pub inputs: Vec<NeuronInput>,
}

pub fn required_for_output(
    inputs: &Vec<i32>,
    outputs: &Vec<i32>,
    links: &Vec<LinkGene>,
) -> HashSet<i32> {
    let mut required = HashSet::new();
    for i in outputs {
        required.insert(*i);
    }
    let mut s = required.clone();

    loop {
        // find the nodes in the links that outputs to the current layer, and input not in the
        // current layer
        let t: HashSet<i32> = links
            .clone()
            .iter()
            .filter(|x| s.contains(&x.id.out_id) && !s.contains(&x.id.in_id))
            .map(|x| x.id.in_id)
            .collect();

        if t.is_empty() {
            break;
        }

        // Why copied? because filter returns an iterator of references
        // only add nodes that is not in the input nodes
        let layer_nodes: HashSet<i32> = t.iter().filter(|x| !inputs.contains(x)).copied().collect();
        if layer_nodes.is_empty() {
            break;
        }

        required = required.union(&layer_nodes).copied().collect(); // or &required | &layer_nodes
        s = s.union(&t).copied().collect();
    }

    required
}

pub fn feed_forward_layers(
    inputs: &Vec<i32>,
    outputs: &Vec<i32>,
    links: &Vec<LinkGene>,
) -> Vec<Vec<i32>> {
    let mut layers: Vec<Vec<i32>> = Vec::new();
    let required = required_for_output(&inputs, &outputs, &links);

    let mut potential_input: HashSet<i32> = inputs.iter().copied().collect();

    loop {
        // Candidate nodes c for the next layer. The nodes should connect a node IN s to a node
        // NOT IN s
        let c: HashSet<i32> = links
            .clone()
            .iter()
            .filter(|x| {
                potential_input.contains(&x.id.in_id) && !potential_input.contains(&x.id.out_id)
            })
            .map(|x| x.id.out_id)
            .collect();

        // Keep only the used nodes whose entire input set is contained in s
        let mut next_layer: Vec<i32> = Vec::new();
        for n in c {
            let inputs_to_n: Vec<i32> = links
                .clone()
                .iter()
                .filter(|x| x.id.out_id == n && required.contains(&x.id.in_id))
                .map(|x| x.id.in_id)
                .collect();

            if required.contains(&n) && inputs_to_n.iter().all(|x| potential_input.contains(x)) {
                next_layer.push(n);
            }
        }

        if next_layer.is_empty() {
            break;
        }

        layers.push(next_layer.clone());
        potential_input = &potential_input | &next_layer.iter().copied().collect();
    }

    layers
}

#[derive(Debug, PartialEq)]
pub struct FeedForwardNeuralNetwork {
    pub input_ids: Vec<i32>,
    pub output_ids: Vec<i32>,
    pub neurons: Vec<Neuron>,
}

impl FeedForwardNeuralNetwork {
    pub fn activate(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        // dbg!(inputs.len());
        // dbg!(self.input_ids.len());
        assert!(inputs.len() == self.input_ids.len());

        let mut values: HashMap<i32, f32> = HashMap::new();
        for i in 0..inputs.len() {
            let input_id = self.input_ids[i];
            values.insert(input_id, inputs[i]);
        }

        for i in 0..self.output_ids.len() {
            values.insert(i as i32, 0.0);
        }

        for neuron in self.neurons.clone() {
            let mut value = 0.0;
            for input in &neuron.inputs {
                if !values.contains_key(&input.input_id) {
                    panic!("Missing input {} for neuron {}", input.input_id, neuron.id);
                }
                value += *(values.get(&input.input_id).unwrap()) * input.weight;
            }
            value += neuron.bias;

            // Only apply ReLU to hidden neurons, not output neurons
            // Output neurons need to be able to express negative values for comparison
            if !self.output_ids.contains(&neuron.id) {
                value = relu(value);
            }

            values.insert(neuron.id, value);
        }

        let mut outputs: Vec<f32> = Vec::new();
        for output_id in self.output_ids.clone() {
            assert!(values.contains_key(&output_id));
            outputs.push(values[&output_id]);
        }

        outputs
    }

    pub fn create_from_genome(genome: &Genome) -> FeedForwardNeuralNetwork {
        let inputs = genome.make_input_ids();
        let outputs = genome.make_output_ids();

        // Filter only enabled links
        let enabled_links: Vec<LinkGene> = genome
            .links
            .iter()
            .filter(|link| link.is_enabled)
            .cloned()
            .collect();

        let layers = feed_forward_layers(&inputs, &outputs, &enabled_links);

        let mut neurons: Vec<Neuron> = Vec::new();
        for layer in layers {
            for neuron_id in layer {
                let mut neuron_inputs: Vec<NeuronInput> = Vec::new();
                for link in enabled_links.clone() {
                    if neuron_id == link.id.out_id {
                        neuron_inputs.push(NeuronInput {
                            input_id: link.id.in_id,
                            weight: link.weight,
                        });
                    }
                }
                if let Some(neuron_gene) = genome.find_neuron(&neuron_id) {
                    neurons.push(Neuron {
                        id: neuron_gene.id,
                        bias: neuron_gene.bias,
                        inputs: neuron_inputs,
                    })
                }
            }
        }
        FeedForwardNeuralNetwork {
            input_ids: inputs,
            output_ids: outputs,
            neurons,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neat::*;

    #[test]
    fn simple_nn_activation_test() {
        let neurons = vec![
            Neuron {
                id: 0,
                bias: 0.,
                inputs: vec![
                    NeuronInput {
                        input_id: -1,
                        weight: 1.,
                    },
                    NeuronInput {
                        input_id: -2,
                        weight: 2.,
                    },
                    NeuronInput {
                        input_id: -3,
                        weight: 3.,
                    },
                ],
            },
            Neuron {
                id: 1,
                bias: 1.,
                inputs: vec![
                    NeuronInput {
                        input_id: -1,
                        weight: 1.,
                    },
                    NeuronInput {
                        input_id: -2,
                        weight: 2.,
                    },
                    NeuronInput {
                        input_id: -3,
                        weight: 3.,
                    },
                ],
            },
            Neuron {
                id: 2,
                bias: 2.,
                inputs: vec![
                    NeuronInput {
                        input_id: -1,
                        weight: 1.,
                    },
                    NeuronInput {
                        input_id: -2,
                        weight: 2.,
                    },
                    NeuronInput {
                        input_id: -3,
                        weight: 3.,
                    },
                ],
            },
        ];
        let mut nn = FeedForwardNeuralNetwork {
            input_ids: vec![-1, -2, -3],
            output_ids: vec![0, 1, 2],
            neurons,
        };

        let output = nn.activate(vec![1., 2., 3.]);

        assert_eq!(output[0], 14.0);
        assert_eq!(output[1], 15.0);
        assert_eq!(output[2], 16.0);
    }

    #[test]
    fn create_from_genome_test() {
        let neurons = vec![
            Neuron {
                id: 0,
                bias: 0.,
                inputs: vec![
                    NeuronInput {
                        input_id: -1,
                        weight: 1.,
                    },
                    NeuronInput {
                        input_id: -2,
                        weight: 2.,
                    },
                    NeuronInput {
                        input_id: -3,
                        weight: 3.,
                    },
                ],
            },
            Neuron {
                id: 1,
                bias: 1.,
                inputs: vec![
                    NeuronInput {
                        input_id: -1,
                        weight: 1.,
                    },
                    NeuronInput {
                        input_id: -2,
                        weight: 2.,
                    },
                    NeuronInput {
                        input_id: -3,
                        weight: 3.,
                    },
                ],
            },
            Neuron {
                id: 2,
                bias: 2.,
                inputs: vec![
                    NeuronInput {
                        input_id: -1,
                        weight: 1.,
                    },
                    NeuronInput {
                        input_id: -2,
                        weight: 2.,
                    },
                    NeuronInput {
                        input_id: -3,
                        weight: 3.,
                    },
                ],
            },
        ];
        let mut expected_nn = FeedForwardNeuralNetwork {
            input_ids: vec![-1, -2, -3],
            output_ids: vec![0, 1, 2],
            neurons,
        };

        let genome = Genome {
            id: 1,
            num_inputs: 3,
            num_outputs: 3,
            neurons: vec![
                NeuronGene { id: -1, bias: -1. },
                NeuronGene { id: -2, bias: -2. },
                NeuronGene { id: -3, bias: -3. },
                NeuronGene { id: 0, bias: 0. },
                NeuronGene { id: 1, bias: 1. },
                NeuronGene { id: 2, bias: 2. },
            ],
            links: vec![
                LinkGene {
                    id: LinkID {
                        in_id: -1,
                        out_id: 0,
                    },
                    weight: 1.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -1,
                        out_id: 1,
                    },
                    weight: 1.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -1,
                        out_id: 2,
                    },
                    weight: 1.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -2,
                        out_id: 0,
                    },
                    weight: 2.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -2,
                        out_id: 1,
                    },
                    weight: 2.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -2,
                        out_id: 2,
                    },
                    weight: 2.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -3,
                        out_id: 0,
                    },
                    weight: 3.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -3,
                        out_id: 1,
                    },
                    weight: 3.,
                    is_enabled: true,
                },
                LinkGene {
                    id: LinkID {
                        in_id: -3,
                        out_id: 2,
                    },
                    weight: 3.,
                    is_enabled: true,
                },
            ],
        };

        let mut nn_from_genome = FeedForwardNeuralNetwork::create_from_genome(&genome);

        // Sort neurons by ID for comparison (order doesn't matter functionally)
        expected_nn.neurons.sort_by_key(|n| n.id);
        nn_from_genome.neurons.sort_by_key(|n| n.id);

        assert_eq!(expected_nn, nn_from_genome);
    }
}
