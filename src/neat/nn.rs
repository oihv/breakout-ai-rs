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
            for input in neuron.inputs {
                assert!(values.contains_key(&input.input_id));
                value += *(values.get(&input.input_id).unwrap()) * input.weight;
            }
            value += neuron.bias;
            value = relu(value);
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
        let layers = feed_forward_layers(&inputs, &outputs, &genome.links);

        let mut neurons: Vec<Neuron> = Vec::new();
        for layer in layers {
            for neuron_id in layer {
                let mut neuron_inputs: Vec<NeuronInput> = Vec::new();
                for link in genome.links.clone() {
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
