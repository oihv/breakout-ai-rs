use crate::neat::*;
// TODO: maybe add activation function too?
pub fn crossover_neuron(a: &NeuronGene, b: &NeuronGene) -> NeuronGene {
    assert_eq!(a.id, b.id);

    let mut rng = rand::rng();
    let biases = [a.bias, b.bias];
    let bias = biases.choose(&mut rng).unwrap();

    NeuronGene {
        id: a.id,
        bias: *bias,
    }
}

pub fn crossover_link(a: &LinkGene, b: &LinkGene) -> LinkGene {
    assert_eq!(a.id, b.id);

    let mut rng = rand::rng();
    let weights = [a.weight, b.weight];
    let weight = weights.choose(&mut rng).unwrap();
    let enables = [a.is_enabled, b.is_enabled];
    let enable = enables.choose(&mut rng).unwrap();

    LinkGene {
        id: a.id.clone(),
        weight: *weight,
        is_enabled: *enable,
    }
}

pub fn crossover(dominant: &Individual, recessive: &Individual) -> Genome {
    let mut offspring = Genome {
        id: Genome::genome_indexer(),
        num_inputs: dominant.genome.num_inputs,
        num_outputs: dominant.genome.num_outputs,
        neurons: Vec::new(),
        links: Vec::new(),
    };

    // inherit neuron genes
    for dominant_neuron in dominant.genome.neurons.clone() {
        if let Some(recessive_neuron) = recessive.genome.find_neuron(&dominant_neuron.id) {
            offspring.neurons.push(crossover_neuron(&dominant_neuron, recessive_neuron));
        } else {
            offspring.neurons.push(dominant_neuron);
        }
    }

    // inherit link gene
    for dominant_link in dominant.genome.links.clone() {
        if let Some(recessive_link) = dominant.genome.find_link(&dominant_link.id) {
            offspring.links.push(crossover_link(&dominant_link, recessive_link));
        } else {
            offspring.links.push(dominant_link);
        }
    }

    offspring
}
