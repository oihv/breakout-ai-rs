use rand::{seq::IndexedRandom};

use crate::neat::mutation::new_value;
mod crossover;
mod mutation;
mod population;
mod nn;

#[derive(Clone)]
struct NeuronGene {
    id: i32,
    bias: f32,
}

#[derive(Debug, PartialEq, Eq, Clone)]
struct LinkID {
    in_id: i32,
    out_id: i32,
}

#[derive(Clone)]
struct LinkGene {
    id: LinkID,
    weight: f32,
    is_enabled: bool,
}

#[derive(Clone)]
struct Genome {
    id: i32,
    num_inputs: i32,
    num_outputs: i32,
    neurons: Vec<NeuronGene>,
    links: Vec<LinkGene>,
}

static mut GENOME_INDEXER: i32 = 0;
impl Genome {
    pub fn genome_indexer() -> i32 {
        unsafe {
            GENOME_INDEXER += 1;
            GENOME_INDEXER
        }
    }

    pub fn new(num_in: i32, num_out: i32) -> Genome {
        let mut new_genome = Genome {
            id: Genome::genome_indexer(),
            num_inputs: num_in,
            num_outputs: num_out,
            neurons: Vec::new(),
            links: Vec::new(),
        };

        for i in 0..num_out {
            new_genome.neurons.push(NeuronGene {id: i, bias: new_value()})
        }

        for i in 0.. num_in {
            let input_id = -i - 1;
            new_genome.neurons.push(NeuronGene {id: input_id, bias: new_value()});
            for output_id in 0..num_out {
                new_genome.links.push(LinkGene {id: LinkID { in_id: input_id, out_id: output_id }, weight: new_value(), is_enabled: true })
            }
        }
        new_genome
    }

    pub fn find_neuron(&self, id: &i32) -> Option<&NeuronGene> {
        self.neurons.iter().find(|x| x.id == *id)
    }

    pub fn find_link(&self, id: &LinkID) -> Option<&LinkGene> {
        self.links.iter().find(|x| x.id == *id)
    }

    pub fn find_link_mut(&mut self, id: &LinkID) -> Option<&mut LinkGene> {
        self.links.iter_mut().find(|x| x.id == *id)
    }

    pub fn make_input_ids(&self) -> Vec<i32> {
        let mut inputs: Vec<i32> = Vec::new();
        let mut id = -1;
        for _ in 0..self.num_inputs {
            inputs.push(id);
            id -= 1;
        }
        inputs
    }

    pub fn make_output_ids(&self) -> Vec<i32> {
        let mut outputs: Vec<i32> = Vec::new();
        for i in 0..self.num_inputs {
            outputs.push(i);
        }
        outputs
    }
}

#[derive(Clone)]
struct Individual {
    genome: Genome,
    fitness: f32,
}
