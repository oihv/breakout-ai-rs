use rand::{seq::IndexedRandom};

use crate::neat::mutation::new_value;
pub mod crossover;
pub mod mutation;
pub mod population;
pub mod nn;

#[derive(Clone, Debug)]
pub struct NeuronGene {
    pub id: i32,
    pub bias: f32,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct LinkID {
    pub in_id: i32,
    pub out_id: i32,
}

#[derive(Clone, Debug)]
pub struct LinkGene {
    pub id: LinkID,
    pub weight: f32,
    pub is_enabled: bool,
}

#[derive(Clone, Debug)]
pub struct Genome {
    pub id: i32,
    pub num_inputs: i32,
    pub num_outputs: i32,
    pub neurons: Vec<NeuronGene>,
    pub links: Vec<LinkGene>,
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
        for i in 0..self.num_outputs {
            outputs.push(i);
        }
        outputs
    }
}

#[derive(Clone)]
pub struct Individual {
    pub genome: Genome,
    pub fitness: f32,
}
