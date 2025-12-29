use rand::{random_range, seq::IndexedMutRandom};
use rand_distr::{Normal, Distribution};

use crate::neat::*;

// Structural Mutations
fn choose_random_input_or_hidden(genome: &Genome) -> i32 {
    loop {
        let mut rng = rand::rng();
        let chosen = genome.neurons.choose(&mut rng).unwrap();
        if chosen.id < genome.num_outputs && chosen.id >= 0 {
            // output neuron
            continue;
        }
        return chosen.id;
    }
}

fn choose_random_output_or_hidden(genome: &Genome) -> i32 {
    loop {
        let mut rng = rand::rng();
        let chosen = genome.neurons.choose(&mut rng).unwrap();
        if chosen.id < 0 {
            // input neuron
            continue;
        }
        return chosen.id;
    }
}

fn choose_random_hidden(genome: &Genome) -> i32 {
    loop {
        let mut rng = rand::rng();
        let chosen = genome.neurons.choose(&mut rng).unwrap();
        if chosen.id < genome.num_outputs {
            // input neuron
            continue;
        }
        return chosen.id;
    }
}
fn would_create_cycle(links: &Vec<LinkGene>, in_id: i32, out_id: i32) -> bool {
    if in_id == out_id {
        return true;
    }

    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![out_id];

    while let Some(curr) = stack.pop() {
        if !visited.insert(curr) {
            continue;
        }

        if curr == in_id {
            return true;
        }

        for link in links {
            if link.id.in_id == curr {
                stack.push(link.id.out_id);
            }
        }
    }

    false
}

pub fn mutate_add_link(genome: &mut Genome) -> Option<()> {
    let input_id = choose_random_input_or_hidden(genome);
    let output_id = choose_random_output_or_hidden(genome);
    let link_id = LinkID {
        in_id: input_id,
        out_id: output_id,
    };

    // Don't duplicate existing links
    if let Some(existing_link) = genome.find_link_mut(&link_id) {
        existing_link.is_enabled = true;
        return None;
    }

    // Only support feed forwards with no cycle
    if would_create_cycle(&genome.links, input_id, output_id) {
        return None;
    }

    let new_link = LinkGene {
        id: link_id,
        weight: (rand::random::<f32>() * 2.0 - 1.0) * 1000.0,
        is_enabled: true,
    };
    genome.links.push(new_link);

    Some(())
}

pub fn mutate_remove_link(genome: &mut Genome) -> Option<()> {
    if genome.links.is_empty() {
        return None;
    }

    let to_remove_id = random_range((genome.num_outputs)..=(genome.links.len() as i32));

    genome.links.remove(to_remove_id as usize);
    Some(())
}

pub fn mutate_add_neuron(genome: &mut Genome) -> Option<()> {
    if genome.links.is_empty() {
        return None;
    }

    let mut rng = rand::rng();
    let link_to_split: &mut LinkGene = genome.links.choose_mut(&mut rng).unwrap();
    link_to_split.is_enabled = true;

    let new_neuron = NeuronGene {
        id: genome.neurons.len() as i32,
        bias: (rand::random::<f32>() * 2.0 - 1.0) * 1000.0,
    };
    genome.neurons.push(new_neuron.clone());

    let link_id = link_to_split.id.clone();
    let prev_weight = link_to_split.weight;

    genome.links.push(LinkGene {
        id: LinkID {in_id: link_id.in_id, out_id: new_neuron.id},
        weight: 1.0,
        is_enabled: true,
    });
    genome.links.push(LinkGene {
        id: LinkID {in_id: new_neuron.id, out_id: link_id.out_id},
        weight: prev_weight,
        is_enabled: true,
    });

    Some(())
}

pub fn mutate_remove_neuron(genome: &mut Genome) -> Option<()> {
    if genome.links.is_empty() {
        return None;
    }

    let random_neuron_id = choose_random_hidden(&genome);

    // Remove associated links with this neuron
    genome.links.retain(|x| x.id.in_id != random_neuron_id && x.id.out_id != random_neuron_id);

    genome.neurons.remove(random_neuron_id as usize);

    Some(())
}

enum MutationsType {
    AddLink,
    RemoveLink,
    AddNeuron,
    RemoveNeuron,
}

pub fn mutate(genome: &mut Genome) -> Option<()> {
    let mutations_list = [mutate_add_link, mutate_add_neuron, mutate_remove_link, mutate_remove_neuron];

    let mut rng = rand::rng();
    let mutation = mutations_list.choose(&mut rng).unwrap();

    mutation(genome)
}

// Unstructured Mutations
pub fn new_value() -> f32 {
    let config = crate::config::Config::global();
    clamp(Normal::new(config.init_mean, config.init_stdev).unwrap().sample(&mut rand::rng()))
}

pub fn mutate_delta(value: f32) -> f32 {
    let config = crate::config::Config::global();
    let delta = clamp(Normal::new(0.0, config.mutate_power).unwrap().sample(&mut rand::rng()));
    clamp(value + delta)
}

pub fn clamp(x: f32) -> f32 {
    let config = crate::config::Config::global();
    f32::min(config.max, f32::max(config.min, x))
}
