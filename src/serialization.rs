use crate::neat::{Genome, Individual, LinkGene, LinkID, NeuronGene};
use std::fs;
use std::io::{Read, Write};
use prost::Message;

// Include the generated protobuf code
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/neat.rs"));
}

// Conversion functions from NEAT types to Proto types
impl From<&NeuronGene> for proto::NeuronGene {
    fn from(gene: &NeuronGene) -> Self {
        proto::NeuronGene {
            id: gene.id,
            bias: gene.bias,
        }
    }
}

impl From<&proto::NeuronGene> for NeuronGene {
    fn from(proto: &proto::NeuronGene) -> Self {
        NeuronGene {
            id: proto.id,
            bias: proto.bias,
        }
    }
}

impl From<&LinkID> for proto::LinkId {
    fn from(link_id: &LinkID) -> Self {
        proto::LinkId {
            in_id: link_id.in_id,
            out_id: link_id.out_id,
        }
    }
}

impl From<&proto::LinkId> for LinkID {
    fn from(proto: &proto::LinkId) -> Self {
        LinkID {
            in_id: proto.in_id,
            out_id: proto.out_id,
        }
    }
}

impl From<&LinkGene> for proto::LinkGene {
    fn from(gene: &LinkGene) -> Self {
        proto::LinkGene {
            id: Some((&gene.id).into()),
            weight: gene.weight,
            is_enabled: gene.is_enabled,
        }
    }
}

impl From<&proto::LinkGene> for LinkGene {
    fn from(proto: &proto::LinkGene) -> Self {
        LinkGene {
            id: proto.id.as_ref().unwrap().into(),
            weight: proto.weight,
            is_enabled: proto.is_enabled,
        }
    }
}

impl From<&Genome> for proto::Genome {
    fn from(genome: &Genome) -> Self {
        proto::Genome {
            id: genome.id,
            num_inputs: genome.num_inputs,
            num_outputs: genome.num_outputs,
            neurons: genome.neurons.iter().map(|n| n.into()).collect(),
            links: genome.links.iter().map(|l| l.into()).collect(),
        }
    }
}

impl From<&proto::Genome> for Genome {
    fn from(proto: &proto::Genome) -> Self {
        Genome {
            id: proto.id,
            num_inputs: proto.num_inputs,
            num_outputs: proto.num_outputs,
            neurons: proto.neurons.iter().map(|n| n.into()).collect(),
            links: proto.links.iter().map(|l| l.into()).collect(),
        }
    }
}

impl From<&Individual> for proto::Individual {
    fn from(individual: &Individual) -> Self {
        proto::Individual {
            genome: Some((&individual.genome).into()),
            fitness: individual.fitness,
        }
    }
}

impl From<&proto::Individual> for Individual {
    fn from(proto: &proto::Individual) -> Self {
        Individual {
            genome: proto.genome.as_ref().unwrap().into(),
            fitness: proto.fitness,
        }
    }
}

/// Save a genome to a file using protobuf
pub fn save_genome(genome: &Genome, filename: &str) -> std::io::Result<()> {
    let proto_genome: proto::Genome = genome.into();
    let mut buf = Vec::new();
    proto_genome.encode(&mut buf).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e)
    })?;
    
    let mut file = fs::File::create(filename)?;
    file.write_all(&buf)?;
    
    println!("Genome saved to {}", filename);
    Ok(())
}

/// Load a genome from a file using protobuf
pub fn load_genome(filename: &str) -> std::io::Result<Genome> {
    let mut file = fs::File::open(filename)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    
    let proto_genome = proto::Genome::decode(&buf[..]).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    })?;
    
    println!("Genome loaded from {}", filename);
    Ok((&proto_genome).into())
}

/// Save an individual to a file using protobuf
pub fn save_individual(individual: &Individual, filename: &str) -> std::io::Result<()> {
    let proto_individual: proto::Individual = individual.into();
    let mut buf = Vec::new();
    proto_individual.encode(&mut buf).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::Other, e)
    })?;
    
    let mut file = fs::File::create(filename)?;
    file.write_all(&buf)?;
    
    println!("Individual saved to {}", filename);
    Ok(())
}

/// Load an individual from a file using protobuf
pub fn load_individual(filename: &str) -> std::io::Result<Individual> {
    let mut file = fs::File::open(filename)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;
    
    let proto_individual = proto::Individual::decode(&buf[..]).map_err(|e| {
        std::io::Error::new(std::io::ErrorKind::InvalidData, e)
    })?;
    
    println!("Individual loaded from {}", filename);
    Ok((&proto_individual).into())
}
