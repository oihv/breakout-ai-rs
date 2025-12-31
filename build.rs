fn main() {
    prost_build::compile_protos(&["proto/neat.proto"], &["proto/"]).unwrap();
}
