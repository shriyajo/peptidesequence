mod data; 
mod neuralnetwork; 

use crate::data::{load_peptides_from_csv, encode_peptide_sequence, encode_class};
use crate::neuralnetwork::{NeuralNetwork, train, test};
use ndarray::Array2;

fn main() {
    println!("Running your peptide classification model...");

    let train_peptides = load_peptides_from_csv("/opt/app-root/src/peptideneuralnetwork/src/csv-cancer-train.csv").expect("Failed to load training data");

    let train_inputs: Vec<Vec<u8>> = train_peptides
        .iter()
        .map(|p| encode_peptide_sequence(&p.sequence))
        .collect();

    let train_targets: Vec<[u8; 3]> = train_peptides
        .iter()
        .map(|p| encode_class(&p.class))
        .collect();

    let mut nn = NeuralNetwork::new(20, 128, 64, 3, 0.1);

    train(&mut nn, train_inputs, train_targets);

    let test_peptides = load_peptides_from_csv("/opt/app-root/src/peptideneuralnetwork/src/csv-cancer-test.csv").expect("Failed to load test data");

    let test_inputs: Vec<Array2<f32>> = test_peptides
        .iter()
        .map(|p| {
            let encoded = encode_peptide_sequence(&p.sequence);
            let input_f32: Vec<f32> = encoded.iter().map(|&x| x as f32 / 19.0).collect();
            Array2::from_shape_vec((1, input_f32.len()), input_f32).unwrap()
        })
        .collect();

    let test_targets: Vec<Array2<f32>> = test_peptides
        .iter()
        .map(|p| {
            let encoded = encode_class(&p.class);
            let target_f32: Vec<f32> = encoded.iter().map(|&x| x as f32).collect();
            Array2::from_shape_vec((1, 3), target_f32).unwrap()
        })
        .collect();

    test(&nn, test_inputs, test_targets);
    
}
