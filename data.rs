/// this module handles loading, cleaning, and encoding of peptide data from CSV files to 
/// use in the neural network classifier.

use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use csv::Reader;
use serde::Deserialize;

// represents a peptide with its amino acid sequence and activity class label.
#[derive(Debug, Clone, Deserialize)]
pub struct Peptide {
    pub sequence: String,
    pub class: String,
}

/// Loads peptides from a CSV file and normalizes the class labels.
///
/// Inputs: 
/// Path to the CSV file containing peptide data.
///
/// Outputs:
/// returns a vec of peptides when successful, or an error.
///
/// Logic: 
/// reads each row, trims and lowercases the class label, and stores it in a vector.

pub fn load_peptides_from_csv(path: &str) -> Result<Vec<Peptide>, Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(BufReader::new(file));
    let mut peptides = Vec::new();

    for result in reader.deserialize() {
        let mut record: Peptide = result?;
        record.class = record.class.trim().to_lowercase();
        peptides.push(record);
    }

    Ok(peptides)
}

/// encodes a peptide sequence as a vector of integers representing amino acids
///
/// Inputs: 
/// a string of uppercase letters
///
/// Outputs
/// vector u8 values representing the encoded sequence.
///
/// Logic
/// matches each amino acid to a unique integer. resizes to ensure length is 20 to avoic complications
/// when doing matrix multiplicaton. 
pub fn encode_peptide_sequence(sequence: &str) -> Vec<u8> {
    let mut encoded = sequence.chars().map(|c| {
        match c {
            'A' => 0,
            'C' => 1,
            'D' => 2,
            'E' => 3,
            'F' => 4,
            'G' => 5,
            'H' => 6,
            'I' => 7,
            'K' => 8,
            'L' => 9,
            'M' => 10,
            'N' => 11,
            'P' => 12,
            'Q' => 13,
            'R' => 14,
            'S' => 15,
            'T' => 16,
            'V' => 17,
            'W' => 18,
            'Y' => 19,
            _ => 20,
        }
    }).collect::<Vec<u8>>();

    /// i had to do this so matrix multiplication during forward and backward pass works 
    /// in all cases

    let desired_len = 20;
    if encoded.len() < desired_len {
        encoded.resize(desired_len, 20); //padding 
    } else if encoded.len() > desired_len {
        encoded.truncate(desired_len); //dropping values after 20
    }

    encoded
}

/// Encodes class labels
///
/// Inputs
/// a lowercase string representing class (like "active").
///
/// Outputs
/// one-hot encoded array: [1, 0, 0] for "mod. active", 
/// [0, 1, 0] for "inactive - exp" and "inactive - virtual", 
/// [0, 0, 1] for "very active", and 
/// [0, 0, 0] for all others. 
///
/// Logic
/// Simple string match with default [0, 0, 0]

pub fn encode_class(class: &str) -> [u8; 3] {

    match class.to_lowercase().as_str() {
        "mod " => [1, 0, 0],
        "exp" => [0, 1, 0],
        "very" => [0, 0, 1],
        "virtual" => [0, 1, 0],
        _ => [0, 0, 0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_peptide_sequence() {
        let sequence = "ACDEFGHIKLMNPQRSTVWY";
        let encoded = encode_peptide_sequence(sequence);
        assert_eq!(encoded.len(), 20);
        assert_eq!(encoded[0], 0);  // A
        assert_eq!(encoded[1], 1);  // C
        assert_eq!(encoded[19], 19); // Y
    }
} 