use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use csv::Reader;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Peptide {
    pub sequence: String,
    pub class: String,
}

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

    let desired_len = 20;
    if encoded.len() < desired_len {
        encoded.resize(desired_len, 20); 
    } else if encoded.len() > desired_len {
        encoded.truncate(desired_len);
    }

    encoded
}

pub fn encode_class(class: &str) -> [u8; 3] {

    match class.to_lowercase().as_str() {
        "active" => [1, 0, 0],
        "inactive" => [0, 1, 0],
        "very active" => [0, 0, 1],
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