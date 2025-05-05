use ndarray::Array2;
use rand::Rng;

pub struct NeuralNetwork {
    weights_input_to_layer1: Array2<f32>,   
    weights_layer1_to_layer2: Array2<f32>, 
    weights_layer2_to_output: Array2<f32>,
    learning_rate: f32,
}

pub fn generate_random_weights(rows: usize, cols: usize) -> Array2<f32> {
    let mut rng = rand::thread_rng();
    Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-0.1..0.1))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
    x * &(1.0 - x)
}

impl NeuralNetwork {
    pub fn new(_input_size: usize, _layer1_size: usize, _layer2_size: usize, _output_size: usize, _learning_rate: f32) -> Self {
        let weights_input_to_layer1 = generate_random_weights(20, 32);
        let weights_layer1_to_layer2 = generate_random_weights(32, 16);
        let weights_layer2_to_output = generate_random_weights(16, 3);

        NeuralNetwork {
            weights_input_to_layer1,
            weights_layer1_to_layer2,
            weights_layer2_to_output,
            learning_rate: 0.1,
        }
    }

    pub fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let z1 = input.dot(&self.weights_input_to_layer1);    
        let a1 = z1.mapv(sigmoid);

        let z2 = a1.dot(&self.weights_layer1_to_layer2);       
        let a2 = z2.mapv(sigmoid);

        let z3 = a2.dot(&self.weights_layer2_to_output); 
        let a3 = z3.mapv(sigmoid);

        return (a1, a2, a3);
    }

    pub fn backward(
        &mut self,
        input: &Array2<f32>,
        layer1_output: &Array2<f32>,
        layer2_output: &Array2<f32>,
        final_output: &Array2<f32>,
        target: &Array2<f32>,
    ) {
        let error_output = target - final_output;
        let delta_output = &error_output * &sigmoid_derivative(final_output);

        let error_layer2 = delta_output.dot(&self.weights_layer2_to_output.t());
        let delta_layer2 = &error_layer2 * &sigmoid_derivative(layer2_output);

        let error_layer1 = delta_layer2.dot(&self.weights_layer1_to_layer2.t());
        let delta_layer1 = &error_layer1 * &sigmoid_derivative(layer1_output);

        let update_out = layer2_output.t().dot(&delta_output) * self.learning_rate;
        self.weights_layer2_to_output += &update_out;

        let update_mid = layer1_output.t().dot(&delta_layer2) * self.learning_rate;
        self.weights_layer1_to_layer2 += &update_mid;

        let update_in = input.t().dot(&delta_layer1) * self.learning_rate;
        self.weights_input_to_layer1 += &update_in;
    }
    
}

pub fn train(nn: &mut NeuralNetwork, inputs: Vec<Vec<u8>>, targets: Vec<[u8; 3]>) {
    for epoch in 0..100 {

        let mut total_loss = 0.0; 
        
        for (input, target) in inputs.iter().zip(targets.iter()) {
            let input_f32: Vec<f32> = input.iter().map(|&x| x as f32 / 19.0).collect();
            let input_array = Array2::from_shape_vec((1, input.len()), input_f32).unwrap();

            let target_f32: Vec<f32> = target.iter().map(|&x| x as f32).collect();
            let target_array = Array2::from_shape_vec((1, target.len()), target_f32).unwrap();

            let (layer1_output, layer2_output, final_output) = nn.forward(&input_array);

            let loss = (&target_array - &final_output).mapv(|v| v.powi(2)).sum();
            total_loss += loss;

            nn.backward(&input_array, &layer1_output, &layer2_output, &final_output, &target_array);
        }

        println!("Epoch {}: Loss = {:.8}", epoch + 1, total_loss / inputs.len() as f32);
    }

    println!("Training completed!");
}

pub fn test(nn: &NeuralNetwork, test_inputs: Vec<Array2<f32>>, test_targets: Vec<Array2<f32>>) {
    let mut total = 0;
    let mut correct = 0;

    for i in 0..test_inputs.len() {
        let input = &test_inputs[i];
        let target = &test_targets[i];

        let (_l1, _l2, output) = nn.forward(input);

        let mut guess = 0;
        let mut biggest = output[[0, 0]];
        for j in 0..3 {
            if output[[0, j]] > biggest {
                biggest = output[[0, j]];
                guess = j;
            }
        }

        let mut actual = 0;
        for j in 0..3 {
            if target[[0, j]] == 1.0 {
                actual = j;
                break;
            }
        }

        if guess == actual {
            correct += 1;
        }

        total += 1;
    }

    let accuracy = correct as f32 / total as f32;
    println!("Accuracy: {:.5}", accuracy);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_initialization() {
        let nn = NeuralNetwork::new(20, 32, 16, 3, 0.1);

        assert_eq!(nn.weights_input_to_layer1.shape(), &[20, 32]);
        assert_eq!(nn.weights_layer1_to_layer2.shape(), &[32, 16]);
        assert_eq!(nn.weights_layer2_to_output.shape(), &[16, 3]);
    }
}