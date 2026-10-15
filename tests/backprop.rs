//! Gradients of the intended real arithmetic.

use handwired_rs::{BackpropOptions, BooleanGradient, Circuit, Error, IndexGradient, OutOfRange, Wire};

fn gradients_of(circuit: &Circuit, source: &[f32], output: Wire, options: BackpropOptions) -> Result<Vec<f32>, Error> {
    let values = circuit.evaluate_reference(source).unwrap();
    let mut seed = vec![0.0; circuit.width()];
    seed[output.index()] = 1.0;
    circuit.backpropagate_reference(&values, &seed, options)
}

#[test]
fn products_and_selects_are_differentiated_as_a_whole() {
    // z = relu(2a - b) × a + (bit ? a : b), at a = 3, b = 1, bit = 1:
    // relu(5) × 3 + 3, so dz/da = 2 × 3 + 5 + 1 = 12, dz/db = -3, dz/dbit = a - b = 2.
    let mut circuit = Circuit::new(3);
    let (a, b, bit) = (circuit.source(0), circuit.source(1), circuit.source(2));
    circuit.assume_range(a, -10.0, 10.0);
    circuit.assume_range(b, -10.0, 10.0);
    circuit.assume_bit(bit);
    let hinge = circuit.relu(a * 2.0 - b);
    let product = circuit.multiply(hinge, a);
    let chosen = circuit.select(bit, b, a);
    let z = circuit.affine(product + chosen);
    let gradients = gradients_of(&circuit, &[3.0, 1.0, 1.0], z, BackpropOptions::default()).unwrap();
    assert_eq!(&gradients[..3], &[12.0, -3.0, 2.0]);
    // Past the hinge, relu(2a - b) contributes nothing.
    let gradients = gradients_of(&circuit, &[-3.0, 1.0, 0.0], z, BackpropOptions::default()).unwrap();
    assert_eq!(&gradients[..3], &[0.0, 1.0, -4.0]);
}

#[test]
fn boolean_atoms_need_an_explicit_gradient_rule() {
    let mut circuit = Circuit::new(1);
    let score = circuit.source(0);
    circuit.assume_range(score, 0.0, 1.0);
    let bit = circuit.boolean(score);
    let doubled = circuit.affine(bit * 2.0);
    assert!(matches!(
        gradients_of(&circuit, &[0.7], doubled, BackpropOptions::default()),
        Err(Error::BooleanGradient { .. })
    ));
    let options = BackpropOptions { boolean_gradient: BooleanGradient::StraightThrough, ..Default::default() };
    assert_eq!(gradients_of(&circuit, &[0.7], doubled, options).unwrap()[0], 2.0);
}

#[test]
fn index_reads_route_gradient_to_the_selected_candidate() {
    let mut circuit = Circuit::new(4);
    let index = circuit.source(0);
    circuit.assume_range(index, 0.0, 2.0);
    let table: Vec<Wire> = (1..4).map(|position| circuit.source(position)).collect();
    let read = circuit.index_read(index, &table, OutOfRange::Clamp);
    let source = [1.0, 10.0, 20.0, 40.0];
    assert!(matches!(
        gradients_of(&circuit, &source, read, BackpropOptions::default()),
        Err(Error::IndexGradient { .. })
    ));
    let zero = BackpropOptions { index_gradient: IndexGradient::Zero, ..Default::default() };
    assert_eq!(gradients_of(&circuit, &source, read, zero).unwrap(), vec![0.0, 0.0, 1.0, 0.0, 1.0]);
    // A backward difference gives the index read(1) - read(0).
    let difference = BackpropOptions { index_gradient: IndexGradient::BackwardDifference, ..Default::default() };
    assert_eq!(gradients_of(&circuit, &source, read, difference).unwrap()[0], 10.0);
}
