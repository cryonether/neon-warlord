//! Unit Tests for neat.rs

// use crate::reinforcement_learning::neat;

use super::*;

fn _test_logic_function(truth_table: &[(f32, f32, f32); 4]) {
    let size = 400;
    let n = 1000;

    let mut neat = Neat::new(2, 1, size);

    for _i in 0..n {
        for genom in &mut neat.genomes {
            let mut fitness = 0.0;
            for &(a, b, expected) in truth_table {
                let sensors = genom.sensors();
                sensors[0].value = a;
                sensors[1].value = b;

                genom.evaluate();

                let outputs = genom.outputs();
                let error = expected - outputs[0].value;

                // Max reward is 1.0 per case.
                fitness += 1.0 - error * error;
            }

            genom.fitness = fitness;
        }

        neat.rank();
        neat.survival_selection();
        neat.evolve();
    }

    let genome = neat.get_rank_0().unwrap();
    let fitness = genome.fitness;
    assert!(fitness >= 3.9, "assertion failed: {fitness} >= 3.9");
}

#[test]
fn and() {
    let truth_table = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ];
    _test_logic_function(&truth_table);
}

#[test]
fn or() {
    let truth_table = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
    ];
    _test_logic_function(&truth_table);
}

#[test]
fn not() {
    let truth_table = [
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ];
    _test_logic_function(&truth_table);
}

#[test]
fn xor() {
    let truth_table = [
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];
    _test_logic_function(&truth_table);
}

#[test]
fn nand() {
    let truth_table = [
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
    ];
    _test_logic_function(&truth_table);
}

#[test]
fn nor() {
    let truth_table = [
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
    ];
    _test_logic_function(&truth_table);
}

#[test]
fn xnor() {
    let truth_table = [
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    ];
    _test_logic_function(&truth_table);
}
