//! Exports and imports a neural network as SafeTensor

use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensorError};

use crate::reinforcement_learning::neural_network_simd::simd_math::simd_mat::SMat16;
use crate::reinforcement_learning::neural_network_simd::simd_math::simd_vec::SVec16;
use crate::reinforcement_learning::neural_network_simd::{NeuralNetworkSimd, activation_function::ActivationFunction};



impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivation: ActivationFunction<OUTPUTS>,
> NeuralNetworkSimd<
    INPUTS,
    OUTPUTS,
    NEURONS,
    NR_LAYERS,
    RESIDUAL,
    OutputActivation,
> {
    pub fn export_safetensors(
        &self,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {

        // Owned tensor buffers
        let mut storage = Vec::new();

        // Input layer
        push_mat16(
            &mut storage,
            "input.weight".into(),
            &self.input.w,
            vec![NEURONS, INPUTS],
        );

        push_vec16(
            &mut storage,
            "input.bias".into(),
            &self.input.b,
            vec![NEURONS],
        );

        // Hidden layers
        for (i, layer) in self.layers.iter().enumerate() {
            push_mat16(
                &mut storage,
                format!("layers.{i}.weight"),
                &layer.w,
                vec![NEURONS, NEURONS],
            );

            push_vec16(
                &mut storage,
                format!("layers.{i}.bias"),
                &layer.b,
                vec![NEURONS],
            );
        }

        // Output layer
        push_mat16(
            &mut storage,
            "output.weight".into(),
            &self.output.w,
            vec![OUTPUTS, NEURONS],
        );

        push_vec16(
            &mut storage,
            "output.bias".into(),
            &self.output.b,
            vec![OUTPUTS],
        );

        // Create TensorViews referencing owned buffers
        let tensors: HashMap<String, TensorView<'_>> =
            storage.iter()
                .map(|(name, shape, bytes)| {
                    Ok((
                        name.clone(),
                        TensorView::new(
                            Dtype::F32,
                            shape.clone(),
                            bytes,
                        )?,
                    ))
                })
                .collect::<Result<_, SafeTensorError>>()?;

        safetensors::serialize_to_file(
            &tensors,
            None,
            path,
        )?;

        Ok(())
    }
}

fn push_vec16<const N: usize>(
    tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: String,
    data: &SVec16<N>,
    shape: Vec<usize>,
)
{
    let bytes = data
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect();

    tensors.push((name, shape, bytes));
}

fn push_mat16<const M: usize, const N: usize>(
    tensors: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
    name: String,
    data: &SMat16<M, N>,
    shape: Vec<usize>,
)
{
    let bytes = data
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|x| x.to_le_bytes())
        .collect();

    tensors.push((name, shape, bytes));
}