//! Exports and imports a neural network as SafeTensor

use std::collections::HashMap;

use safetensors::tensor::{Dtype, TensorView};
use safetensors::{SafeTensorError};

use crate::reinforcement_learning::neural_network_simd::simd_math::simd_mat::SMat16;
use crate::reinforcement_learning::neural_network_simd::simd_math::simd_vec::SVec16;
use crate::reinforcement_learning::neural_network_simd::{NeuralNetworkSimd, activation_function::ActivationFunction};


pub fn export_ppo<ActorNetwork, CriticNetwork>(
    actor: &ActorNetwork,
    critic: &CriticNetwork,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>>
where 
    ActorNetwork: ExportSafetensors, 
    CriticNetwork: ExportSafetensors,
{
    let bytes = export_ppo_bytes(actor, critic)?;
    std::fs::write(path, bytes)?;

    Ok(())
}


pub fn export_ppo_bytes<ActorNetwork, CriticNetwork>(
    actor: &ActorNetwork,
    critic: &CriticNetwork,
) -> Result<Vec<u8>, Box<dyn std::error::Error>>
where 
    ActorNetwork: ExportSafetensors, 
    CriticNetwork: ExportSafetensors,
{
    let mut storage = Vec::new();

    actor.append_safetensors(&mut storage, "actor");
    critic.append_safetensors(&mut storage, "critic");

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

    let bytes = safetensors::serialize(&tensors, None)?;

    Ok(bytes)
}

pub fn import_ppo<ActorNetwork, CriticNetwork>(
    actor: &mut ActorNetwork,
    critic: &mut CriticNetwork,
    path: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>>
where
    ActorNetwork: ImportSafetensors,
    CriticNetwork: ImportSafetensors,
{
    let data = std::fs::read(path)?;
    import_ppo_bytes(actor, critic, data)
}

pub fn import_ppo_bytes<ActorNetwork, CriticNetwork>(
    actor: &mut ActorNetwork,
    critic: &mut CriticNetwork,
    data: Vec<u8>,
) -> Result<(), Box<dyn std::error::Error>>
where
    ActorNetwork: ImportSafetensors,
    CriticNetwork: ImportSafetensors,
{
    let tensors = safetensors::SafeTensors::deserialize(&data)?;

    actor.load_safetensors(&tensors, "actor")?;
    critic.load_safetensors(&tensors, "critic")?;

    Ok(())
}

fn load_f32<const N: usize>(
    tensors: &safetensors::SafeTensors,
    name: &str,
    expected_shape: &[usize],
) -> Result<[f32; N], Box<dyn std::error::Error>> {
    let tensor = tensors.tensor(name)?;

    if tensor.dtype() != Dtype::F32 {
        return Err(format!("Invalid dtype for {name}").into());
    }

    if tensor.shape() != expected_shape {
        return Err(format!("Invalid shape for {name}").into());
    }

    let bytes = tensor.data();

    if bytes.len() != N * 4 {
        return Err(format!("Invalid data length for {name}").into());
    }

    let mut result = [0.0f32; N];

    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        result[i] = f32::from_le_bytes(
            chunk.try_into()?
        );
    }

    Ok(result)
}

pub trait ExportSafetensors {
    fn append_safetensors(
        &self,
        storage: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
    );
}

pub trait ImportSafetensors {
    fn load_safetensors(
        &mut self,
        tensors: &safetensors::SafeTensors,
        prefix: &str,
    ) -> Result<(), Box<dyn std::error::Error>>;
}

impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivation: ActivationFunction<OUTPUTS>,
> ExportSafetensors for NeuralNetworkSimd<
    INPUTS,
    OUTPUTS,
    NEURONS,
    NR_LAYERS,
    RESIDUAL,
    OutputActivation,
> {    
    fn append_safetensors(
        &self,
        storage: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
    ) {
        push_mat16(
            storage,
            format!("{prefix}.input.weight"),
            &self.input.w,
            vec![NEURONS, INPUTS],
        );

        push_vec16(
            storage,
            format!("{prefix}.input.bias"),
            &self.input.b,
            vec![NEURONS],
        );

        for (i, layer) in self.layers.iter().enumerate() {
            push_mat16(
                storage,
                format!("{prefix}.layers.{i}.weight"),
                &layer.w,
                vec![NEURONS, NEURONS],
            );

            push_vec16(
                storage,
                format!("{prefix}.layers.{i}.bias"),
                &layer.b,
                vec![NEURONS],
            );
        }

        push_mat16(
            storage,
            format!("{prefix}.output.weight"),
            &self.output.w,
            vec![OUTPUTS, NEURONS],
        );

        push_vec16(
            storage,
            format!("{prefix}.output.bias"),
            &self.output.b,
            vec![OUTPUTS],
        );
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


impl<
    const INPUTS: usize,
    const OUTPUTS: usize,
    const NEURONS: usize,
    const NR_LAYERS: usize,
    const RESIDUAL: bool,
    OutputActivation: ActivationFunction<OUTPUTS>,
> ImportSafetensors for NeuralNetworkSimd<
    INPUTS,
    OUTPUTS,
    NEURONS,
    NR_LAYERS,
    RESIDUAL,
    OutputActivation,
> {
    fn load_safetensors(
        &mut self,
        tensors: &safetensors::SafeTensors,
        prefix: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {

        // Load input layer
        let input_w = load_mat16::<NEURONS, INPUTS>(
            tensors,
            &format!("{prefix}.input.weight"),
        )?;

        let input_b = load_vec16::<NEURONS>(
            tensors,
            &format!("{prefix}.input.bias"),
        )?;

        // Load hidden layers
        let mut layers = Vec::with_capacity(NR_LAYERS);

        for i in 0..NR_LAYERS {
            let w = load_mat16::<NEURONS, NEURONS>(
                tensors,
                &format!("{prefix}.layers.{i}.weight"),
            )?;

            let b = load_vec16::<NEURONS>(
                tensors,
                &format!("{prefix}.layers.{i}.bias"),
            )?;

            layers.push((w, b));
        }

        // Load output layer
        let output_w = load_mat16::<OUTPUTS, NEURONS>(
            tensors,
            &format!("{prefix}.output.weight"),
        )?;

        let output_b = load_vec16::<OUTPUTS>(
            tensors,
            &format!("{prefix}.output.bias"),
        )?;

        // All tensors successfully loaded and validated.
        // Now update the network.

        self.input.w = input_w;
        self.input.b = input_b;

        for (layer, (w, b)) in self.layers.iter_mut().zip(layers) {
            layer.w = w;
            layer.b = b;
        }

        self.output.w = output_w;
        self.output.b = output_b;

        Ok(())
    }
}

fn load_vec16<const N: usize>(
    tensors: &safetensors::SafeTensors,
    name: &str,
) -> Result<SVec16<N>, Box<dyn std::error::Error>> {

    let tensor = tensors.tensor(name)?;

    if tensor.dtype() != Dtype::F32 {
        return Err(format!(
            "Invalid dtype for {name}: expected F32"
        ).into());
    }

    if tensor.shape() != [N] {
        return Err(format!(
            "Invalid shape for {name}: expected [{N}], got {:?}",
            tensor.shape()
        ).into());
    }

    let bytes = tensor.data();

    if bytes.len() != N * 4 {
        return Err(format!(
            "Invalid byte length for {name}"
        ).into());
    }

    let mut result = SVec16::<N>::zero();

    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        result[i] = f32::from_le_bytes(
            chunk.try_into()?
        );
    }

    Ok(result)
}

fn load_mat16<const M: usize, const N: usize>(
    tensors: &safetensors::SafeTensors,
    name: &str,
) -> Result<SMat16<M, N>, Box<dyn std::error::Error>> {

    let tensor = tensors.tensor(name)?;

    if tensor.dtype() != Dtype::F32 {
        return Err(format!(
            "Invalid dtype for {name}: expected F32"
        ).into());
    }

    if tensor.shape() != [M, N] {
        return Err(format!(
            "Invalid shape for {name}: expected [{M}, {N}], got {:?}",
            tensor.shape()
        ).into());
    }

    let bytes = tensor.data();

    if bytes.len() != M * N * 4 {
        return Err(format!(
            "Invalid byte length for {name}"
        ).into());
    }

    let mut result = SMat16::<M, N>::zero();

    for (i, chunk) in bytes.chunks_exact(4).enumerate() {
        let row = i / N;
        let col = i % N;

        result[row][col] = f32::from_le_bytes(
            chunk.try_into()?
        );
    }

    Ok(result)
}
