mod dice_states;
mod score_states;
use hdf5;
use ndarray::Array3;
mod errors;
mod reward_evaluation;

fn main() -> Result<(), errors::Error> {
    // If the file already exists, we just load the transition function from it.
    match hdf5::File::open_rw("yahtzee-solver.h5") {
        Ok(file) => {
            let all_dice_states = dice_states::get_all_dice_states();
            let all_keep_actions = dice_states::get_all_keep_actions();

            let transition_function_dataset = file.dataset("transition_function")?;
            let transition_function: Array3<f32> = transition_function_dataset.read()?;

            reward_evaluation::calculate_and_save_all_score_state_reward(
                &all_dice_states,
                &all_keep_actions,
                &transition_function,
                file,
            )?;
        }
        Err(_) => {
            // The file doesn't exist, so we generate the transition function and save it to the
            // file.

            let all_dice_states = dice_states::get_all_dice_states();
            println!("Number of dice states: {}", all_dice_states.len());
            let all_keep_actions = dice_states::get_all_keep_actions();
            println!("Number of keep actions: {}", all_keep_actions.len());
            println!(
                "Number of score states: {}",
                score_states::ScoreState::num_all_states()
            );

            let transition_function =
                dice_states::get_transition_function(&all_dice_states, &all_keep_actions);

            let file = hdf5::File::create("yahtzee-solver.h5")?;

            // Save the transition function to a dataset.
            let transition_function_dataset = file
                .new_dataset::<f32>()
                .shape(transition_function.shape())
                .create("transition_function")?;
            transition_function_dataset.write(&transition_function)?;
        }
    }
    Ok(())
}
