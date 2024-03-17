mod dice_states;
mod score_states;
use hdf5;
use ndarray::prelude::*;
use std::cmp::Reverse;

use crate::reward_evaluation::{IntKeyedArrayMap, NUM_ROLLS};
mod errors;
mod reward_evaluation;

fn main() -> Result<(), errors::Error> {
    // If the file already exists, we just load the transition function from it.
    match hdf5::File::open_rw("/result/yahtzee-solver.h5") {
        Ok(file) => {
            let all_dice_states = dice_states::get_all_dice_states();
            let all_keep_actions = dice_states::get_all_keep_actions();

            let transition_function_dataset = file.dataset("transition_function")?;
            let transition_function: Array3<f32> = transition_function_dataset.read()?;
            // Attemp to read the reward of all layers
            if let Ok(groups) = file.groups() {
                let mut num_explored_states = 0;
                let mut layer_groups = groups
                    .iter()
                    .filter(|group| group.name().starts_with("/layer_"))
                    .collect::<Vec<_>>();
                layer_groups.sort_by_key(|group| {
                    Reverse(
                        group
                            .name()
                            .split("_")
                            .last()
                            .unwrap()
                            .parse::<u8>()
                            .unwrap(),
                    )
                });
                for &group in layer_groups.iter() {
                    let num_states = group.dataset("keys")?.shape().get(0).unwrap().clone();
                    num_explored_states += num_states;
                }
                println!("Number of explored score states: {}", num_explored_states);
                for &group in layer_groups.iter() {
                    let key_dataset = group.dataset("keys")?;
                    let values_dataset = group.dataset("values")?;
                    let keys: Array1<u32> = key_dataset.read()?;
                    let values: Array3<f32> = values_dataset.read()?;
                    let state_reward_map = IntKeyedArrayMap {
                        keys: keys.to_vec(),
                        values,
                    };
                    println!(
                        "Number of states in {}: {}",
                        group.name(),
                        state_reward_map.keys.len()
                    );
                    let mut score_state = score_states::ScoreState::empty();
                    for num_roll in (0..=NUM_ROLLS).rev() {
                        println!("Rolls left: {}", num_roll);
                        println!("Score state: {}", score_state);
                        print!("Dice state: ");
                        // Get the dice state in the format: num_1s,num_2s,num_3s,num_4s,num_5s,num_6s
                        let mut dice_state = String::new();
                        std::io::stdin().read_line(&mut dice_state)?;
                        let dice_state = dice_state
                            .split(',')
                            .map(|x| x.trim().parse::<u8>().unwrap())
                            .collect::<Vec<_>>();
                        let dice_state = array![
                            dice_state[0],
                            dice_state[1],
                            dice_state[2],
                            dice_state[3],
                            dice_state[4],
                            dice_state[5]
                        ];
                        if let Some(dice_state_index) =
                            all_dice_states.iter().position(|x| x == &dice_state)
                        {
                            let score_state_reward =
                                state_reward_map.get(score_state.into()).unwrap();
                            if num_roll == 0 {
                                // We want to choose the best score action now instead of the keep action
                                //let mut best_action;
                                //let mut best_action_reward;
                                //for (action_index, action_reward) in
                                //    score_state.possible_score_actions().iter().enumerate()
                                //{
                                //    if let Some(next_state) =
                                //        score_state.apply_action(*action_reward, &dice_state)
                                //    {
                                //        let next_state_reward = score_state_reward.slice(s![2, ..]);
                                //
                                //        let action_reward =
                                //            score_state.reward(*score_action, &dice_state);
                                //        let child_score_state = score_state
                                //            .apply_action(*score_action, dice_state)
                                //            .expect("possible_score_actions should only return valid actions");
                                //        let child_score_state_index: u32 = child_score_state.into();
                                //        let all_child_rewards = state_reward_map.get(child_score_state_index).expect(
                                //            "previous_layer_reward should contain all ScoreStates reachable from ScoreState",
                                //        );
                                //        let child_reward: f32 = first_roll_probability
                                //            .dot(&all_child_rewards.slice(s![2, ..]));
                                //        max_reward =
                                //            max_reward.max(action_reward as f32 + child_reward);
                                //    }
                                //}
                            } else {
                                let next_roll_state_reward =
                                    score_state_reward.slice(s![num_roll - 1, ..]);
                                let keep_action_reward: Array2<f32> =
                                    transition_function.slice(s![dice_state_index, .., ..]).dot(
                                        &(next_roll_state_reward
                                            .to_shape((next_roll_state_reward.len(), 1))
                                            .unwrap()),
                                    );
                                let mut keep_action_reward =
                                    keep_action_reward.iter().enumerate().collect::<Vec<_>>();
                                keep_action_reward
                                    .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
                                // Top 5 keep actions with the highest reward
                                for (best_action_index, best_action_reward) in
                                    keep_action_reward.iter().rev()
                                {
                                    println!(
                                        "Reward: {}: Keep action: {}",
                                        best_action_reward, all_keep_actions[*best_action_index]
                                    );
                                }
                            }
                        } else {
                            println!("Invalid dice state");
                        }
                    }
                }
            } else {
                // Attempt to stich together the reward of all layers
                reward_evaluation::calculate_and_save_all_score_state_reward(
                    &all_dice_states,
                    &all_keep_actions,
                    &transition_function,
                    file,
                )?;
            }
        }
        Err(_) => {
            // The file doesn't exist, so we generate the transition function and save it to the
            // file.

            let all_dice_states = dice_states::get_all_dice_states();
            println!("Number of dice states: {}", all_dice_states.len());
            let all_keep_actions = dice_states::get_all_keep_actions();
            println!("Number of keep actions: {}", all_keep_actions.len());

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
