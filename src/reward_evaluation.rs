use crate::errors::Error;
use crate::score_states::ScoreState;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
const NUM_ROLLS: usize = 3;
#[derive(Debug, Clone)]
pub struct IntKeyedArrayMap {
    pub keys: Vec<u32>,
    pub values: Array3<f32>,
}

impl IntKeyedArrayMap {
    pub fn new<'a, I>(keys: I, value_dim: (usize, usize)) -> Self
    where
        I: Iterator<Item = &'a ScoreState>,
    {
        let mut sorted_keys = keys.map(|x| (*x).into()).collect::<Vec<u32>>();
        sorted_keys.sort();
        sorted_keys.dedup();
        let num_keys = sorted_keys.len();
        Self {
            keys: sorted_keys,
            values: Array3::zeros((num_keys, value_dim.0, value_dim.1)),
        }
    }

    pub fn get(&self, key: u32) -> Option<ArrayView2<f32>> {
        match self.keys.binary_search(&key) {
            Ok(index) => Some(self.values.slice(s![index, .., ..])),
            Err(_) => None,
        }
    }

    pub fn set(&mut self, key: u32, value: &Array2<f32>) {
        match self.keys.binary_search(&key) {
            Ok(index) => {
                self.values.slice_mut(s![index, .., ..]).assign(value);
            }
            Err(_) => {
                panic!("key not found");
            }
        }
    }
}

pub fn calculate_and_save_all_score_state_reward(
    all_dice_states: &[Array1<u8>],
    all_keep_actions: &[Array1<bool>],
    transition_function: &Array3<f32>,
    hdf5_file: hdf5::File,
) -> Result<(), Error> {
    // Calculate the reward of all ScoreStates, and return it as a ndarray
    // of shape (NUM_SCORE_STATES, NUM_ROLLS, num_dice_states).
    let num_dice_states: usize = all_dice_states.len();

    let terminal_states = ScoreState::get_all_terminal_states();

    let mut exploration_set = HashSet::new();
    for terminal_state in terminal_states.iter() {
        let parent_states = terminal_state.get_parent_states();
        exploration_set.extend(parent_states.into_iter());
    }
    let mut previous_layer_reward =
        IntKeyedArrayMap::new(terminal_states.iter(), (NUM_ROLLS, num_dice_states));

    let mut layer_count = 0;
    while exploration_set.len() > 0 {
        layer_count += 1;
        let exploration_states: Vec<ScoreState> = exploration_set.into_iter().collect();
        println!(
            "Exploring layer {} with {} states...",
            layer_count,
            exploration_states.len()
        );
        let current_layer_reward = Arc::new(Mutex::new(IntKeyedArrayMap::new(
            exploration_states.iter(),
            (NUM_ROLLS, num_dice_states),
        )));
        let mut next_exploration_set = HashSet::new();
        next_exploration_set.par_extend(exploration_states.par_chunks(50000).flat_map(
            |score_states| {
                let result = score_states
                    .into_iter()
                    .map(|score_state| {
                        (
                            *score_state,
                            calculate_score_state_reward(
                                *score_state,
                                &previous_layer_reward,
                                all_dice_states,
                                all_keep_actions,
                                transition_function,
                            ),
                        )
                    })
                    .collect::<Vec<(ScoreState, Array2<f32>)>>();
                let mut parent_states = HashSet::new();
                if let Ok(mut current_layer_reward) = current_layer_reward.lock() {
                    for (score_state, reward) in result.iter() {
                        current_layer_reward.set((*score_state).into(), &reward);
                        parent_states.extend(score_state.get_parent_states());
                    }
                } else {
                    panic!("current_layer_reward lock failed");
                }
                parent_states
            },
        ));

        // Save the previous layer to the hdf5_file as two datasets in a group.
        let current_layer_reward = Arc::try_unwrap(current_layer_reward)
            .expect("current_layer_reward should only have one owner")
            .into_inner()
            .expect("current_layer_reward should not be locked");
        // Save the current layer to the hdf5_file as two datasets in a group.
        let current_layer_group = hdf5_file.create_group(&format!("layer_{}", layer_count))?;
        let current_layer_keys_dataset = current_layer_group
            .new_dataset::<u32>()
            .shape((current_layer_reward.keys.len(),))
            .create("keys")?;
        current_layer_keys_dataset.write(&current_layer_reward.keys)?;
        let current_layer_values_dataset = current_layer_group
            .new_dataset::<f32>()
            .shape(current_layer_reward.values.shape())
            .create("values")?;
        current_layer_values_dataset.write(&current_layer_reward.values)?;

        previous_layer_reward = current_layer_reward;
        exploration_set = next_exploration_set;
    }

    Ok(())
}

fn calculate_score_state_reward(
    score_state: ScoreState,
    previous_layer_reward: &IntKeyedArrayMap,
    all_dice_states: &[Array1<u8>],
    all_keep_actions: &[Array1<bool>],
    transition_function: &Array3<f32>,
) -> Array2<f32> {
    // Calculate the reward of a ScoreState, and return it as a ndarray
    // of shape (NUM_ROLLS, num_dice_states).
    let num_dice_states: usize = all_dice_states.len();
    let num_keep_actions: usize = all_keep_actions.len();
    let mut score_state_reward = Array2::zeros((NUM_ROLLS, num_dice_states));

    let keep_none_action = array![false, false, false, false, false];
    let keep_none_action_index = all_keep_actions
        .iter()
        .position(|x| x == keep_none_action)
        .expect("all_keep_actions should contain keep_none_action");

    // The probablity of rolling any state by rerolling all dices
    let first_roll_probability: ArrayView1<f32> =
        transition_function.slice(s![0, keep_none_action_index, ..]);
    // 0 reroll, the reward is the
    // Reward(ScoreState, DiceState, ScoreAction)
    // + Sum of (
    //   ProbFirstRoll * Reward(ChildScoreState, FirstRollDiceState, Reroll=2)
    // ) over all first roll dice states
    // Maximize over the possible actions to get
    // Reward(ScoreState, DiceState, Reroll=0)
    let score_actions = score_state.possible_score_actions();
    for (dice_state_index, dice_state) in all_dice_states.iter().enumerate() {
        let mut max_reward: f32 = 0.0;
        for score_action in score_actions.iter() {
            let action_reward = score_state.reward(*score_action, dice_state);
            let child_score_state = score_state
                .apply_action(*score_action, dice_state)
                .expect("possible_score_actions should only return valid actions");
            let child_score_state_index: u32 = child_score_state.into();
            let all_child_rewards = previous_layer_reward.get(child_score_state_index).expect(
                "previous_layer_reward should contain all ScoreStates reachable from ScoreState",
            );
            let child_reward: f32 = first_roll_probability.dot(&all_child_rewards.slice(s![2, ..]));
            max_reward = max_reward.max(action_reward as f32 + child_reward);
        }
        score_state_reward[[0, dice_state_index]] = max_reward;
    }

    // 1 and 2 reroll, the reward is the
    // Sum of (
    //   TransitionProbability(DiceState, KeepAction, ToDiceState)
    //   * Reward(ScoreState, ToDiceState, Reroll - 1)
    // ) over all ToDiceStates

    // Maximize over the possible KeepAction to get
    // Reward(ScoreState, DiceState, Reroll)
    for reroll in 1..NUM_ROLLS {
        for dice_state_index in 0..num_dice_states {
            let mut max_reward: f32 = 0.0;
            for keep_action_index in 0..num_keep_actions {
                let keep_probability: ArrayView1<f32> =
                    transition_function.slice(s![dice_state_index, keep_action_index, ..]);
                let keep_reward =
                    keep_probability.dot(&score_state_reward.slice(s![reroll - 1, ..]));
                max_reward = max_reward.max(keep_reward)
            }
            score_state_reward[[reroll, dice_state_index]] = max_reward;
        }
    }
    score_state_reward
}
