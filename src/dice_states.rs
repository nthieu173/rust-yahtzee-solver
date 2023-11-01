use ndarray::parallel::prelude::*;
use ndarray::prelude::*;

pub const NUM_DICES: usize = 5;

pub fn get_transition_function(
    all_dice_states: &[Array1<u8>],
    all_keep_actions: &[Array1<bool>],
) -> Array3<f32> {
    // Generate the transition function when rolling NUM_DICES dice and keeping a subset of them.

    let num_dice_states = all_dice_states.len();
    let num_keep_actions = all_keep_actions.len();

    let mut transition_function: Array3<f32> =
        Array3::zeros((num_dice_states, num_keep_actions, num_dice_states));

    transition_function
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(index, mut view)| {
            for (keep_action_index, keep_action) in all_keep_actions.iter().enumerate() {
                for (next_dice_state_index, next_dice_state) in all_dice_states.iter().enumerate() {
                    view[[keep_action_index, next_dice_state_index]] = transition_probability(
                        &all_dice_states[index],
                        keep_action,
                        next_dice_state,
                    );
                }
            }
        });

    transition_function
}

pub fn get_all_dice_states() -> Vec<Array1<u8>> {
    let mut dice_states = Vec::new();
    for ones in 0..=NUM_DICES {
        for twos in 0..=NUM_DICES - ones {
            for threes in 0..=NUM_DICES - ones - twos {
                for fours in 0..=NUM_DICES - ones - twos - threes {
                    for fives in 0..=NUM_DICES - ones - twos - threes - fours {
                        let sixes = NUM_DICES - ones - twos - threes - fours - fives;
                        dice_states.push(array![
                            ones as u8,
                            twos as u8,
                            threes as u8,
                            fours as u8,
                            fives as u8,
                            sixes as u8
                        ]);
                    }
                }
            }
        }
    }
    dice_states
}

pub fn get_all_keep_actions() -> Vec<Array1<bool>> {
    let mut keep_actions = Vec::new();
    for first in 0..2 {
        for second in 0..2 {
            for third in 0..2 {
                for fourth in 0..2 {
                    for fifth in 0..2 {
                        keep_actions.push(array![
                            first == 1,
                            second == 1,
                            third == 1,
                            fourth == 1,
                            fifth == 1,
                        ]);
                    }
                }
            }
        }
    }
    keep_actions
}

fn transition_probability(
    dice_state: &Array1<u8>,
    keep_action: &Array1<bool>,
    next_dice_state: &Array1<u8>,
) -> f32 {
    // Given a state, action, and next state, return the probability of transitioning from the
    // given state to the given next state given the given action.

    // Get kept dices.
    let kept = action_to_kept_array(dice_state, keep_action);

    // If any kept dice is greater than the corresponding next dice, then the transition
    // is impossible (you can't go from (2, 3, 1, 0, 0, 0) to (1, 3, 1, 0, 0, 0) by keeping
    // the 2 of the first dice
    if kept
        .iter()
        .zip(next_dice_state.iter())
        .any(|(&x, &y)| x > y)
    {
        return 0.0;
    }

    // Get goal reroll dices.
    let goal_reroll = next_dice_state - kept;

    // If there are no goal reroll dices, then the transition is certain.
    if goal_reroll.iter().all(|&x| x == 0) {
        return 1.0;
    }

    // Else, we need to calculate the probability of rolling the goal reroll dices
    probability_of_goal_roll(&goal_reroll)
}

fn action_to_kept_array(dice_state: &Array1<u8>, keep_action: &Array1<bool>) -> Array1<u8> {
    // Given a state and an action, return the state that results from keeping the dice
    // specified by the action.

    // For example, if state = (2, 3, 1, 0, 0, 0) and action = (1, 0, 1, 0, 0, 0), then
    // action_to_kept_tuple(state, action) = (1, 1, 0, 0, 0, 0).
    let mut action_index = 0;
    let mut kept = array![0, 0, 0, 0, 0, 0];

    for (state_index, num_dice) in dice_state.iter().enumerate() {
        for _ in 0..*num_dice {
            if keep_action[action_index] {
                kept[state_index] += 1;
            }
            action_index += 1;
            if action_index >= keep_action.len() {
                break;
            }
        }
        if action_index >= keep_action.len() {
            break;
        }
    }
    kept
}

fn probability_of_goal_roll(goal_roll: &Array1<u8>) -> f32 {
    // Possibility of rolling sum(goal_roll) dice and
    // getting the desired positive goal_roll values.

    // For example, if goal_roll = [1, 1, 0, 0, 0, 0], then
    // we are trying to roll 2 dice and get exactly one 1 and one 2.
    let num_rolls = goal_roll.sum();

    // The total number of ways to roll num_rolls dice is 6^num_rolls.
    let total_num_rolls = u32::pow(6, num_rolls as u32);

    // The total number of ways to roll num_rolls dice and get exactly
    // the positive_goal_rolls values is the multinomial coefficient
    let total_accepted_rolls = multinomial_coefficient(goal_roll);

    total_accepted_rolls as f32 / total_num_rolls as f32
}

fn multinomial_coefficient(all_k: &Array1<u8>) -> u64 {
    let mut result = 1;
    let mut cum_k = 0;
    for k in all_k.iter() {
        cum_k += *k;
        result *= binomial_coefficient(cum_k as u64, *k as u64);
    }
    result
}

fn binomial_coefficient(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }
    let k = k.min(n - k); // Take advantage of symmetry
    let mut c = 1;
    for i in 0..k {
        c = c * (n - i) / (i + 1);
    }
    c
}
