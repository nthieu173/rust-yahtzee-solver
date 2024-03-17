use crate::dice_states::NUM_DICES;
use ndarray::Array1;
use std::{convert::From, fmt::Display};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum ScoreAction {
    // The number here is also the number of bit shifted in ScoreState
    Ones = 0,
    Twos = 1,
    Threes = 2,
    Fours = 3,
    Fives = 4,
    Sixes = 5,
    ThreeOfAKind = 6,
    FourOfAKind = 7,
    FullHouse = 8,
    SmallStraight = 9,
    LargeStraight = 10,
    Chance = 11,
    Yahtzee = 12,
}
const UPPER_SCORE_SHIFT: u8 = ScoreAction::Yahtzee as u8 + 4;

const MAX_UPPER_SCORE: u8 = 5 + 10 + 15 + 20 + 25 + 30; // 105
const UPPER_SCORE_THRESHOLD: u8 = 63;
const UPPER_SCORE_BONUS: u8 = 35;

impl ScoreAction {
    pub fn all() -> [Self; 13] {
        [
            Self::Ones,
            Self::Twos,
            Self::Threes,
            Self::Fours,
            Self::Fives,
            Self::Sixes,
            Self::ThreeOfAKind,
            Self::FourOfAKind,
            Self::FullHouse,
            Self::SmallStraight,
            Self::LargeStraight,
            Self::Chance,
            Self::Yahtzee,
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScoreState {
    state: u32,
}

impl ScoreState {
    pub fn new(
        upper_score: u8,       // 0-105: 7 bits
        yahtzee: Option<u8>,   // 0-13 taken yahtzee reward, 14 not taken, 15 not applicable: 4 bits
        ones: bool,            // 0-1: 1 bit
        twos: bool,            // 0-1: 1 bit
        threes: bool,          // 0-1: 1 bit
        fours: bool,           // 0-1: 1 bit
        fives: bool,           // 0-1: 1 bit
        sixes: bool,           // 0-1: 1 bit
        three_of_a_kind: bool, // 0-1: 1 bit
        four_of_a_kind: bool,  // 0-1: 1 bit
        full_house: bool,      // 0-1: 1 bit
        small_straight: bool,  // 0-1: 1 bit
        large_straight: bool,  // 0-1: 1 bit
        chance: bool,          // 0-1: 1 bit
    ) -> Self {
        let mut state: u32 = 0;
        state |= (ones as u32) << ScoreAction::Ones as u8;
        state |= (twos as u32) << ScoreAction::Twos as u8;
        state |= (threes as u32) << ScoreAction::Threes as u8;
        state |= (fours as u32) << ScoreAction::Fours as u8;
        state |= (fives as u32) << ScoreAction::Fives as u8;
        state |= (sixes as u32) << ScoreAction::Sixes as u8;
        state |= (three_of_a_kind as u32) << ScoreAction::ThreeOfAKind as u8;
        state |= (four_of_a_kind as u32) << ScoreAction::FourOfAKind as u8;
        state |= (full_house as u32) << ScoreAction::FullHouse as u8;
        state |= (small_straight as u32) << ScoreAction::SmallStraight as u8;
        state |= (large_straight as u32) << ScoreAction::LargeStraight as u8;
        state |= (chance as u32) << ScoreAction::Chance as u8;
        let yahtzee = yahtzee.unwrap_or(14);
        if yahtzee > 14 {
            panic!("Yahtzee count must be 0-13, 14 is also not taken");
        } else {
            state |= (yahtzee as u32) << ScoreAction::Yahtzee as u8;
        }
        if upper_score > MAX_UPPER_SCORE {
            panic!("Upper sum must be 0-105");
        } else {
            state |= (upper_score as u32) << UPPER_SCORE_SHIFT
        }
        Self { state }
    }
    pub fn empty() -> Self {
        Self::new(
            0, None, false, false, false, false, false, false, false, false, false, false, false,
            false,
        )
    }
    pub fn is_taken(&self, score_action: ScoreAction) -> bool {
        if score_action != ScoreAction::Yahtzee {
            self.state & (0b1 << score_action as u8) != 0
        } else {
            self.state & (0b1111 << score_action as u8) < 14
        }
    }
    pub fn set_taken(&mut self, score_action: ScoreAction) {
        if score_action != ScoreAction::Yahtzee {
            self.state |= 0b1 << score_action as u8;
        } else {
            self.state |= 0b1111 << 14;
        }
    }
    pub fn upper_score(&self) -> u8 {
        ((self.state & (0b1111111 << UPPER_SCORE_SHIFT)) >> UPPER_SCORE_SHIFT) as u8
    }
    pub fn set_upper_score(&mut self, upper_score: u8) {
        if upper_score > MAX_UPPER_SCORE {
            panic!("Upper sum must be 0-105");
        }
        self.state |= (upper_score as u32) << UPPER_SCORE_SHIFT;
    }
    pub fn apply_action(&self, score_action: ScoreAction, dice_state: &Array1<u8>) -> Option<Self> {
        let mut new_state = Self { state: self.state };
        if self.is_taken(score_action) {
            return None;
        }
        new_state.set_taken(score_action);
        match score_action {
            ScoreAction::Ones
            | ScoreAction::Twos
            | ScoreAction::Threes
            | ScoreAction::Fours
            | ScoreAction::Fives
            | ScoreAction::Sixes => {
                let score = self.upper_score()
                    + (dice_state[score_action as usize] as u8) * (score_action as u8 + 1);
                new_state.set_upper_score(score);
            }
            _ => (),
        }
        Some(new_state)
    }
    pub fn score(&self) -> u16 {
        let mut score = self.upper_score() as u16;
        if score >= UPPER_SCORE_THRESHOLD as u16 {
            score += UPPER_SCORE_BONUS as u16;
        }

        score
    }
    pub fn possible_score_actions(&self) -> Vec<ScoreAction> {
        let mut possible_score_actions = Vec::new();
        if self.ones().is_none() {
            possible_score_actions.push(ScoreAction::Ones);
        }
        if self.twos().is_none() {
            possible_score_actions.push(ScoreAction::Twos);
        }
        if self.threes().is_none() {
            possible_score_actions.push(ScoreAction::Threes);
        }
        if self.fours().is_none() {
            possible_score_actions.push(ScoreAction::Fours);
        }
        if self.fives().is_none() {
            possible_score_actions.push(ScoreAction::Fives);
        }
        if self.sixes().is_none() {
            possible_score_actions.push(ScoreAction::Sixes);
        }
        if !self.three_of_a_kind() {
            possible_score_actions.push(ScoreAction::ThreeOfAKind);
        }
        if !self.four_of_a_kind() {
            possible_score_actions.push(ScoreAction::FourOfAKind);
        }
        if !self.full_house() {
            possible_score_actions.push(ScoreAction::FullHouse);
        }
        if !self.small_straight() {
            possible_score_actions.push(ScoreAction::SmallStraight);
        }
        if !self.large_straight() {
            possible_score_actions.push(ScoreAction::LargeStraight);
        }
        if !self.chance() {
            possible_score_actions.push(ScoreAction::Chance);
        }
        if self.yahtzee().is_none() {
            possible_score_actions.push(ScoreAction::Yahtzee);
        }
        possible_score_actions
    }
    pub fn get_parent_states(&self) -> Vec<Self> {
        let mut parent_states = Vec::new();
        if let Some(_) = self.ones() {
            parent_states.push(Self {
                state: self.state | (0b111 << 23),
            });
        }
        if let Some(_) = self.twos() {
            parent_states.push(Self {
                state: self.state | (0b111 << 20),
            });
        }
        if let Some(_) = self.threes() {
            parent_states.push(Self {
                state: self.state | (0b111 << 17),
            });
        }
        if let Some(_) = self.fours() {
            parent_states.push(Self {
                state: self.state | (0b111 << 14),
            });
        }
        if let Some(_) = self.fives() {
            parent_states.push(Self {
                state: self.state | (0b111 << 11),
            });
        }
        if let Some(_) = self.sixes() {
            parent_states.push(Self {
                state: self.state | (0b111 << 8),
            });
        }
        if self.three_of_a_kind() {
            parent_states.push(Self {
                state: self.state & !(0b1 << 7),
            });
        }
        if self.four_of_a_kind() {
            parent_states.push(Self {
                state: self.state & !(0b1 << 6),
            });
        }
        if self.full_house() {
            parent_states.push(Self {
                state: self.state & !(0b1 << 5),
            });
        }
        if self.small_straight() {
            parent_states.push(Self {
                state: self.state & !(0b1 << 4),
            });
        }
        if self.large_straight() {
            parent_states.push(Self {
                state: self.state & !(0b1 << 3),
            });
        }
        if self.chance() {
            parent_states.push(Self {
                state: self.state & !(0b1 << 2),
            });
        }
        if let Some(yahtzee) = self.yahtzee() {
            if yahtzee {
                parent_states.push(Self {
                    state: (self.state & !(0b11)) | 0b10,
                });
            }
        }
        parent_states
    }
    pub fn get_all_terminal_states() -> Vec<ScoreState> {
        let mut terminal_states = Vec::new();
        for ones in 0..=NUM_DICES {
            for twos in 0..=NUM_DICES {
                for threes in 0..=NUM_DICES {
                    for fours in 0..=NUM_DICES {
                        for fives in 0..=NUM_DICES {
                            for sixes in 0..=NUM_DICES {
                                for yahtzee in 0..=13 {
                                    terminal_states.push(Self::new(
                                        (ones
                                            + twos * 2
                                            + threes * 3
                                            + fours * 4
                                            + fives * 5
                                            + sixes * 6)
                                            as u8,
                                        Some(yahtzee),
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                        false,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        terminal_states
    }
    fn three_of_a_kind_reward(&self, dice_state: &Array1<u8>) -> u16 {
        if !self.three_of_a_kind() {
            return 0;
        }
        if let Some(yahtzee) = self.yahtzee() {
            if is_yahtzee(dice_state) {
                let reward: u16 = dice_state.iter().map(|&x| x as u16).sum();
                if yahtzee {
                    // If the dice state is yahtzee and
                    // the score state has already taken yahtzee
                    // with a non-zero reward, then we add 100
                    return reward + 100;
                } else {
                    return reward;
                }
            }
        }
        if is_three_of_a_kind(dice_state) {
            dice_state.iter().map(|&x| x as u16).sum()
        } else {
            0
        }
    }
    fn four_of_a_kind_reward(&self, dice_state: &Array1<u8>) -> u16 {
        if !self.four_of_a_kind() {
            return 0;
        }
        if let Some(yahtzee) = self.yahtzee() {
            if is_yahtzee(dice_state) {
                let reward: u16 = dice_state.iter().map(|&x| x as u16).sum();
                if yahtzee {
                    // If the dice state is yahtzee and
                    // the score state has already taken yahtzee
                    // with a non-zero reward, then we add 100
                    return reward + 100;
                } else {
                    return reward;
                }
            }
        }
        if is_four_of_a_kind(dice_state) {
            dice_state.iter().map(|&x| x as u16).sum()
        } else {
            0
        }
    }
    fn full_house_reward(&self, dice_state: &Array1<u8>) -> u16 {
        if !self.full_house() {
            return 0;
        }
        let reward: u16 = 25;
        if let Some(yahtzee) = self.yahtzee() {
            if is_yahtzee(dice_state) {
                // Yahtzee is a wild card
                if yahtzee {
                    // If the dice state is yahtzee and
                    // the score state has already taken yahtzee
                    // with a non-zero reward, then we add 100
                    return reward + 100;
                } else {
                    // If the taken yahtzee reward is zero, then
                    // we don't gain extra 100 points.
                    return reward;
                }
            }
        }
        if is_full_house(dice_state) {
            reward
        } else {
            0
        }
    }
    fn small_straight_reward(&self, dice_state: &Array1<u8>) -> u16 {
        if !self.small_straight() {
            return 0;
        }
        let reward: u16 = 30;
        if let Some(yahtzee) = self.yahtzee() {
            if is_yahtzee(dice_state) {
                // Yahtzee is a wild card
                if yahtzee {
                    // If the dice state is yahtzee and
                    // the score state has already taken yahtzee
                    // with a non-zero reward, then we add 100
                    return reward + 100;
                } else {
                    // If the taken yahtzee reward is zero, then
                    // we don't gain extra 100 points.
                    return reward;
                }
            }
        }
        if is_small_straight(dice_state) {
            reward
        } else {
            0
        }
    }
    fn large_straight_reward(&self, dice_state: &Array1<u8>) -> u16 {
        if !self.large_straight() {
            return 0;
        }
        let reward: u16 = 40;
        if let Some(yahtzee) = self.yahtzee() {
            if is_yahtzee(dice_state) {
                // Yahtzee is a wild card
                if yahtzee {
                    // If the dice state is yahtzee and
                    // the score state has already taken yahtzee
                    // with a non-zero reward, then we add 100
                    return reward + 100;
                } else {
                    // If the taken yahtzee reward is zero, then
                    // we don't gain extra 100 points.
                    return reward;
                }
            }
        }
        if is_large_straight(dice_state) {
            reward
        } else {
            0
        }
    }
    fn chance_reward(&self, dice_state: &Array1<u8>) -> u16 {
        if !self.chance() {
            return 0;
        }
        dice_state.iter().map(|&x| x as u16).sum()
    }
    fn yahtzee_reward(&self, dice_state: &Array1<u8>) -> u16 {
        if self.yahtzee().is_none() {
            return 0;
        }
        if is_yahtzee(dice_state) {
            50
        } else {
            0
        }
    }
    fn sum_of_upper(&self) -> Option<u8> {
        let mut sum = 0;
        if let Some(val) = self.ones() {
            sum += val;
        } else {
            return None;
        }
        if let Some(val) = self.twos() {
            sum += val;
        } else {
            return None;
        }
        if let Some(val) = self.threes() {
            sum += val;
        } else {
            return None;
        }
        if let Some(val) = self.fours() {
            sum += val;
        } else {
            return None;
        }
        if let Some(val) = self.fives() {
            sum += val;
        } else {
            return None;
        }
        if let Some(val) = self.sixes() {
            sum += val;
        } else {
            return None;
        }
        Some(sum)
    }
}

impl Display for ScoreState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut data = "ScoreState(".to_string();
        if let Some(val) = self.ones() {
            data += &format!("ones: {}, ", val);
        } else {
            data += "ones: None, ";
        }
        if let Some(val) = self.twos() {
            data += &format!("twos: {}, ", val);
        } else {
            data += "twos: None, ";
        }
        if let Some(val) = self.threes() {
            data += &format!("threes: {}, ", val);
        } else {
            data += "threes: None, ";
        }
        if let Some(val) = self.fours() {
            data += &format!("fours: {}, ", val);
        } else {
            data += "fours: None, ";
        }
        if let Some(val) = self.fives() {
            data += &format!("fives: {}, ", val);
        } else {
            data += "fives: None, ";
        }
        if let Some(val) = self.sixes() {
            data += &format!("sixes: {}, ", val);
        } else {
            data += "sixes: None, ";
        }
        data += &format!("three_of_a_kind: {}, ", self.three_of_a_kind());
        data += &format!("four_of_a_kind: {}, ", self.four_of_a_kind());
        data += &format!("full_house: {}, ", self.full_house());
        data += &format!("small_straight: {}, ", self.small_straight());
        data += &format!("large_straight: {}, ", self.large_straight());
        data += &format!("chance: {}, ", self.chance());
        if let Some(val) = self.yahtzee() {
            data += &format!("yahtzee: {}", val);
        } else {
            data += "yahtzee: None";
        }
        data += ")";
        f.write_str(&data)
    }
}

impl From<u32> for ScoreState {
    fn from(state: u32) -> Self {
        Self { state }
    }
}

impl From<ScoreState> for u32 {
    fn from(score_state: ScoreState) -> Self {
        score_state.state
    }
}

fn is_three_of_a_kind(dice_state: &Array1<u8>) -> bool {
    for num_dice in dice_state.iter() {
        if *num_dice >= 3 {
            return true;
        }
    }
    false
}

fn is_four_of_a_kind(dice_state: &Array1<u8>) -> bool {
    for num_dice in dice_state.iter() {
        if *num_dice >= 4 {
            return true;
        }
    }
    false
}

fn is_full_house(dice_state: &Array1<u8>) -> bool {
    let mut found_two = false;
    let mut found_three = false;
    for num_dice in dice_state.iter() {
        if *num_dice == 2 {
            found_two = true;
        } else if *num_dice == 3 {
            found_three = true;
        }
    }
    found_two && found_three
}

fn is_small_straight(dice_state: &Array1<u8>) -> bool {
    for shift in 0..=NUM_DICES - 4 {
        let mut found = true;
        for i in 0..4 {
            if dice_state[shift + i] == 0 {
                found = false;
                break;
            }
        }
        if found {
            return true;
        }
    }
    false
}

fn is_large_straight(dice_state: &Array1<u8>) -> bool {
    dice_state.iter().all(|&x| x == 1)
}

fn is_yahtzee(dice_state: &Array1<u8>) -> bool {
    dice_state.iter().all(|&x| x == NUM_DICES as u8)
}
