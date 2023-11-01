use crate::dice_states::NUM_DICES;
use ndarray::Array1;
use std::{convert::From, fmt::Display};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Ord, PartialOrd)]
pub enum ScoreAction {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScoreState {
    state: u32,
}

impl ScoreState {
    pub fn new(
        ones: Option<u8>,
        twos: Option<u8>,
        threes: Option<u8>,
        fours: Option<u8>,
        fives: Option<u8>,
        sixes: Option<u8>,
        three_of_a_kind: bool,
        four_of_a_kind: bool,
        full_house: bool,
        small_straight: bool,
        large_straight: bool,
        chance: bool,
        yahtzee: Option<bool>,
    ) -> Self {
        let mut state: u32 = 0;
        if let Some(ones) = ones {
            state |= (ones as u32 & 0b111) << 23;
        } else {
            state |= 0b111 << 23;
        }
        if let Some(twos) = twos {
            state |= (twos as u32 & 0b111) << 20;
        } else {
            state |= 0b111 << 20;
        }
        if let Some(threes) = threes {
            state |= (threes as u32 & 0b111) << 17;
        } else {
            state |= 0b111 << 17;
        }
        if let Some(fours) = fours {
            state |= (fours as u32 & 0b111) << 14;
        } else {
            state |= 0b111 << 14;
        }
        if let Some(fives) = fives {
            state |= (fives as u32 & 0b111) << 11;
        } else {
            state |= 0b111 << 11;
        }
        if let Some(sixes) = sixes {
            state |= (sixes as u32 & 0b111) << 8;
        } else {
            state |= 0b111 << 8;
        }
        if three_of_a_kind {
            state |= 0b1 << 7;
        }
        if four_of_a_kind {
            state |= 0b1 << 6;
        }
        if full_house {
            state |= 0b1 << 5;
        }
        if small_straight {
            state |= 0b1 << 4;
        }
        if large_straight {
            state |= 0b1 << 3;
        }
        if chance {
            state |= 0b1 << 2;
        }
        if let Some(yahtzee) = yahtzee {
            if yahtzee {
                state |= 0b01;
            }
        } else {
            state |= 0b11;
        }
        Self { state }
    }
    pub fn num_all_states() -> u32 {
        // We have 26 bits to represent the score state.
        // the last two bits are for yahtzee, where 3 is not taken
        return 1 << 27 - 2;
    }
    pub fn apply_action(&self, score_action: ScoreAction, dice_state: &Array1<u8>) -> Option<Self> {
        let mut new_state = self.state;
        match score_action {
            ScoreAction::Ones => {
                if self.ones().is_some() {
                    return None;
                }
                // Zero out the ones bits.
                new_state &= !(0b111 << 23);
                // Set the ones bits.
                new_state |= (dice_state[0] as u32) << 23;
            }
            ScoreAction::Twos => {
                if self.twos().is_some() {
                    return None;
                }
                // Zero out the twos bits.
                new_state &= !(0b111 << 20);
                // Set the twos bits.
                new_state |= (dice_state[1] as u32) << 20;
            }
            ScoreAction::Threes => {
                if self.threes().is_some() {
                    return None;
                }
                // Zero out the threes bits.
                new_state &= !(0b111 << 17);
                // Set the threes bits.
                new_state |= (dice_state[2] as u32) << 17;
            }
            ScoreAction::Fours => {
                if self.fours().is_some() {
                    return None;
                }
                // Zero out the fours bits.
                new_state &= !(0b111 << 14);
                // Set the fours bits.
                new_state |= (dice_state[3] as u32) << 14;
            }
            ScoreAction::Fives => {
                if self.fives().is_some() {
                    return None;
                }
                // Zero out the fives bits.
                new_state &= !(0b111 << 11);
                // Set the fives bits.
                new_state |= (dice_state[4] as u32) << 11;
            }
            ScoreAction::Sixes => {
                if self.sixes().is_some() {
                    return None;
                }
                // Zero out the sixes bits.
                new_state &= !(0b111 << 8);
                // Set the sixes bits.
                new_state |= (dice_state[5] as u32) << 8;
            }
            ScoreAction::ThreeOfAKind => {
                if self.three_of_a_kind() {
                    return None;
                }
                // Set the three of a kind bit.
                new_state |= 0b1 << 7;
            }
            ScoreAction::FourOfAKind => {
                if self.four_of_a_kind() {
                    return None;
                }
                // Set the four of a kind bit.
                new_state |= 0b1 << 6;
            }
            ScoreAction::FullHouse => {
                if self.full_house() {
                    return None;
                }
                // Set the full house bit.
                new_state |= 0b1 << 5;
            }
            ScoreAction::SmallStraight => {
                if self.small_straight() {
                    return None;
                }
                // Set the small straight bit.
                new_state |= 0b1 << 4;
            }
            ScoreAction::LargeStraight => {
                if self.large_straight() {
                    return None;
                }
                // Set the large straight bit.
                new_state |= 0b1 << 3;
            }
            ScoreAction::Chance => {
                if self.chance() {
                    return None;
                }
                // Set the chance bit.
                new_state |= 0b1 << 2;
            }
            ScoreAction::Yahtzee => {
                if self.yahtzee().is_some() {
                    return None;
                }
                // Set the yahtzee bits if
                // the dice state is yahtzee.
                if is_yahtzee(dice_state) {
                    new_state |= 0b01;
                } else {
                    new_state &= !(0b11);
                }
            }
        }
        Some(Self { state: new_state })
    }
    pub fn ones(&self) -> Option<u8> {
        let val = ((self.state & (0b111 << 23)) >> 23) as u8;
        match val {
            0b111 => None,
            _ => Some(val),
        }
    }
    pub fn twos(&self) -> Option<u8> {
        let val = ((self.state & (0b111 << 20)) >> 20) as u8;
        match val {
            0b111 => None,
            _ => Some(val),
        }
    }
    pub fn threes(&self) -> Option<u8> {
        let val = ((self.state & (0b111 << 17)) >> 17) as u8;
        match val {
            0b111 => None,
            _ => Some(val),
        }
    }
    pub fn fours(&self) -> Option<u8> {
        let val = ((self.state & (0b111 << 14)) >> 14) as u8;
        match val {
            0b111 => None,
            _ => Some(val),
        }
    }
    pub fn fives(&self) -> Option<u8> {
        let val = ((self.state & (0b111 << 11)) >> 11) as u8;
        match val {
            0b111 => None,
            _ => Some(val),
        }
    }
    pub fn sixes(&self) -> Option<u8> {
        let val = ((self.state & (0b111 << 8)) >> 8) as u8;
        match val {
            0b111 => None,
            _ => Some(val),
        }
    }
    pub fn three_of_a_kind(&self) -> bool {
        ((self.state & (0b1 << 7)) >> 7) == 1
    }
    pub fn four_of_a_kind(&self) -> bool {
        ((self.state & (0b1 << 6)) >> 6) == 1
    }
    pub fn full_house(&self) -> bool {
        ((self.state & (0b1 << 5)) >> 5) == 1
    }
    pub fn small_straight(&self) -> bool {
        ((self.state & (0b1 << 4)) >> 4) == 1
    }
    pub fn large_straight(&self) -> bool {
        ((self.state & (0b1 << 3)) >> 3) == 1
    }
    pub fn chance(&self) -> bool {
        ((self.state & (0b1 << 2)) >> 2) == 1
    }
    pub fn yahtzee(&self) -> Option<bool> {
        let val = (self.state & 0b11) as u8;
        match val {
            0b00 => Some(false),
            0b01 => Some(true),
            _ => None,
        }
    }
    pub fn reward(&self, score_action: ScoreAction, dice_state: &Array1<u8>) -> u16 {
        if let Some(next_state) = self.apply_action(score_action, dice_state) {
            match score_action {
                ScoreAction::ThreeOfAKind => self.three_of_a_kind_reward(dice_state),
                ScoreAction::FourOfAKind => self.four_of_a_kind_reward(dice_state),
                ScoreAction::FullHouse => self.full_house_reward(dice_state),
                ScoreAction::SmallStraight => self.small_straight_reward(dice_state),
                ScoreAction::LargeStraight => self.large_straight_reward(dice_state),
                ScoreAction::Chance => self.chance_reward(dice_state),
                ScoreAction::Yahtzee => self.yahtzee_reward(dice_state),
                simple_score_action => {
                    let reward = dice_state[simple_score_action as usize] as u16;
                    if let Some(upper_sum) = next_state.sum_of_upper() {
                        if upper_sum >= 63 {
                            reward + 35
                        } else {
                            reward
                        }
                    } else {
                        reward
                    }
                }
            }
        } else {
            0
        }
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
                                for yahtzee in vec![false, true].into_iter() {
                                    terminal_states.push(Self::new(
                                        Some(ones as u8),
                                        Some(twos as u8),
                                        Some(threes as u8),
                                        Some(fours as u8),
                                        Some(fives as u8),
                                        Some(sixes as u8),
                                        true,
                                        true,
                                        true,
                                        true,
                                        true,
                                        true,
                                        Some(yahtzee),
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
